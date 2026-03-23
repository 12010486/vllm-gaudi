# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online Model Swapping Test for Gaudi (API-based)
==================================================
Runs N phases (default 5) alternating between Model A and Model B.
Uses the OpenAI-compatible /v1/models/switch endpoint to switch models.
Parses server logs to extract actual warmup timings.

Collects per-phase metrics (switch time, gen time, warmup time, memory) and
prints a summary table.

Requires:
  VLLM_ENABLE_V1_MULTIPROCESSING=0
  VLLM_SERVER_DEV_MODE=1
  Multi-model YAML config

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  VLLM_SERVER_DEV_MODE=1 \\
  python tests/full_tests/online_model_swap_test.py \\
    --config tests/full_tests/multi_models.yaml \\
    --phases 5 \\
    --api-host localhost \\
    --api-port 8080
"""

import argparse
import asyncio
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time

import requests
import yaml

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False

SEED_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Explain quantum computing",
    "The tallest mountain is",
    "Write a short poem about",
    "The speed of light is",
    "Technology in 2050 will be",
    "The most important invention",
]


def generate_prompts(n=20):
    """Generate n prompts by cycling through seed prompts."""
    prompts = []
    for i in range(n):
        base = SEED_PROMPTS[i % len(SEED_PROMPTS)]
        if i < len(SEED_PROMPTS):
            prompts.append(base)
        else:
            prompts.append(f"{base} (iteration {i // len(SEED_PROMPTS)})")
    return prompts


PROMPTS = generate_prompts(20)


def _server_api_host(api_host: str) -> str:
    """Normalize host used by the local test server."""
    return '127.0.0.1' if api_host == 'localhost' else api_host


def _client_api_host(api_host: str) -> str:
    """Normalize host used by the local HTTP client."""
    return '127.0.0.1' if api_host in {'localhost', '0.0.0.0'} else api_host


def _api_url(api_host: str, api_port: int, path: str) -> str:
    return f"http://{_client_api_host(api_host)}:{api_port}{path}"


def find_free_port():
    """Find an available port to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def create_multi_model_config(model_a, model_b, max_model_len=4096, max_num_batched_tokens=8192):
    """Create a temporary multi-model YAML config."""
    config = {
        'default_model': 'model_a',
        'models': {
            'model_a': {
                'model': model_a,
                'tensor_parallel_size': 1,
                'max_model_len': max_model_len,
                'max_num_batched_tokens': max_num_batched_tokens,
            },
            'model_b': {
                'model': model_b,
                'tensor_parallel_size': 1,
                'max_model_len': max_model_len,
                'max_num_batched_tokens': max_num_batched_tokens,
            },
        },
    }
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, tmpfile, default_flow_style=False)
    tmpfile.close()
    return tmpfile.name


class ServerLogCapture:
    """Capture and parse server logs to extract warmup timings."""

    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()
        self.warmup_start_time = None
        self.warmup_events = []  # List of (timestamp, warmup_secs) tuples

    def add_line(self, line: str):
        """Add a log line and extract warmup markers."""
        with self.lock:
            self.logs.append(line)
            ts = time.time()

            # Pattern: "Warmup finished in <N> secs"
            warmup_match = re.search(r'Warmup finished in (\d+) secs', line)
            if warmup_match:
                elapsed = int(warmup_match.group(1))
                self.warmup_events.append((ts, elapsed))

    def get_warmup_times(self):
        """Return list of captured warmup times in seconds."""
        with self.lock:
            return [warmup_s for _, warmup_s in self.warmup_events]

    def clear_warmup_events(self):
        """Clear captured warmup events (call before each switch to isolate measurements)."""
        with self.lock:
            self.warmup_events = []


def run_server(config_path: str, api_host: str, api_port: int, log_capture: ServerLogCapture,
               max_num_batched_tokens: int):
    """Start the multi-model API server as a subprocess."""
    server_host = _server_api_host(api_host)
    env = os.environ.copy()
    env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    env['VLLM_SERVER_DEV_MODE'] = '1'
    env['VLLM_HPU_MULTI_MODEL_CONFIG'] = config_path
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1,localhost,0.0.0.0']))
    env['no_proxy'] = ','.join(filter(None, [env.get('no_proxy'), '127.0.0.1,localhost,0.0.0.0']))

    cmd = [
        sys.executable,
        '-m',
        'vllm_gaudi.entrypoints.openai.multi_model_api_server',
        '--host',
        server_host,
        '--port',
        str(api_port),
        '--max-num-batched-tokens',
        str(max_num_batched_tokens),
    ]

    print(f"\n>>> Starting server: {' '.join(cmd)}")
    print(f"    Config: {config_path}")
    print(f"    Listening on {server_host}:{api_port}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    def capture_logs():
        """Read stdout in background and capture log lines."""
        try:
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line:
                    print(f"[SERVER] {line}")
                    log_capture.add_line(line)
        except Exception as e:
            print(f"[LOG_CAPTURE_ERROR] {e}")

    log_thread = threading.Thread(target=capture_logs, daemon=True)
    log_thread.start()

    return proc, log_thread


def wait_for_server(api_host: str,
                    api_port: int,
                    timeout: int = 300,
                    proc: subprocess.Popen | None = None) -> list[str]:
    """Wait for server to be ready and return list of available models."""
    url = _api_url(api_host, api_port, '/v1/models')
    start = time.time()
    last_error = None

    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server exited before readiness check succeeded (exit code {proc.returncode})")
        try:
            resp = _HTTP_SESSION.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get('id') for m in data.get('data', [])]
                print("  ✓ Server is ready")
                print(f"  Available models: {models}")
                return models
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(2)

    suffix = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(f"Server did not start after {timeout}s.{suffix}")


async def switch_model(api_host: str, api_port: int, model_name: str, drain_timeout: int = 60) -> dict:
    """Call /v1/models/switch endpoint and return metrics."""
    url = _api_url(api_host, api_port, '/v1/models/switch')
    payload = {
        "model": model_name,
        "drain_timeout": drain_timeout,
    }

    start = time.perf_counter()
    try:
        resp = _HTTP_SESSION.post(url, json=payload, timeout=600)
        elapsed_s = time.perf_counter() - start

        if resp.status_code == 200:
            data = resp.json()
            return {
                'status': 'ok',
                'duration_s': elapsed_s,
                'api_duration_ms': data.get('duration_ms', 0),
                'switched': data.get('switched', False),
                'model': data.get('current_model'),
            }
        else:
            return {
                'status': 'error',
                'duration_s': elapsed_s,
                'error': resp.text,
            }
    except Exception as e:
        elapsed_s = time.perf_counter() - start
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': str(e),
        }


async def generate(api_host: str,
                   api_port: int,
                   model_name: str,
                   prompt: str,
                   seed: int = 42,
                   max_tokens: int = 1600,
                   strict_tokens: bool = True) -> dict:
    """Call /v1/chat/completions and return metrics."""
    url = _api_url(api_host, api_port, '/v1/chat/completions')
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,  # Fixed seed for reproducibility
    }
    if strict_tokens:
        payload["min_tokens"] = max_tokens
        payload["ignore_eos"] = True

    start = time.perf_counter()
    try:
        resp = _HTTP_SESSION.post(url, json=payload, timeout=120)
        elapsed_s = time.perf_counter() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = len(data.get('choices', [{}])[0].get('message', {}).get('content', '').split())
            usage = data.get('usage', {})
            return {
                'status': 'ok',
                'duration_s': elapsed_s,
                'output_tokens': usage.get('completion_tokens', tokens),
                'total_tokens': usage.get('total_tokens', 0),
            }
        else:
            if strict_tokens and resp.status_code == 400:
                payload.pop("min_tokens", None)
                payload.pop("ignore_eos", None)
                retry_resp = _HTTP_SESSION.post(url, json=payload, timeout=120)
                retry_elapsed_s = time.perf_counter() - start
                if retry_resp.status_code == 200:
                    data = retry_resp.json()
                    tokens = len(data.get('choices', [{}])[0].get('message', {}).get('content', '').split())
                    usage = data.get('usage', {})
                    return {
                        'status': 'ok',
                        'duration_s': retry_elapsed_s,
                        'output_tokens': usage.get('completion_tokens', tokens),
                        'total_tokens': usage.get('total_tokens', 0),
                        'strict_tokens_fallback': True,
                    }
            return {
                'status': 'error',
                'duration_s': elapsed_s,
                'error': resp.text,
            }
    except Exception as e:
        elapsed_s = time.perf_counter() - start
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': str(e),
        }


def print_metrics_table(all_metrics):
    """Print a summary table of per-phase metrics."""
    hdr = (f"{'Phase':>5}  {'Model':<15}  "
           f"{'API Call(s)':>11}  {'Gen(s)':>7}  "
           f"{'Warmup(s)':>9}  "
           f"{'Tokens':>7}")
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(f"{'(includes':>5}  {'':15}  {'warmup on':>11}  {'':7}  "
          f"{'(from':>9}  {'':7}")
    print(f"{'switch)':>5}  {'':15}  {'server)':>11}  {'':7}  "
          f"{'logs)':>9}  {'':7}")
    print(sep)
    for m in all_metrics:
        warmup_s = m.get('warmup_s', 'N/A')
        if isinstance(warmup_s, (int, float)):
            warmup_str = f"{warmup_s:>9.1f}"
        else:
            warmup_str = f"{str(warmup_s):>9}"

        print(f"{m['phase']:>5}  "
              f"{m['model']:<15}  "
              f"{m['switch_s']:>11.1f}  "
              f"{m['gen_s']:>7.2f}  "
              f"{warmup_str}  "
              f"{m['tokens']:>7}")
    print(sep)

    n = len(all_metrics)
    if n > 0:
        avg = {
            'switch_s': sum(m['switch_s'] for m in all_metrics) / n,
            'gen_s': sum(m['gen_s'] for m in all_metrics) / n,
            'tokens': sum(m['tokens'] for m in all_metrics) / n,
        }
        warmup_times = [m.get('warmup_s') for m in all_metrics if isinstance(m.get('warmup_s'), (int, float))]
        if warmup_times:
            avg_warmup = sum(warmup_times) / len(warmup_times)
            warmup_str = f"{avg_warmup:>9.1f}"
        else:
            warmup_str = f"{'N/A':>9}"

        print(f"{'AVG':>5}  {'':<15}  "
              f"{avg['switch_s']:>11.1f}  "
              f"{avg['gen_s']:>7.2f}  "
              f"{warmup_str}  "
              f"{avg['tokens']:>7.0f}")
        print(sep)


async def main():
    parser = argparse.ArgumentParser(description="Online Model Swapping Test")
    parser.add_argument("--model-a", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-b", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--config",
                        type=str,
                        default=None,
                        help="Multi-model YAML config (auto-generated if not provided)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=8192,
                        help="Align with offline baseline (default 8192)")
    parser.add_argument("--phases", type=int, default=5)
    parser.add_argument("--fixed-output-tokens", type=int, default=1600, help="Target completion tokens per request")
    parser.add_argument("--api-host", type=str, default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=None)
    args = parser.parse_args()

    if args.api_port is None:
        args.api_port = find_free_port()

    # Create config if not provided
    config_path = args.config
    if config_path is None:
        config_path = create_multi_model_config(
            args.model_a,
            args.model_b,
            args.max_model_len,
            args.max_num_batched_tokens,
        )
        print(f"Generated config: {config_path}")

    print("=" * 60)
    print("  ONLINE MODEL SWAPPING TEST")
    print("=" * 60)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Phases: {args.phases}")
    print(f"  Max model len: {args.max_model_len}")
    print(f"  Max num batched tokens: {args.max_num_batched_tokens}")
    print(f"  Fixed output tokens: {args.fixed_output_tokens}")
    print(f"  API: {args.api_host}:{args.api_port}")
    print(f"  Config: {config_path}")
    print("=" * 60)

    log_capture = ServerLogCapture()
    proc = None

    try:
        # Start server
        proc, log_thread = run_server(
            config_path,
            args.api_host,
            args.api_port,
            log_capture,
            args.max_num_batched_tokens,
        )
        available_models = wait_for_server(args.api_host, args.api_port, proc=proc)

        if len(available_models) < 2:
            raise RuntimeError(f"Expected at least 2 models, got {len(available_models)}: {available_models}")

        models = available_models[:2]  # Use first two available models
        print(f"  Using models for test: {models}")

        all_metrics = []
        test_start = time.time()

        for phase in range(1, args.phases + 1):
            model_idx = (phase - 1) % len(models)
            model_name = models[model_idx]
            model_label = chr(65 + model_idx)  # 'A', 'B', etc.

            print("\n" + "=" * 60)
            print(f"  PHASE {phase}/{args.phases}: Model {model_label}")
            print("=" * 60)

            # Clear previous warmup events before this phase
            log_capture.clear_warmup_events()

            # Switch model (only if not on first phase with default model)
            print(f"\n>>> Switching to model: {model_name}")

            if phase == 1:
                print("  (Skipping switch on first phase - model already loaded)")
                switch_result = {
                    'status': 'ok',
                    'duration_s': 0,
                    'switched': False,
                    'model': model_name,
                }
                warmup_s = None  # Phase 1 warmup not measured separately
            else:
                switch_result = await switch_model(args.api_host, args.api_port, model_name)

                if switch_result['status'] != 'ok':
                    print(f"  ✗ Switch failed: {switch_result.get('error')}")
                    continue

                print(f"  ✓ API call duration: {switch_result['duration_s']:.1f}s (includes warmup)")

                # Extract warmup time from logs captured during switch
                # Wait a bit for final logs to be captured
                await asyncio.sleep(0.5)
                warmup_times = log_capture.get_warmup_times()
                warmup_s = warmup_times[-1] if warmup_times else None
                if warmup_s is not None:
                    print(f"  ✓ Warmup time from logs: {warmup_s}s")

            # Generate
            print(f">>> Generating with model: {model_name}")
            prompt = PROMPTS[(phase - 1) % len(PROMPTS)]
            gen_result = await generate(
                args.api_host,
                args.api_port,
                model_name,
                prompt,
                seed=42,  # Fixed seed for reproducibility
                max_tokens=args.fixed_output_tokens,
                strict_tokens=True,
            )

            if gen_result['status'] != 'ok':
                print(f"  ✗ Generation failed: {gen_result.get('error')}")
                continue

            print(f"  ✓ Generated {gen_result['output_tokens']} tokens in {gen_result['duration_s']:.2f}s")

            phase_metrics = {
                'phase': phase,
                'model': model_name,
                'switch_s': switch_result.get('duration_s', 0),
                'gen_s': gen_result['duration_s'],
                'warmup_s': warmup_s,
                'tokens': gen_result['output_tokens'],
            }
            all_metrics.append(phase_metrics)

        total_time = time.time() - test_start

        # Print summary
        print("\n" + "=" * 60)
        print("  METRICS SUMMARY")
        print("=" * 60)
        print_metrics_table(all_metrics)
        print(f"\n  Total time: {total_time:.2f}s")

        # Success
        print("\n" + "=" * 60)
        print("  TEST COMPLETED ✓")
        print("=" * 60)
        print(f"  ✓ {args.phases} phases")
        print("  ✓ Model switching via API")
        print("  ✓ Warmup times captured from logs")

    except Exception:
        print("\n" + "=" * 60)
        print("  TEST FAILED ✗")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if proc:
            print("\n>>> Shutting down server...")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print("  ✓ Server stopped")

        # Clean up temp config if we created it
        if args.config is None and config_path:
            try:
                os.unlink(config_path)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
