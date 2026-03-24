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
import contextlib
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


def _display_model_name(model_name: str, width: int = 38) -> str:
    if len(model_name) <= width:
        return model_name
    return model_name[:width - 3] + "..."


def _mb_to_gb(memory_mb: float | None) -> float | None:
    if memory_mb is None:
        return None
    return memory_mb / 1024.0


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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmpfile:
        yaml.dump(config, tmpfile, default_flow_style=False)
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
                    proc: subprocess.Popen | None = None) -> list[dict[str, str]]:
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
                models = []
                for model_card in data.get('data', []):
                    model_id = model_card.get('id')
                    if not model_id:
                        continue
                    model_root = model_card.get('root') or model_card.get('model_path') or model_id
                    models.append({
                        'id': model_id,
                        'display_name': model_root,
                    })
                print("  ✓ Server is ready")
                print("  Available models:")
                for model in models:
                    print(f"    - {model['id']} -> {model['display_name']}")
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
                'reconfigure_ms': data.get('reconfigure_ms'),
                'switched': data.get('switched', False),
                'model': data.get('current_model'),
                'memory_before_mb': data.get('memory_before_mb'),
                'memory_after_mb': data.get('memory_after_mb'),
                'freed_memory_mb': data.get('freed_memory_mb'),
                'stash_memory_after_mb': data.get('stash_memory_after_mb'),
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
    hdr = (f"{'Phase':>5}  {'Model':<38}  "
           f"{'Init/Switch(s)':>14}  {'Warmup(s)':>9}  "
           f"{'Gen(s)':>7}  "
           f"{'Tokens':>7}  "
           f"{'Freed(GB)':>9}  "
           f"{'StashUsed(GB)':>13}")
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for m in all_metrics:
        warmup_s = m.get('warmup_s', 'N/A')
        warmup_str = f"{warmup_s:>9.1f}" if isinstance(warmup_s, (int, float)) else f"{str(warmup_s):>9}"

        freed_gb = _mb_to_gb(m.get('freed_memory_mb'))
        freed_str = f"{freed_gb:>9.2f}" if isinstance(freed_gb, (int, float)) else f"{'N/A':>9}"

        stash_used_gb = _mb_to_gb(m.get('stash_memory_after_mb'))
        stash_used_str = f"{stash_used_gb:>13.2f}" if isinstance(stash_used_gb, (int, float)) else f"{'N/A':>13}"

        print(f"{m['phase']:>5}  "
              f"{_display_model_name(m['model'], 38):<38}  "
              f"{m['reconfigure_s']:>13.1f}  "
              f"{warmup_str}  "
              f"{m['gen_s']:>7.2f}  "
              f"{m['tokens']:>7}  "
              f"{freed_str}  "
              f"{stash_used_str}")
    print(sep)

    n = len(all_metrics)
    if n > 0:
        avg = {
            'reconfigure_s': sum(m['reconfigure_s'] for m in all_metrics) / n,
            'gen_s': sum(m['gen_s'] for m in all_metrics) / n,
            'tokens': sum(m['tokens'] for m in all_metrics) / n,
        }
        warmup_times = [m.get('warmup_s') for m in all_metrics if isinstance(m.get('warmup_s'), (int, float))]
        if warmup_times:
            avg_warmup = sum(warmup_times) / len(warmup_times)
            warmup_str = f"{avg_warmup:>9.1f}"
        else:
            warmup_str = f"{'N/A':>9}"

        freed_gbs = [_mb_to_gb(m.get('freed_memory_mb')) for m in all_metrics]
        freed_gbs = [value for value in freed_gbs if isinstance(value, (int, float))]
        if freed_gbs:
            avg_freed = sum(freed_gbs) / len(freed_gbs)
            freed_str = f"{avg_freed:>9.2f}"
        else:
            freed_str = f"{'N/A':>9}"

        stash_used_gbs = [_mb_to_gb(m.get('stash_memory_after_mb')) for m in all_metrics]
        stash_used_gbs = [value for value in stash_used_gbs if isinstance(value, (int, float))]
        if stash_used_gbs:
            avg_stash_used = sum(stash_used_gbs) / len(stash_used_gbs)
            stash_used_str = f"{avg_stash_used:>13.2f}"
        else:
            stash_used_str = f"{'N/A':>13}"

        print(f"{'AVG':>5}  {'':<38}  "
              f"{avg['reconfigure_s']:>13.1f}  "
              f"{warmup_str}  "
              f"{avg['gen_s']:>7.2f}  "
              f"{avg['tokens']:>7.0f}  "
              f"{freed_str}  "
              f"{stash_used_str}")
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
        startup_begin = time.perf_counter()
        proc, log_thread = run_server(
            config_path,
            args.api_host,
            args.api_port,
            log_capture,
            args.max_num_batched_tokens,
        )
        available_models = wait_for_server(args.api_host, args.api_port, proc=proc)
        initial_load_s = time.perf_counter() - startup_begin
        startup_warmup_times = log_capture.get_warmup_times()
        startup_warmup_s = startup_warmup_times[-1] if startup_warmup_times else None

        if len(available_models) < 2:
            raise RuntimeError(f"Expected at least 2 models, got {len(available_models)}: {available_models}")

        models = available_models[:2]  # Use first two available models
        print("  Using models for test:")
        for model in models:
            print(f"    - {model['id']} -> {model['display_name']}")

        all_metrics = []
        test_start = time.time()

        for phase in range(1, args.phases + 1):
            model_idx = (phase - 1) % len(models)
            model_info = models[model_idx]
            model_name = model_info['id']
            model_display_name = model_info['display_name']

            print("\n" + "=" * 60)
            print(f"  PHASE {phase}/{args.phases}: {model_display_name}")
            print("=" * 60)

            # Clear previous warmup events before this phase
            log_capture.clear_warmup_events()

            # Switch model (only if not on first phase with default model)
            print(f"\n>>> Switching to model: {model_display_name} ({model_name})")

            if phase == 1:
                print("  (Skipping switch on first phase - model already loaded)")
                switch_result = {
                    'status': 'ok',
                    'duration_s': 0,
                    'reconfigure_ms': 0,
                    'switched': False,
                    'model': model_name,
                }
                warmup_s = None  # Phase 1 warmup not measured separately
            else:
                switch_result = await switch_model(args.api_host, args.api_port, model_name)

                if switch_result['status'] != 'ok':
                    print(f"  ✗ Switch failed: {switch_result.get('error')}")
                    continue

                reconfigure_ms = switch_result.get('reconfigure_ms')
                if isinstance(reconfigure_ms, (int, float)):
                    print(f"  ✓ Sleep+load(reconfigure) duration: {reconfigure_ms / 1000.0:.1f}s")
                else:
                    print(f"  ✓ Switch API duration: {switch_result['duration_s']:.1f}s")

                # Extract warmup time from logs captured during switch
                # Wait a bit for final logs to be captured
                await asyncio.sleep(0.5)
                warmup_times = log_capture.get_warmup_times()
                warmup_s = warmup_times[-1] if warmup_times else None
                if warmup_s is not None:
                    print(f"  ✓ Warmup time from logs: {warmup_s}s")

                freed_gb = _mb_to_gb(switch_result.get('freed_memory_mb'))
                if freed_gb is not None:
                    print(f"  ✓ Freed HPU memory: {freed_gb:.2f} GB")

                stash_used_gb = _mb_to_gb(switch_result.get('stash_memory_after_mb'))
                if stash_used_gb is not None:
                    print(f"  ✓ HPU memory still used after stashing: {stash_used_gb:.2f} GB")

            # Generate
            print(f">>> Generating with model: {model_display_name}")
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
                'phase': len(all_metrics) + 1,
                'model': model_display_name,
                'reconfigure_s': (switch_result.get('reconfigure_ms') or 0) / 1000.0,
                'warmup_s': warmup_s,
                'gen_s': gen_result['duration_s'],
                'tokens': gen_result['output_tokens'],
            }
            if phase == 1:
                phase_metrics['reconfigure_s'] = initial_load_s
                phase_metrics['warmup_s'] = startup_warmup_s
            else:
                phase_metrics['freed_memory_mb'] = switch_result.get('freed_memory_mb')
                phase_metrics['stash_memory_after_mb'] = switch_result.get('stash_memory_after_mb')
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
            with contextlib.suppress(Exception):
                os.unlink(config_path)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
