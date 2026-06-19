# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online Model Swapping Benchmark for Gaudi (API-based)
======================================================
Runs N phases alternating between two models configured in a multi-model YAML.
Each phase performs an in-process model switch (except first phase if already
on the default model), then executes `vllm bench serve` with fixed benchmark
settings. Optionally, each phase can keep launching benchmark runs until a
phase-time budget is reached.

Collects:
  - Switch/reconfigure timing
  - Warmup time from server logs
  - Freed/stash memory from /v1/models/switch response
  - Benchmark metrics (req/s, tok/s, TTFT/TPOT/ITL/E2EL percentiles)

Usage:
  python tests/full_tests/online_model_swap_bench.py \
    --config multi_models.yaml \
    --phases 6 \
    --phase-time-s 300 \
    --api-host 127.0.0.1 \
    --api-port 8090
"""

import argparse
import asyncio
import contextlib
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import requests
import yaml

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False


def _server_api_host(api_host: str) -> str:
    return '127.0.0.1' if api_host == 'localhost' else api_host


def _client_api_host(api_host: str) -> str:
    return '127.0.0.1' if api_host in {'localhost', '0.0.0.0'} else api_host


def _api_url(api_host: str, api_port: int, path: str) -> str:
    return f"http://{_client_api_host(api_host)}:{api_port}{path}"


def _display_model_name(model_name: str, width: int = 40) -> str:
    if len(model_name) <= width:
        return model_name
    return model_name[:width - 3] + "..."


def _mb_to_gb(memory_mb: float | None) -> float | None:
    if memory_mb is None:
        return None
    return memory_mb / 1024.0


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def read_multi_model_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding='utf-8') as file_obj:
        config = yaml.safe_load(file_obj) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Config {config_path} must be a YAML mapping")
    models = config.get('models')
    if not isinstance(models, dict) or len(models) < 2:
        raise RuntimeError(f"Config {config_path} must contain at least two models")
    return config


def pick_two_models(config: dict[str, Any]) -> list[str]:
    default_model = config.get('default_model')
    model_ids = [str(key) for key in config['models'].keys()]
    if isinstance(default_model, str) and default_model in model_ids:
        first = default_model
    else:
        first = model_ids[0]
    second = next((model for model in model_ids if model != first), None)
    if second is None:
        raise RuntimeError("Could not pick a second model from config")
    return [first, second]


class ServerLogCapture:
    """Capture server logs and parse warmup timings."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.lock = threading.Lock()
        self.warmup_events: list[tuple[float, int]] = []

    def add_line(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            match = re.search(r'Warmup finished in (\d+) secs', line)
            if match:
                self.warmup_events.append((time.time(), int(match.group(1))))

    def get_warmup_times(self) -> list[int]:
        with self.lock:
            return [value for _, value in self.warmup_events]

    def clear_warmup_events(self) -> None:
        with self.lock:
            self.warmup_events = []


def run_server(config_path: str,
               api_host: str,
               api_port: int,
               log_capture: ServerLogCapture) -> tuple[subprocess.Popen[str], threading.Thread]:
    server_host = _server_api_host(api_host)
    env = os.environ.copy()
    env['VLLM_SERVER_DEV_MODE'] = '1'
    env['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
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
        '--disable-log-stats',
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

    def capture_logs() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip('\n')
                if stripped:
                    print(f"[SERVER] {stripped}")
                    log_capture.add_line(stripped)
        except Exception as exc:  # noqa: BLE001
            print(f"[LOG_CAPTURE_ERROR] {exc}")

    log_thread = threading.Thread(target=capture_logs, daemon=True)
    log_thread.start()
    return proc, log_thread


async def wait_for_server(api_host: str,
                          api_port: int,
                          timeout: int = 600,
                          proc: subprocess.Popen[str] | None = None) -> None:
    url = _api_url(api_host, api_port, '/v1/models')
    start = time.time()
    last_error = None

    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server exited before readiness check succeeded (exit code {proc.returncode})")
        try:
            response = await asyncio.to_thread(_HTTP_SESSION.get, url, timeout=5)
            if response.status_code == 200:
                print("  ✓ Server is ready")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        await asyncio.sleep(2)

    suffix = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(f"Server did not start after {timeout}s.{suffix}")


async def switch_model(api_host: str, api_port: int, model_name: str, drain_timeout: int = 60) -> dict[str, Any]:
    url = _api_url(api_host, api_port, '/v1/models/switch')
    payload = {
        'model': model_name,
        'drain_timeout': drain_timeout,
    }

    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(_HTTP_SESSION.post, url, json=payload, timeout=600)
        elapsed_s = time.perf_counter() - start
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'ok',
                'duration_s': elapsed_s,
                'api_duration_ms': data.get('duration_ms', 0),
                'reconfigure_ms': data.get('reconfigure_ms'),
                'switched': data.get('switched', False),
                'model': data.get('current_model'),
                'memory_before_mb': data.get('memory_before_mb'),
                'memory_after_unload_mb': data.get('memory_after_unload_mb'),
                'freed_memory_mb': data.get('freed_memory_mb'),
                'stash_memory_after_mb': data.get('stash_memory_after_mb'),
                'restored_from_stash': data.get('restored_from_stash'),
                'restored_from_stash_workers': data.get('restored_from_stash_workers'),
                'cold_load_workers': data.get('cold_load_workers'),
                'num_gpu_blocks': data.get('num_gpu_blocks'),
            }
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': response.text,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            'status': 'error',
            'duration_s': time.perf_counter() - start,
            'error': str(exc),
        }


_BENCH_KEY_REGEX = re.compile(r'^([A-Za-z0-9()/ .-]+):\s+(.+?)\s*$')
_LATENCY_REGEX = re.compile(r'^(Mean|Median|P\d+)\s+(TTFT|TPOT|ITL|E2EL) \(ms\):\s+([0-9]+(?:\.[0-9]+)?)\s*$')
_FIRST_FLOAT_REGEX = re.compile(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')


def _extract_first_float(text: str) -> float | None:
    match = _FIRST_FLOAT_REGEX.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(',', ''))
    except ValueError:
        return None


def parse_bench_output(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lat_match = _LATENCY_REGEX.match(line)
        if lat_match:
            prefix, metric, value = lat_match.groups()
            key = f"{metric.lower()}_{prefix.lower()}_ms"
            metrics[key] = float(value)
            continue

        key_match = _BENCH_KEY_REGEX.match(line)
        if not key_match:
            continue
        text_key, value_text = key_match.groups()
        value_float = _extract_first_float(value_text)
        if value_float is None:
            continue
        normalized = text_key.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        normalized = re.sub(r'[^a-z0-9_]+', '', normalized)

        if normalized.startswith('request_throughput'):
            normalized = 'request_throughput_reqs'
        elif normalized.startswith('output_token_throughput'):
            normalized = 'output_token_throughput_toks'
        elif normalized.startswith('peak_output_token_throughput'):
            normalized = 'peak_output_token_throughput_toks'
        elif normalized.startswith('total_token_throughput'):
            normalized = 'total_token_throughput_toks'

        # Keep only the top-level counters/throughputs from the benchmark summary.
        if normalized in {
            'successful_requests',
            'failed_requests',
            'benchmark_duration_s',
            'request_throughput_reqs',
            'output_token_throughput_toks',
            'peak_output_token_throughput_toks',
            'peak_concurrent_requests',
            'total_token_throughput_toks',
        }:
            metrics[normalized] = value_float

    return metrics


def _metric_average(phase_runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in phase_runs if key in run and isinstance(run[key], (int, float))]
    return _avg(values)


def aggregate_phase_bench_metrics(phase_runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        'bench_runs': len(phase_runs),
        'bench_failures': sum(1 for run in phase_runs if run.get('status') != 'ok'),
        'successful_requests': int(sum(run.get('successful_requests', 0) for run in phase_runs)),
        'failed_requests': int(sum(run.get('failed_requests', 0) for run in phase_runs)),
    }

    for key in [
            'request_throughput_reqs',
            'output_token_throughput_toks',
            'total_token_throughput_toks',
            'ttft_p50_ms',
            'ttft_p90_ms',
            'tpot_p50_ms',
            'tpot_p90_ms',
            'itl_p50_ms',
            'itl_p90_ms',
            'e2el_p50_ms',
            'e2el_p90_ms',
    ]:
        averaged = _metric_average(phase_runs, key)
        if averaged is not None:
            aggregate[key] = averaged

    return aggregate


async def run_bench_once(api_host: str,
                         api_port: int,
                         model_name: str,
                         num_prompts: int,
                         max_concurrency: int,
                         random_input_len: int,
                         random_output_len: int,
                         request_rate: str,
                         timeout: int) -> dict[str, Any]:
    cmd = [
        'vllm',
        'bench',
        'serve',
        '--model',
        model_name,
        '--dataset-name',
        'random',
        '--random-input-len',
        str(random_input_len),
        '--random-output-len',
        str(random_output_len),
        '--base-url',
        f"http://{_client_api_host(api_host)}:{api_port}",
        '--num-prompts',
        str(num_prompts),
        '--max-concurrency',
        str(max_concurrency),
        '--request-rate',
        request_rate,
        '--port',
        str(api_port),
        '--percentile-metrics',
        'ttft,tpot,itl,e2el',
        '--metric-percentiles',
        '50,90,95,99',
        '--save-result',
        '--save-detailed',
        '--temperature',
        '0',
        '--trust-remote-code',
    ]

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1,localhost,0.0.0.0']))
    env['no_proxy'] = ','.join(filter(None, [env.get('no_proxy'), '127.0.0.1,localhost,0.0.0.0']))

    print(f"    [bench] Running: {' '.join(cmd)}")
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    timed_out = False
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_bytes, _ = await proc.communicate()

    elapsed_s = time.perf_counter() - start
    output_text = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
    if output_text:
        print(output_text)

    parsed_metrics = parse_bench_output(output_text)
    if timed_out:
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': f'bench timed out after {timeout}s',
        }

    if proc.returncode != 0:
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': f'bench exited with code {proc.returncode}',
            **parsed_metrics,
        }

    return {
        'status': 'ok',
        'duration_s': elapsed_s,
        **parsed_metrics,
    }


def print_metrics_table(all_metrics: list[dict[str, Any]]) -> None:
    hdr = (f"{'Phase':>5}  {'Model':<40}  "
           f"{'Init/Switch(s)':>14}  {'Warmup(s)':>9}  "
           f"{'Req/s':>8}  {'Tok/s':>8}  "
           f"{'P50 TTFT':>9}  {'P90 TTFT':>9}  "
           f"{'P50 E2EL':>9}  {'P90 E2EL':>9}  "
           f"{'Freed(GB)':>9}  {'StashUsed(GB)':>13}")
    sep = '-' * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for metric in all_metrics:
        warmup_s = metric.get('warmup_s')
        warmup_str = f"{warmup_s:>9.1f}" if isinstance(warmup_s, (int, float)) else f"{'N/A':>9}"
        freed_gb = _mb_to_gb(metric.get('freed_memory_mb'))
        freed_str = f"{freed_gb:>9.2f}" if isinstance(freed_gb, (int, float)) else f"{'N/A':>9}"
        stash_used_gb = _mb_to_gb(metric.get('stash_memory_after_mb'))
        stash_used_str = f"{stash_used_gb:>13.2f}" if isinstance(stash_used_gb, (int, float)) else f"{'N/A':>13}"

        req_s = metric.get('request_throughput_reqs')
        req_s_str = f"{req_s:>8.2f}" if isinstance(req_s, (int, float)) else f"{'N/A':>8}"
        tok_s = metric.get('output_token_throughput_toks')
        tok_s_str = f"{tok_s:>8.2f}" if isinstance(tok_s, (int, float)) else f"{'N/A':>8}"

        ttft_p50 = metric.get('ttft_p50_ms')
        ttft_p50_str = f"{ttft_p50:>9.2f}" if isinstance(ttft_p50, (int, float)) else f"{'N/A':>9}"
        ttft_p90 = metric.get('ttft_p90_ms')
        ttft_p90_str = f"{ttft_p90:>9.2f}" if isinstance(ttft_p90, (int, float)) else f"{'N/A':>9}"

        e2el_p50 = metric.get('e2el_p50_ms')
        e2el_p50_str = f"{e2el_p50:>9.2f}" if isinstance(e2el_p50, (int, float)) else f"{'N/A':>9}"
        e2el_p90 = metric.get('e2el_p90_ms')
        e2el_p90_str = f"{e2el_p90:>9.2f}" if isinstance(e2el_p90, (int, float)) else f"{'N/A':>9}"

        print(f"{metric['phase']:>5}  "
              f"{_display_model_name(metric['model'], 40):<40}  "
              f"{metric['reconfigure_s']:>13.1f}  "
              f"{warmup_str}  "
              f"{req_s_str}  "
              f"{tok_s_str}  "
              f"{ttft_p50_str}  "
              f"{ttft_p90_str}  "
              f"{e2el_p50_str}  "
              f"{e2el_p90_str}  "
              f"{freed_str}  "
              f"{stash_used_str}")
    print(sep)


async def main() -> int:
    parser = argparse.ArgumentParser(description='Online Model Swapping Benchmark (bench serve)')
    parser.add_argument('--config', type=str, default='multi_models.yaml', help='Multi-model YAML config path')
    parser.add_argument('--api-host', type=str, default='127.0.0.1')
    parser.add_argument('--api-port', type=int, default=8090)
    parser.add_argument('--phases', type=int, default=4, help='Number of alternating swap phases')
    parser.add_argument('--phase-time-s',
                        type=float,
                        default=0.0,
                        help='If > 0, keep running benchmark within each phase until this time budget is reached')
    parser.add_argument('--drain-timeout', type=int, default=60)
    parser.add_argument('--num-prompts', type=int, default=40)
    parser.add_argument('--max-concurrency', type=int, default=4)
    parser.add_argument('--request-rate', type=str, default='inf')
    parser.add_argument('--random-input-len', type=int, default=4028)
    parser.add_argument('--random-output-len', type=int, default=1024)
    parser.add_argument('--bench-timeout-s', type=int, default=3600)
    parser.add_argument('--keep-server',
                        action='store_true',
                        help='Do not stop the server at the end (useful for manual debugging)')
    args = parser.parse_args()

    config = read_multi_model_config(args.config)
    test_models = pick_two_models(config)

    if args.api_port <= 0:
        args.api_port = find_free_port()

    print('=' * 72)
    print('  ONLINE MODEL SWAP BENCHMARK (BENCH SERVE)')
    print('=' * 72)
    print(f"  Config: {args.config}")
    print(f"  Models: {test_models[0]} <-> {test_models[1]}")
    print(f"  API: {args.api_host}:{args.api_port}")
    print(f"  Phases: {args.phases}")
    print(f"  Phase time budget: {args.phase_time_s:.1f}s (0 means one run per phase)")
    print(f"  Bench prompts/concurrency: {args.num_prompts}/{args.max_concurrency}")
    print(f"  Random input/output len: {args.random_input_len}/{args.random_output_len}")
    print('=' * 72)

    log_capture = ServerLogCapture()
    proc: subprocess.Popen[str] | None = None

    try:
        startup_begin = time.perf_counter()
        proc, _ = run_server(args.config, args.api_host, args.api_port, log_capture)
        await wait_for_server(args.api_host, args.api_port, proc=proc)
        initial_load_s = time.perf_counter() - startup_begin
        startup_warmup_times = log_capture.get_warmup_times()
        startup_warmup_s = startup_warmup_times[-1] if startup_warmup_times else None

        all_metrics: list[dict[str, Any]] = []
        phase_failures: list[str] = []
        test_start = time.time()
        server_died = False

        for phase in range(1, args.phases + 1):
            if server_died:
                break
            model_name = test_models[(phase - 1) % 2]
            print('\n' + '=' * 72)
            print(f"  PHASE {phase}/{args.phases}: {model_name}")
            print('=' * 72)

            log_capture.clear_warmup_events()
            if phase == 1:
                print('>>> First phase on default model (skip switch)')
                switch_result: dict[str, Any] = {
                    'status': 'ok',
                    'duration_s': 0.0,
                    'reconfigure_ms': 0.0,
                    'switched': False,
                    'model': model_name,
                }
                warmup_s = startup_warmup_s
            else:
                print(f"\n>>> Switching to model: {model_name}")
                switch_result = await switch_model(args.api_host, args.api_port, model_name, args.drain_timeout)
                if switch_result.get('status') != 'ok':
                    error_msg = str(switch_result.get('error'))
                    print(f"  ✗ Switch failed: {error_msg}")
                    phase_failures.append(f"phase{phase} switch failed: {error_msg}")
                    if proc is not None and proc.poll() is not None:
                        print(f"  ✗ Server process exited (code {proc.returncode}), aborting remaining phases")
                        server_died = True
                        break
                    continue

                reconfigure_ms = switch_result.get('reconfigure_ms')
                if isinstance(reconfigure_ms, (int, float)):
                    print(f"  ✓ Sleep+load(reconfigure) duration: {reconfigure_ms / 1000.0:.1f}s")
                else:
                    print(f"  ✓ Switch API duration: {switch_result['duration_s']:.1f}s")

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
                restored_from_stash = switch_result.get('restored_from_stash')
                if isinstance(restored_from_stash, bool):
                    restore_mode = 'stash-restore' if restored_from_stash else 'cold-load'
                    restored_workers = switch_result.get('restored_from_stash_workers')
                    cold_workers = switch_result.get('cold_load_workers')
                    print(f"  ✓ Reload path: {restore_mode} "
                          f"(restored_workers={restored_workers}, cold_workers={cold_workers})")
                num_gpu_blocks = switch_result.get('num_gpu_blocks')
                if isinstance(num_gpu_blocks, int):
                    print(f"  ✓ KV cache num_gpu_blocks: {num_gpu_blocks}")

            print('>>> Running benchmark phase')
            phase_runs: list[dict[str, Any]] = []
            phase_begin = time.perf_counter()
            run_idx = 1
            while True:
                if proc is not None and proc.poll() is not None:
                    print(f"  ✗ Server process exited (code {proc.returncode}) before bench run {run_idx}")
                    server_died = True
                    break
                print(f"  [phase {phase}] bench run {run_idx}")
                bench_result = await run_bench_once(
                    args.api_host,
                    args.api_port,
                    model_name,
                    args.num_prompts,
                    args.max_concurrency,
                    args.random_input_len,
                    args.random_output_len,
                    args.request_rate,
                    args.bench_timeout_s,
                )
                phase_runs.append(bench_result)
                if bench_result.get('status') != 'ok':
                    print(f"  ✗ Bench run {run_idx} failed: {bench_result.get('error')}")
                    if proc is not None and proc.poll() is not None:
                        print(f"  ✗ Server process exited (code {proc.returncode}) during bench run")
                        server_died = True
                    break

                # Treat all-requests-failed as a failure even when bench exits 0
                # (server died mid-run; aiohttp gets ConnectionRefused for every request)
                n_ok = int(bench_result.get('successful_requests', 0))
                n_failed = int(bench_result.get('failed_requests', 0))
                if n_ok == 0 and n_failed > 0:
                    print(f"  ✗ Bench run {run_idx}: all {n_failed} requests failed (no successful requests)")
                    phase_runs.pop()  # discard this unusable run
                    if proc is not None and proc.poll() is not None:
                        print(f"  ✗ Server process exited (code {proc.returncode})")
                        server_died = True
                    break

                if args.phase_time_s <= 0:
                    break
                elapsed = time.perf_counter() - phase_begin
                if elapsed >= args.phase_time_s:
                    break
                run_idx += 1

            if server_died:
                if phase_runs:
                    phase_failures.append(f"phase{phase} server died during benchmark (partial data discarded)")
                break

            bench_agg = aggregate_phase_bench_metrics(phase_runs)
            if bench_agg['bench_failures'] > 0 and bench_agg['successful_requests'] == 0:
                phase_failures.append(f"phase{phase} benchmark failed")
                continue

            phase_metrics = {
                'phase': len(all_metrics) + 1,
                'model': model_name,
                'reconfigure_s': (switch_result.get('reconfigure_ms') or 0) / 1000.0,
                'warmup_s': warmup_s,
                **bench_agg,
            }
            if phase == 1:
                phase_metrics['reconfigure_s'] = initial_load_s
            else:
                phase_metrics['freed_memory_mb'] = switch_result.get('freed_memory_mb')
                phase_metrics['stash_memory_after_mb'] = switch_result.get('stash_memory_after_mb')
                phase_metrics['restored_from_stash'] = switch_result.get('restored_from_stash')
                phase_metrics['restored_from_stash_workers'] = switch_result.get('restored_from_stash_workers')
                phase_metrics['cold_load_workers'] = switch_result.get('cold_load_workers')
                phase_metrics['num_gpu_blocks'] = switch_result.get('num_gpu_blocks')
            all_metrics.append(phase_metrics)

        total_time = time.time() - test_start
        print('\n' + '=' * 72)
        print('  METRICS SUMMARY')
        print('=' * 72)
        print_metrics_table(all_metrics)
        print(f"\n  Total time: {total_time:.2f}s")

        if phase_failures:
            print('\n' + '=' * 72)
            print('  TEST FAILED ✗')
            print('=' * 72)
            for failure in phase_failures:
                print(f"  - {failure}")
            return 1

        print('\n' + '=' * 72)
        print('  TEST COMPLETED ✓')
        print('=' * 72)
        print(f"  ✓ {len(all_metrics)} phases completed")
        print('  ✓ Model switching via API')
        print('  ✓ Benchmark metrics collected from vllm bench serve')
        return 0

    except Exception:  # noqa: BLE001
        print('\n' + '=' * 72)
        print('  TEST FAILED ✗')
        print('=' * 72)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if proc and not args.keep_server:
            print('\n>>> Shutting down server...')
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=10)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    proc.wait(timeout=5)
            print('  ✓ Server stopped')


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
