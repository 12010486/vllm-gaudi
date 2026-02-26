#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for vLLM Model Swap Server

This script tests the custom model swap server that keeps the same process alive
while swapping between different models on a single Gaudi card.

Advantages over process restart approach:
- Preserves kernel compilation caches
- Faster subsequent model loads
- Single persistent process

Usage:
  # Terminal 1: Start the server
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python vllm_model_swap_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000

  # Terminal 2: Run tests
  python test_model_swap_server.py --url http://localhost:8000 \
    --model-a meta-llama/Llama-3.1-8B-Instruct \
    --model-b Qwen/Qwen3-0.6B
"""

import argparse
import sys
import time
from typing import Optional

import requests


# ============================================================================
# API Client Functions
# ============================================================================

def check_health(url: str) -> dict:
    """Check server health."""
    response = requests.get(f"{url}/health")
    response.raise_for_status()
    return response.json()


def get_server_info(url: str) -> dict:
    """Get server information."""
    response = requests.get(f"{url}/server_info")
    response.raise_for_status()
    return response.json()


def list_models(url: str) -> dict:
    """List available models."""
    response = requests.get(f"{url}/v1/models")
    response.raise_for_status()
    return response.json()


def is_sleeping(url: str) -> bool:
    """Check if model is sleeping."""
    response = requests.get(f"{url}/is_sleeping")
    response.raise_for_status()
    return response.json()


def sleep_model(url: str) -> dict:
    """Put model to sleep."""
    response = requests.post(f"{url}/sleep")
    response.raise_for_status()
    return response.json()


def wake_model(url: str) -> dict:
    """Wake up model."""
    response = requests.post(f"{url}/wake_up")
    response.raise_for_status()
    return response.json()


def swap_model(url: str, model: str) -> dict:
    """Swap to a different model."""
    response = requests.post(
        f"{url}/swap_model",
        json={"model": model}
    )
    response.raise_for_status()
    return response.json()


def generate(url: str, prompt: str, max_tokens: int = 160) -> dict:
    """Generate completions."""
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "current",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
    )
    response.raise_for_status()
    return response.json()


# ============================================================================
# Test Functions
# ============================================================================

def test_initial_state(url: str):
    """Test initial server state."""
    print("\n" + "=" * 60)
    print("  TEST 1: Initial State")
    print("=" * 60)
    
    health = check_health(url)
    print(f"✓ Server health: {health['status']}")
    print(f"  Current model: {health['model']}")
    print(f"  Is sleeping: {health['is_sleeping']}")
    
    info = get_server_info(url)
    print(f"✓ Server info:")
    print(f"  VLLM_ENABLE_V1_MULTIPROCESSING: {info['vllm_enable_v1_multiprocessing']}")
    print(f"  Load count: {info['load_count']}")
    print(f"  Swap count: {info['swap_count']}")
    
    models = list_models(url)
    print(f"✓ Available models: {len(models['data'])}")
    for model in models['data']:
        print(f"  - {model['id']}")


def test_generation(url: str, prompt: str = "The capital of France is"):
    """Test text generation."""
    print("\n" + "=" * 60)
    print("  TEST 2: Text Generation")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    
    start = time.time()
    result = generate(url, prompt, max_tokens=160)
    elapsed = time.time() - start
    
    generated_text = result['choices'][0]['text']
    print(f"✓ Generated: '{generated_text}'")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens: {result['usage']['completion_tokens']}")


def test_sleep_wake(url: str):
    """Test sleep and wake functionality."""
    print("\n" + "=" * 60)
    print("  TEST 3: Sleep/Wake Cycle")
    print("=" * 60)
    
    # Sleep
    print("Sleeping model...")
    sleep_result = sleep_model(url)
    print(f"✓ Model sleeping:")
    print(f"  Sleep time: {sleep_result.get('sleep_time_s', 0):.2f}s")
    print(f"  Memory freed: {sleep_result.get('freed_gib', 0):.2f} GiB")
    
    # Verify sleeping
    sleeping = is_sleeping(url)
    assert sleeping, "Model should be sleeping"
    print(f"✓ Confirmed: is_sleeping = {sleeping}")
    
    # Wake
    print("Waking model...")
    wake_result = wake_model(url)
    print(f"✓ Model awake:")
    print(f"  Wake time: {wake_result.get('wake_time_s', 0):.2f}s")
    print(f"  Memory consumed: {wake_result.get('consumed_gib', 0):.2f} GiB")
    
    # Verify awake
    sleeping = is_sleeping(url)
    assert not sleeping, "Model should be awake"
    print(f"✓ Confirmed: is_sleeping = {sleeping}")


def test_model_swap(url: str, model_name: str):
    """Test model swapping."""
    print("\n" + "=" * 60)
    print(f"  TEST 4: Model Swap to {model_name}")
    print("=" * 60)
    
    print(f"Swapping to model: {model_name}")
    swap_result = swap_model(url, model_name)
    
    print(f"✓ Model swapped successfully:")
    print(f"  Total swap time: {swap_result['swap_time_s']:.2f}s")
    
    metrics = swap_result['metrics']
    if metrics.get('sleep'):
        print(f"  Sleep time: {metrics['sleep'].get('sleep_time_s', 0):.2f}s")
    if metrics.get('destroy'):
        print(f"  Destroy time: {metrics['destroy'].get('destroy_time_s', 0):.2f}s")
    if metrics.get('load'):
        print(f"  Load time: {metrics['load'].get('load_time_s', 0):.2f}s")
        print(f"  Load memory: {metrics['load'].get('load_mem_gib', 0):.2f} GiB")
    
    # Verify new model is loaded
    models = list_models(url)
    current_model = models['data'][0]['id'] if models['data'] else None
    assert current_model == model_name, f"Expected {model_name}, got {current_model}"
    print(f"✓ Confirmed current model: {current_model}")


def test_generation_after_swap(url: str, prompt: str = "Hello, how are"):
    """Test generation after model swap."""
    print("\n" + "=" * 60)
    print("  TEST 5: Generation After Swap")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    
    start = time.time()
    result = generate(url, prompt, max_tokens=160)
    elapsed = time.time() - start
    
    generated_text = result['choices'][0]['text']
    print(f"✓ Generated: '{generated_text}'")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens: {result['usage']['completion_tokens']}")


def test_multiple_swaps(url: str, model_a: str, model_b: str, cycles: int = 2):
    """Test multiple back-and-forth swaps."""
    print("\n" + "=" * 60)
    print(f"  TEST 6: Multiple Swaps ({cycles} cycles)")
    print("=" * 60)
    
    swap_times = []
    
    for i in range(cycles):
        print(f"\nCycle {i+1}/{cycles}:")
        
        # Swap to model B
        print(f"  → Swapping to {model_b}")
        result_b = swap_model(url, model_b)
        swap_time_b = result_b['swap_time_s']
        swap_times.append(("A→B", swap_time_b))
        print(f"    Swap time: {swap_time_b:.2f}s")
        
        # Quick generation test
        gen_result = generate(url, "Test", max_tokens=16)
        print(f"    Generated: '{gen_result['choices'][0]['text']}'")
        
        # Swap back to model A
        print(f"  → Swapping to {model_a}")
        result_a = swap_model(url, model_a)
        swap_time_a = result_a['swap_time_s']
        swap_times.append(("B→A", swap_time_a))
        print(f"    Swap time: {swap_time_a:.2f}s")
        
        # Quick generation test
        gen_result = generate(url, "Test", max_tokens=16)
        print(f"    Generated: '{gen_result['choices'][0]['text']}'")
    
    # Summary
    print(f"\n✓ Swap time summary:")
    for label, swap_time in swap_times:
        print(f"  {label}: {swap_time:.2f}s")
    
    avg_swap_time = sum(t for _, t in swap_times) / len(swap_times)
    print(f"  Average: {avg_swap_time:.2f}s")


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {url}...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            health = check_health(url)
            if health['status'] == 'healthy':
                print(f"✓ Server is ready (model: {health['model']})")
                return True
        except Exception:
            pass
        time.sleep(1)
    
    return False


# ============================================================================
# Main Test Suite
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM Model Swap Server"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="First model name (should be initially loaded)"
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Second model name to swap to"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Number of swap cycles for multi-swap test (default: 2)"
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for server (assume it's already ready)"
    )
    
    args = parser.parse_args()
    
    # Print test configuration
    print("=" * 60)
    print("  vLLM MODEL SWAP SERVER TEST SUITE")
    print("=" * 60)
    print(f"Server URL: {args.url}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Swap cycles: {args.cycles}")
    print("=" * 60)
    
    # Wait for server to be ready
    if not args.skip_wait:
        if not wait_for_server(args.url, timeout=60):
            print("✗ Server did not become ready in time")
            sys.exit(1)
    
    try:
        # Run tests
        test_initial_state(args.url)
        test_generation(args.url)
        test_sleep_wake(args.url)
        test_model_swap(args.url, args.model_b)
        test_generation_after_swap(args.url)
        test_multiple_swaps(args.url, args.model_a, args.model_b, args.cycles)
        
        # Final state check
        print("\n" + "=" * 60)
        print("  FINAL STATE")
        print("=" * 60)
        info = get_server_info(args.url)
        print(f"Current model: {info['current_model']}")
        print(f"Total loads: {info['load_count']}")
        print(f"Total swaps: {info['swap_count']}")
        
        print("\n" + "=" * 60)
        print("  ✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
