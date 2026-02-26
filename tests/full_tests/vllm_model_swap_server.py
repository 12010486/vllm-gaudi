#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Model Swap Server (Gaudi)
================================
A custom HTTP server that allows dynamic model swapping using vLLM's Python API.
Keeps the same process alive and reuses compilation caches between swaps.

Features:
- OpenAI-compatible /v1/completions endpoint
- Model swapping via /swap_model endpoint
- Sleep mode support via /sleep and /wake_up endpoints
- Health checks and metrics

Requires:
  VLLM_ENABLE_V1_MULTIPROCESSING=0

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python vllm_model_swap_server.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

  # Swap to a new model:
  curl -X POST http://localhost:8000/swap_model \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B"}'

  # Generate completions:
  curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "current", "prompt": "Hello", "max_tokens": 160}'
"""

import argparse
import asyncio
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm_gaudi.extension.profiler import HabanaMemoryProfiler


# ============================================================================
# Request/Response Models
# ============================================================================

class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class SwapModelRequest(BaseModel):
    model: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "vllm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.current_model: Optional[str] = None
        self.enforce_eager: bool = False
        self.max_model_len: int = 4096
        self.is_sleeping: bool = False
        self.load_count: int = 0
        self.swap_count: int = 0


state = ServerState()

# ============================================================================
# Thread Pool for Non-Blocking Generation
# ============================================================================
# Use a single-worker thread pool to prevent blocking the FastAPI event loop
# during generation, while respecting Gaudi's single-process constraint.
# The generate() call is synchronous, so we run it in a thread pool.
executor = ThreadPoolExecutor(max_workers=1)


# ============================================================================
# Model Management
# ============================================================================

def load_model(model_name: str) -> dict:
    """Load a model and return metrics."""
    print(f"\n>>> Loading model: {model_name}")
    
    with HabanaMemoryProfiler() as m:
        start = time.time()
        state.llm = LLM(
            model=model_name,
            enforce_eager=state.enforce_eager,
            max_model_len=state.max_model_len,
        )
        elapsed = time.time() - start
    
    state.current_model = model_name
    state.is_sleeping = False
    state.load_count += 1
    
    load_mem = m.consumed_device_memory / (1024**3)
    print(f"  ✓ Model loaded in {elapsed:.2f}s")
    print(f"  Memory: {m.get_summary_string()}")
    
    return {
        "load_time_s": elapsed,
        "load_mem_gib": load_mem,
        "model": model_name
    }


def destroy_model() -> dict:
    """Destroy the current model and free memory."""
    if state.llm is None:
        return {"destroyed": False, "message": "No model loaded"}
    
    print(f"\n>>> Destroying model: {state.current_model}")
    
    with HabanaMemoryProfiler() as m:
        start = time.time()
        
        # Explicitly release tensors
        try:
            multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING")
            if multiproc == "0":
                model_runner = state.llm.llm_engine.model_executor.driver_worker.worker.model_runner
                if model_runner and model_runner.model is not None:
                    import torch
                    for param in model_runner.model.parameters():
                        param.data = torch.empty(0)
        except Exception as e:
            print(f"  Warning: Failed to explicitly clear parameters: {e}")
        
        del state.llm
        state.llm = None
        state.current_model = None
        state.is_sleeping = False
        
        gc.collect()
        gc.collect()
        
        # Return freed memory to OS (Linux)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        
        try:
            import torch
            torch.hpu.synchronize()
        except Exception:
            pass
        
        elapsed = time.time() - start
    
    cleanup_gib = -m.consumed_device_memory / (1024**3)
    print(f"  ✓ Model destroyed in {elapsed:.2f}s")
    print(f"  Memory freed: {cleanup_gib:.2f} GiB")
    
    return {
        "destroyed": True,
        "destroy_time_s": elapsed,
        "cleanup_gib": cleanup_gib
    }


def sleep_model() -> dict:
    """Put the current model to sleep."""
    if state.llm is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if state.is_sleeping:
        return {"already_sleeping": True}
    
    print(f"\n>>> Sleeping model: {state.current_model}")
    
    with HabanaMemoryProfiler() as m:
        start = time.time()
        state.llm.sleep()
        elapsed = time.time() - start
    
    state.is_sleeping = True
    freed_bytes = -m.consumed_device_memory
    freed_gib = freed_bytes / (1024**3)
    
    print(f"  ✓ Model sleeping in {elapsed:.2f}s")
    print(f"  Memory freed: {freed_gib:.2f} GiB")
    
    return {
        "sleeping": True,
        "sleep_time_s": elapsed,
        "freed_gib": freed_gib
    }


def wake_model() -> dict:
    """Wake up the current model."""
    if state.llm is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if not state.is_sleeping:
        return {"already_awake": True}
    
    print(f"\n>>> Waking up model: {state.current_model}")
    
    with HabanaMemoryProfiler() as m:
        start = time.time()
        state.llm.wake_up()
        elapsed = time.time() - start
    
    state.is_sleeping = False
    consumed_gib = m.consumed_device_memory / (1024**3)
    
    print(f"  ✓ Model awake in {elapsed:.2f}s")
    print(f"  Memory consumed: {consumed_gib:.2f} GiB")
    
    return {
        "awake": True,
        "wake_time_s": elapsed,
        "consumed_gib": consumed_gib
    }


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load initial model if specified
    if hasattr(state, 'initial_model') and state.initial_model:
        load_model(state.initial_model)
    
    yield
    
    # Shutdown: Clean up
    if state.llm is not None:
        destroy_model()


app = FastAPI(title="vLLM Model Swap Server", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": state.current_model,
        "is_sleeping": state.is_sleeping,
        "load_count": state.load_count,
        "swap_count": state.swap_count
    }


@app.get("/v1/models")
async def list_models():
    """List available models (current model only)."""
    if state.current_model is None:
        return ModelList(data=[])
    
    return ModelList(
        data=[
            ModelInfo(
                id=state.current_model,
                created=int(time.time())
            )
        ]
    )


@app.get("/is_sleeping")
async def is_sleeping():
    """Check if the model is sleeping."""
    return state.is_sleeping


@app.post("/sleep")
async def sleep_endpoint():
    """Put the model to sleep."""
    try:
        result = sleep_model()
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wake_up")
async def wake_up():
    """Wake up the model."""
    try:
        result = wake_model()
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swap_model")
async def swap_model(request: SwapModelRequest):
    """Swap to a different model."""
    try:
        print(f"\n{'='*60}")
        print(f"  MODEL SWAP REQUEST: {request.model}")
        print(f"{'='*60}")
        
        swap_start = time.time()
        
        # Step 1: Sleep current model (if loaded and not already sleeping)
        sleep_metrics = {}
        if state.llm is not None and not state.is_sleeping:
            sleep_metrics = sleep_model()
        
        # Step 2: Destroy current model
        destroy_metrics = destroy_model() if state.llm is not None else {}
        
        # Step 3: Load new model
        load_metrics = load_model(request.model)
        
        swap_time = time.time() - swap_start
        state.swap_count += 1
        
        print(f"  ✓ Model swap completed in {swap_time:.2f}s")
        print(f"{'='*60}")
        
        return JSONResponse(content={
            "success": True,
            "model": request.model,
            "swap_time_s": swap_time,
            "metrics": {
                "sleep": sleep_metrics,
                "destroy": destroy_metrics,
                "load": load_metrics
            }
        })
    
    except Exception as e:
        print(f"  ✗ Model swap failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Generate completions (OpenAI-compatible)."""
    if state.llm is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    if state.is_sleeping:
        raise HTTPException(
            status_code=400,
            detail="Model is sleeping. Call /wake_up first."
        )
    
    try:
        # Prepare prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
        )
        
        # Generate using thread pool to avoid blocking event loop
        # The generate() call is synchronous, so we offload it to a thread
        start = time.time()
        outputs = await asyncio.get_event_loop().run_in_executor(
            executor,
            state.llm.generate,
            prompts,
            sampling_params,
        )
        gen_time = time.time() - start
        
        # Format response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for idx, output in enumerate(outputs):
            for output_item in output.outputs:
                choices.append(CompletionChoice(
                    text=output_item.text,
                    index=idx,
                    finish_reason=output_item.finish_reason or "stop"
                ))
                total_completion_tokens += len(output_item.token_ids)
            
            total_prompt_tokens += len(output.prompt_token_ids)
        
        usage = CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens
        )
        
        response = CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=state.current_model,
            choices=choices,
            usage=usage
        )
        
        print(f"  Generated {len(prompts)} prompts in {gen_time:.2f}s "
              f"({usage.completion_tokens} tokens)")
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server_info")
async def server_info():
    """Get detailed server information."""
    return {
        "current_model": state.current_model,
        "is_sleeping": state.is_sleeping,
        "enforce_eager": state.enforce_eager,
        "max_model_len": state.max_model_len,
        "load_count": state.load_count,
        "swap_count": state.swap_count,
        "vllm_enable_v1_multiprocessing": os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Model Swap Server with dynamic model loading"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Initial model to load (optional, can swap later)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Enforce eager mode (disables torch.compile)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    if multiproc != "0":
        print("=" * 60)
        print("  WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not set to 0")
        print("  Model swapping may not work correctly.")
        print("  Please set: VLLM_ENABLE_V1_MULTIPROCESSING=0")
        print("=" * 60)
        sys.exit(1)
    
    # Set state configuration
    state.enforce_eager = args.enforce_eager
    state.max_model_len = args.max_model_len
    state.initial_model = args.model
    
    # Print banner
    print("=" * 60)
    print("  vLLM MODEL SWAP SERVER")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Initial model: {args.model or 'None (swap later)'}")
    print(f"  Enforce eager: {args.enforce_eager}")
    print(f"  Max model len: {args.max_model_len}")
    print("=" * 60)
    print("\nEndpoints:")
    print(f"  POST   http://{args.host}:{args.port}/v1/completions")
    print(f"  POST   http://{args.host}:{args.port}/swap_model")
    print(f"  POST   http://{args.host}:{args.port}/sleep")
    print(f"  POST   http://{args.host}:{args.port}/wake_up")
    print(f"  GET    http://{args.host}:{args.port}/health")
    print(f"  GET    http://{args.host}:{args.port}/v1/models")
    print(f"  GET    http://{args.host}:{args.port}/is_sleeping")
    print(f"  GET    http://{args.host}:{args.port}/server_info")
    print("=" * 60 + "\n")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
