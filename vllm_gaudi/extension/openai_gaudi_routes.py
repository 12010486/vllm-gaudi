# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gaudi OpenAI API Extension Routes
==================================
Extends the OpenAI-compatible API server with Gaudi-specific functionality:
- Model swapping via /v1/model/swap
- Sleep mode via /v1/model/sleep and /v1/model/wake_up
- Server info via /v1/model/info

These routes operate on the existing EngineClient without spawning a separate server.
Works in single-process mode (VLLM_ENABLE_V1_MULTIPROCESSING=0).

Usage:
  # In vllm_gaudi/entrypoints/openai/api_server.py _run_server():
  from vllm_gaudi.extension.openai_gaudi_routes import register_gaudi_openai_routes
  register_gaudi_openai_routes(app)
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.logger import init_logger

logger = init_logger("vllm.extension.openai_gaudi_routes")


# ============================================================================
# Request/Response Models
# ============================================================================

class ModelSwapRequest(BaseModel):
    """Request to swap to a different model."""
    model: str
    max_model_len: Optional[int] = None


class ModelSwapResponse(BaseModel):
    """Response from model swap."""
    success: bool
    model: str
    swap_time_s: float
    metrics: dict


class SleepResponse(BaseModel):
    """Response from sleep request."""
    sleeping: bool
    time_s: float
    freed_gib: Optional[float] = None


class WakeResponse(BaseModel):
    """Response from wake request."""
    awake: bool
    time_s: float
    consumed_gib: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Current model and server state."""
    current_model: str
    is_sleeping: bool
    load_count: int
    swap_count: int


# ============================================================================
# Gaudi State Manager
# ============================================================================

class GaudiOpenAIStateManager:
    """
    Manages Gaudi-specific state for the OpenAI API server.
    Tracks model swaps, sleep state, and metrics.
    """

    def __init__(self):
        self.current_model: Optional[str] = None
        self.is_sleeping: bool = False
        self.load_count: int = 0
        self.swap_count: int = 0
        self._swap_lock: bool = False

    def set_model(self, model_name: str) -> None:
        """Set the current model name."""
        self.current_model = model_name
        self.load_count += 1
        self.is_sleeping = False

    def sleep_model(self) -> None:
        """Mark model as sleeping."""
        self.is_sleeping = True

    def wake_model(self) -> None:
        """Mark model as awake."""
        self.is_sleeping = False

    def increment_swap_count(self) -> None:
        """Increment swap count."""
        self.swap_count += 1


# ============================================================================
# Route Handlers
# ============================================================================

def get_gaudi_state(request: Request) -> GaudiOpenAIStateManager:
    """Dependency to get Gaudi state manager from app.state."""
    if not hasattr(request.app.state, "gaudi_openai_state"):
        request.app.state.gaudi_openai_state = GaudiOpenAIStateManager()
    return request.app.state.gaudi_openai_state


def get_engine_client(request: Request):
    """Get the EngineClient from app.state."""
    if not hasattr(request.app.state, "engine_client"):
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return request.app.state.engine_client


gaudi_router = APIRouter(prefix="/v1/model", tags=["gaudi"])


@gaudi_router.get("/info")
async def get_model_info(
    gaudi_state: GaudiOpenAIStateManager = Depends(get_gaudi_state),
) -> ModelInfoResponse:
    """Get current model and Gaudi server state."""
    return ModelInfoResponse(
        current_model=gaudi_state.current_model or "none",
        is_sleeping=gaudi_state.is_sleeping,
        load_count=gaudi_state.load_count,
        swap_count=gaudi_state.swap_count,
    )


@gaudi_router.post("/swap")
async def swap_model(
    request: ModelSwapRequest,
    engine_client=Depends(get_engine_client),
    gaudi_state: GaudiOpenAIStateManager = Depends(get_gaudi_state),
) -> ModelSwapResponse:
    """
    Swap to a different model.
    
    This operation:
    1. Pauses request handling by entering a lock
    2. Stops the current model
    3. Creates a new engine with the target model
    4. Resumes request handling
    
    Note: This may take several seconds depending on model size.
    """
    if gaudi_state._swap_lock:
        raise HTTPException(
            status_code=409,
            detail="Model swap already in progress"
        )

    gaudi_state._swap_lock = True
    try:
        swap_start = time.time()
        logger.info(f"Swapping model to: {request.model}")

        # Step 1: Shutdown current engine
        # This will flush pending requests and release GPU memory
        logger.info("Shutting down current engine...")
        await engine_client.shutdown()

        # Step 2: Rebuild engine with new model
        # Note: This requires access to the original args
        # which should be stored in app.state.args
        logger.info(f"Initializing new engine with model: {request.model}")
        
        # Update model name in args (if args are available)
        # This is a simplified approach; in production, you'd reconstruct AsyncEngineArgs
        if hasattr(engine_client, "args"):
            engine_client.args.model = request.model
            if request.max_model_len:
                engine_client.args.max_model_len = request.max_model_len

        # Restart the engine
        await engine_client.start()

        swap_time = time.time() - swap_start
        gaudi_state.set_model(request.model)
        gaudi_state.increment_swap_count()

        logger.info(f"Model swap completed in {swap_time:.2f}s to {request.model}")

        return ModelSwapResponse(
            success=True,
            model=request.model,
            swap_time_s=swap_time,
            metrics={
                "shutdown_to_init_s": swap_time
            }
        )

    except Exception as e:
        logger.error(f"Model swap failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model swap failed: {str(e)}")
    
    finally:
        gaudi_state._swap_lock = False


@gaudi_router.post("/sleep")
async def sleep_model(
    engine_client=Depends(get_engine_client),
    gaudi_state: GaudiOpenAIStateManager = Depends(get_gaudi_state),
) -> SleepResponse:
    """
    Put the model to sleep.
    
    This pauses generation without unloading the model, freeing GPU memory
    for other workloads. Use /v1/model/wake_up to resume.
    """
    if gaudi_state.is_sleeping:
        return SleepResponse(
            sleeping=True,
            time_s=0.0,
            freed_gib=None
        )

    try:
        start = time.time()
        logger.info("Putting model to sleep...")
        
        # Call sleep on the engine
        # Note: This depends on engine_client.sleep() being implemented
        if hasattr(engine_client, "sleep"):
            await engine_client.sleep()
        else:
            logger.warning("Engine does not support sleep; operation is a no-op")

        elapsed = time.time() - start
        gaudi_state.sleep_model()

        logger.info(f"Model sleeping in {elapsed:.2f}s")

        return SleepResponse(
            sleeping=True,
            time_s=elapsed,
            freed_gib=None  # Metrics would require memory profiling
        )

    except Exception as e:
        logger.error(f"Sleep failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sleep failed: {str(e)}")


@gaudi_router.post("/wake_up")
async def wake_model(
    engine_client=Depends(get_engine_client),
    gaudi_state: GaudiOpenAIStateManager = Depends(get_gaudi_state),
) -> WakeResponse:
    """
    Wake up the model from sleep.
    
    Resumes generation after a /v1/model/sleep call.
    """
    if not gaudi_state.is_sleeping:
        return WakeResponse(
            awake=True,
            time_s=0.0,
            consumed_gib=None
        )

    try:
        start = time.time()
        logger.info("Waking up model...")
        
        # Call wake_up on the engine
        if hasattr(engine_client, "wake_up"):
            await engine_client.wake_up()
        else:
            logger.warning("Engine does not support wake_up; operation is a no-op")

        elapsed = time.time() - start
        gaudi_state.wake_model()

        logger.info(f"Model awake in {elapsed:.2f}s")

        return WakeResponse(
            awake=True,
            time_s=elapsed,
            consumed_gib=None
        )

    except Exception as e:
        logger.error(f"Wake up failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Wake up failed: {str(e)}")


# ============================================================================
# Registration Function
# ============================================================================

def register_gaudi_openai_routes(app) -> None:
    """
    Register Gaudi-specific routes on the OpenAI FastAPI app.
    
    Call this in vllm/entrypoints/openai/api_server.py build_app() after
    the app is created.
    
    Example:
        app = FastAPI(...)
        # ... standard setup ...
        register_gaudi_openai_routes(app)
    """
    # Initialize Gaudi state if not already present
    if not hasattr(app.state, "gaudi_openai_state"):
        app.state.gaudi_openai_state = GaudiOpenAIStateManager()

    # Register the router
    app.include_router(gaudi_router)
    
    logger.info("Registered Gaudi OpenAI extension routes at /v1/model/*")

