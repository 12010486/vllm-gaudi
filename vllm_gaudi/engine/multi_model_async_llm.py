# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simplified multi-model support for AsyncLLM on Gaudi platform.

This is a simplified version that removes complex mode/pause handling
and focuses on core functionality: initialize -> generate -> switch -> generate.
"""

from typing import Optional
from collections.abc import AsyncGenerator
import asyncio
import cloudpickle
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.inputs import PromptType, ProcessorInputs

logger = init_logger(__name__)


class MultiModelAsyncLLM:
    """
    Simplified wrapper around AsyncLLM for dynamic model switching.
    
    Usage flow:
    1. Create with model configs: MultiModelAsyncLLM({"model_a": config_a, "model_b": config_b})
    2. Initialize with first model: await manager.initialize("model_a")
    3. Generate: async for output in manager.generate(prompt, params, request_id): ...
    4. Switch models: await manager.switch_model("model_b")
    5. Generate with new model
    6. Cleanup: manager.shutdown()
    
    Example:
        >>> from vllm.engine.arg_utils import AsyncEngineArgs
        >>> from vllm_gaudi.engine import MultiModelAsyncLLM
        >>> 
        >>> models = {
        ...     "model_a": AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct"),
        ...     "model_b": AsyncEngineArgs(model="Qwen/Qwen3-0.6B"),
        ... }
        >>> manager = MultiModelAsyncLLM(models)
        >>> await manager.initialize("model_a")
        >>> async for output in manager.generate("Hello", SamplingParams(max_tokens=20), "req-1"):
        ...     print(output.outputs[0].text)
        >>> await manager.switch_model("model_b")
        >>> manager.shutdown()
    """

    def __init__(
        self,
        model_configs: dict[str, AsyncEngineArgs],
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        disable_log_stats: bool = False,
        enable_log_requests: bool = False,
    ):
        """
        Initialize multi-model manager.
        
        Args:
            model_configs: Dict mapping model names to AsyncEngineArgs
            usage_context: Engine usage context
            disable_log_stats: Disable stats logging
            enable_log_requests: Enable request logging
        """
        self._engine: Optional[AsyncLLM] = None
        self._sleeping: dict[str, bool] = {}
        self._current_model_name: Optional[str] = None
        self._vllm_configs: dict[str, VllmConfig] = {}
        self._switching_lock = asyncio.Lock()

        if not model_configs:
            raise ValueError("model_configs cannot be empty")

        self.model_configs = model_configs
        self.usage_context = usage_context
        self.disable_log_stats = disable_log_stats
        self.enable_log_requests = enable_log_requests

        # Pre-create VllmConfig for each model
        logger.info(f"Creating configs for {len(model_configs)} models")
        for name, args in model_configs.items():
            self._vllm_configs[name] = args.create_engine_config(usage_context)
            logger.info(f"  {name}: {self._vllm_configs[name].model_config.model}")

    @property
    def current_model(self) -> Optional[str]:
        """Return currently loaded model name."""
        return self._current_model_name

    @property
    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.model_configs.keys())

    def get_vllm_config(self, model_name: str) -> VllmConfig:
        """Get VllmConfig for a model."""
        if model_name not in self._vllm_configs:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")
        return self._vllm_configs[model_name]

    def get_all_vllm_configs(self) -> dict[str, VllmConfig]:
        """
        Get all vllm_configs for model registry building.

        Returns a shallow copy to prevent external modification.

        Returns:
            Dictionary mapping model names to their VllmConfig objects
        """
        return self._vllm_configs.copy()

    @property
    def engine(self) -> AsyncLLM:
        """Return underlying AsyncLLM engine."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._engine

    async def initialize(self, model_name: str) -> None:
        """
        Initialize engine with a model.
        
        Args:
            model_name: Model to load (must be in model_configs)
        
        Raises:
            ValueError: If model_name not found
            RuntimeError: If already initialized
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")

        if self._engine is not None:
            raise RuntimeError("Engine already initialized. Use switch_model() instead.")
        logger.info(f"Initializing engine with: {model_name}")
        args = self.model_configs[model_name]
        args.disable_log_stats = self.disable_log_stats
        args.enable_log_requests = self.enable_log_requests

        self._engine = AsyncLLM.from_engine_args(
            args,
            start_engine_loop=True,
            usage_context=self.usage_context,
        )
        self._sleeping[model_name] = False
        logger.info(f"Model sleep state: {model_name}=awake")

        self._current_model_name = model_name

        logger.info(f"Engine initialized with: "
                    f"{self._vllm_configs[model_name].model_config.model}")

    async def switch_model(
        self,
        model_name: str,
        drain_timeout: int = 60,
    ) -> None:
        """
        Switch to a different model with error recovery
        
        Steps:
        1. Drain pending requests (with timeout)
        2. Sleep current model (free KV cache + weights)
        3. Unload current model weights
        4. Reload new model on the same engine
        5. Reinitialize KV cache for new model
        
        If any step fails, attempts to wake up engine to restore state.

        Args:
            model_name: Target model name
            drain_timeout: Seconds to wait for requests to drain
            sleep_level: Sleep level (1 = CPU offload)
        
        Raises:
            ValueError: If model not found
            RuntimeError: If engine not initialized or switch fails

        """
        async with self._switching_lock:
            if self._engine is None:
                raise RuntimeError("Engine not initialized. Call initialize() first.")

            if model_name not in self.model_configs:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")

            if model_name == self._current_model_name:
                logger.info(f"Model '{model_name}' already loaded.")
                return

            current_config = self._vllm_configs[self._current_model_name]
            target_config = self._vllm_configs[model_name]
            current_model_cfg = current_config.model_config
            target_model_cfg = target_config.model_config

            # Single-engine AsyncLLM hot-swap is only safe when scheduler/
            # tokenizer-sensitive runtime shape is effectively unchanged.
            incompatible = (current_model_cfg.model != target_model_cfg.model
                            or current_model_cfg.tokenizer != target_model_cfg.tokenizer
                            or current_model_cfg.max_model_len != target_model_cfg.max_model_len)
            if incompatible:
                raise RuntimeError("Single-engine hot-swap across different model/tokenizer "
                                   "configurations is not supported for AsyncLLM. "
                                   "Use engine reinitialization (shutdown + initialize) or "
                                   "switch to LLM with VLLM_ENABLE_V1_MULTIPROCESSING=0.")

            old_model = self._vllm_configs[self._current_model_name].model_config.model
            new_model = self._vllm_configs[model_name].model_config.model

            logger.info(f"Switching from {self._current_model_name} to {model_name}")

            try:
                # Step 1: Drain pending requests
                logger.info("Draining pending requests...")
                try:
                    await asyncio.wait_for(
                        self._engine.wait_for_requests_to_drain(drain_timeout),
                        timeout=drain_timeout + 5,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Drain timeout ({drain_timeout}s) exceeded. Proceeding with caution.")

                # Step 2: Sleep current model (free memory)
                logger.info(f"Sleeping model: {self._current_model_name}")
                await self._engine.sleep(level=1)
                self._sleeping[self._current_model_name] = True
                logger.info(f"Model sleep state: {self._current_model_name}=sleeping")

                # Step 3: Unload current model weights
                logger.info(f"Unloading model: {self._current_model_name}")
                await self._unload_model_executor()

                # Step 4: Reload new model on same engine
                logger.info(f"Reloading executor for: {model_name}")
                await self._reload_model_executor(model_name)

                # Step 5: Reinitialize KV cache for new model
                logger.info("Reinitializing KV cache after model reload")
                await self._engine.wake_up(tags=["kv_cache"])
                self._sleeping[model_name] = False
                logger.info(f"Model sleep state: {model_name}=awake")

                self._current_model_name = model_name
                logger.info(f"Successfully switched to: {new_model}")

            except Exception as e:
                logger.error(f"Model switch failed during {e.__class__.__name__}: {e}. "
                             f"Attempting to restore engine state...")
                # Attempt recovery: wake up engine if it's stuck in sleep
                try:
                    logger.info("Attempting to wake up engine for recovery...")
                    await self._engine.wake_up(tags=["weights", "kv_cache"])
                    if self._current_model_name is not None:
                        self._sleeping[self._current_model_name] = False
                        logger.info(f"Model sleep state: {self._current_model_name}=awake")
                    logger.warning("Engine woken up. May still be in inconsistent state. "
                                   "Manual restart recommended if issues persist.")
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error.__class__.__name__}: {recovery_error}. "
                                 f"Engine may be unresponsive. Manual server restart required.")

                # Re-raise original exception with context
                raise RuntimeError(f"Failed to switch model from {self._current_model_name} to {model_name}: {e}")

    async def _reload_model_executor(self, model_name: str) -> None:
        """Reload model executor with new config via collective RPC."""
        new_config = self._vllm_configs[model_name]
        serialized_config = cloudpickle.dumps(new_config)

        try:
            await self._engine.collective_rpc(
                method="load_model",
                kwargs={"vllm_config_bytes": serialized_config},
                timeout=300.0,
            )
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            raise RuntimeError(f"Model reload failed: {e}") from e

    async def _unload_model_executor(self) -> None:
        """Unload current model weights via collective RPC."""
        try:
            await self._engine.collective_rpc(
                method="unload_model",
                kwargs={},
                timeout=120.0,
            )
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            raise RuntimeError(f"Model unload failed: {e}") from e

    async def generate(
        self,
        prompt: PromptType | ProcessorInputs,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Generate completion for prompt.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Unique request ID
            **kwargs: Additional args passed to AsyncLLM.generate()
        
        Yields:
            RequestOutput: Generation outputs
        
        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized.")

        async for output in self._engine.generate(prompt, sampling_params, request_id, **kwargs):
            yield output

    async def encode(
        self,
        prompt: PromptType | ProcessorInputs,
        pooling_params: PoolingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Encode input for embedding/pooling models.
        
        Args:
            prompt: Input prompt
            pooling_params: Pooling parameters
            request_id: Unique request ID
            **kwargs: Additional args passed to AsyncLLM.encode()
        
        Yields:
            PoolingRequestOutput: Encoding outputs
        
        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized.")

        async for output in self._engine.encode(prompt, pooling_params, request_id, **kwargs):
            yield output

    async def abort(self, request_id: str | list[str]) -> None:
        """Abort request(s)."""
        if self._engine is not None:
            await self._engine.abort(request_id)

    def shutdown(self):
        """Shutdown engine and cleanup."""
        if self._engine is not None:
            logger.info("Shutting down multi-model engine")
            self._engine.shutdown()
            self._engine = None
        self._sleeping.clear()
        self._current_model_name = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

    async def __aenter__(self):
        """Async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager."""
        self.shutdown()
