# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model support for AsyncLLM on Gaudi platform.

This module provides a wrapper around AsyncLLM that enables dynamic model
switching without destroying and recreating the engine. This is particularly
useful for scenarios like:
- Multi-tenant serving with different models per tenant
- A/B testing with multiple models
- Dynamic model selection based on workload

The implementation uses vLLM's sleep/wake mechanism combined with dynamic
model reloading through the collective RPC interface.
"""

from typing import Dict, Optional, Any, AsyncGenerator
import asyncio
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.inputs import PromptType, ProcessorInputs
from vllm.v1.engine import PauseMode

logger = init_logger(__name__)


class MultiModelAsyncLLM:
    """
    Wrapper around AsyncLLM that supports loading multiple models
    dynamically without stopping the engine.
    
    This is achieved by:
    1. Maintaining configs for multiple models
    2. Using sleep() to offload current model to CPU
    3. Reloading model executor with new config via collective RPC
    4. Using wake_up() to resume with new model
    
    Example:
        >>> from vllm.engine.arg_utils import AsyncEngineArgs
        >>> from vllm_gaudi.engine import MultiModelAsyncLLM
        >>> 
        >>> # Define models
        >>> models = {
        ...     "model_a": AsyncEngineArgs(
        ...         model="meta-llama/Llama-3.1-8B-Instruct",
        ...         max_model_len=4096,
        ...     ),
        ...     "model_b": AsyncEngineArgs(
        ...         model="Qwen/Qwen3-0.6B",
        ...         max_model_len=4096,
        ...     )
        ... }
        >>> 
        >>> # Create manager
        >>> manager = MultiModelAsyncLLM(models)
        >>> await manager.initialize("model_a")
        >>> 
        >>> # Generate with model_a
        >>> from vllm import SamplingParams
        >>> async for output in manager.generate(
        ...     "Hello, my name is",
        ...     SamplingParams(max_tokens=20),
        ...     "request-1"
        ... ):
        ...     print(output.outputs[0].text)
        >>> 
        >>> # Switch to model_b
        >>> await manager.switch_model("model_b")
        >>> 
        >>> # Generate with model_b
        >>> async for output in manager.generate(
        ...     "The capital of France is",
        ...     SamplingParams(max_tokens=20),
        ...     "request-2"
        ... ):
        ...     print(output.outputs[0].text)
        >>> 
        >>> # Cleanup
        >>> manager.shutdown()
    
    Note:
        - Model switching incurs overhead (sleep, reload, wake operations)
        - CPU must have sufficient memory to hold sleeping models
        - Some configs (like tensor_parallel_size) should match across models
        - KV cache is cleared during switch - no context preservation
    """
    
    def __init__(
        self,
        model_configs: Dict[str, AsyncEngineArgs],
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        disable_log_stats: bool = False,
        enable_log_requests: bool = False,
        stat_loggers: Optional[list] = None,
    ):
        """
        Initialize multi-model manager.
        
        Args:
            model_configs: Dictionary mapping model names to their AsyncEngineArgs.
                Each model's configuration should be compatible in terms of
                parallel configs (tensor_parallel_size, pipeline_parallel_size, etc.)
            usage_context: Usage context for the engine
            disable_log_stats: Whether to disable stats logging
            enable_log_requests: Whether to enable request logging
            stat_loggers: Optional custom stat loggers
        """
        if not model_configs:
            raise ValueError("model_configs cannot be empty")
        
        self.model_configs = model_configs
        self.usage_context = usage_context
        self.disable_log_stats = disable_log_stats
        self.enable_log_requests = enable_log_requests
        self.stat_loggers = stat_loggers
        
        self._engine: Optional[AsyncLLM] = None
        self._current_model_name: Optional[str] = None
        self._vllm_configs: Dict[str, VllmConfig] = {}
        self._switching_lock = asyncio.Lock()
        
        # Pre-create VllmConfig for each model
        logger.info(f"Pre-creating configs for {len(model_configs)} models...")
        for name, args in model_configs.items():
            self._vllm_configs[name] = args.create_engine_config(usage_context)
            logger.info(f"  - {name}: {self._vllm_configs[name].model_config.model}")
    
    @property
    def current_model(self) -> Optional[str]:
        """Return the name of the currently loaded model."""
        return self._current_model_name
    
    @property
    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.model_configs.keys())

    def get_vllm_config(self, model_name: str) -> VllmConfig:
        """Return the VllmConfig for a specific model name."""
        if model_name not in self._vllm_configs:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.model_configs.keys())}"
            )
        return self._vllm_configs[model_name]

    def get_all_vllm_configs(self) -> Dict[str, VllmConfig]:
        """Return a shallow copy of all precomputed VllmConfigs."""
        return dict(self._vllm_configs)
    
    @property
    def engine(self) -> AsyncLLM:
        """Return the underlying AsyncLLM engine."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._engine
    
    async def initialize(self, model_name: str) -> None:
        """
        Initialize the engine with a specific model.
        
        This creates the AsyncLLM engine and loads the specified model.
        Must be called before any other operations.
        
        Args:
            model_name: Name of the model to load (must be in model_configs)
        
        Raises:
            ValueError: If model_name is not in model_configs
            RuntimeError: If engine is already initialized
        """
        if model_name not in self.model_configs:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.model_configs.keys())}"
            )
        
        if self._engine is not None:
            raise RuntimeError(
                "Engine already initialized. Use switch_model() to change models."
            )
        
        logger.info(f"Initializing engine with model: {model_name}")
        args = self.model_configs[model_name]

        args.disable_log_stats = self.disable_log_stats
        args.enable_log_requests = self.enable_log_requests
        
        # Create the AsyncLLM engine
        self._engine = AsyncLLM.from_engine_args(
            args,
            start_engine_loop=True,
            usage_context=self.usage_context,
            stat_loggers=self.stat_loggers,
        )
        self._current_model_name = model_name
        
        logger.info(
            f"Successfully initialized engine with model: "
            f"{self._vllm_configs[model_name].model_config.model}"
        )
    
    async def switch_model(
        self,
        model_name: str,
        drain_timeout: int = 60,
        sleep_level: int = 1,
        pause_mode: PauseMode = "abort",
    ) -> None:
        """
        Switch to a different model without destroying the engine.
        
        This method performs the following steps:
        1. Waits for pending requests to complete (with timeout)
        2. Puts current model to sleep (moves weights to CPU)
        3. Reinitializes the model executor with new config
        4. Wakes up with the new model
        
        Args:
            model_name: Name of the model to switch to
            drain_timeout: Seconds to wait for pending requests to drain
            sleep_level: Sleep level (1 = move to CPU, preserves state)
            pause_mode: How to handle in-flight requests:
                - "abort": Abort all in-flight requests immediately
                - "wait": Wait for in-flight requests to complete
                - "keep": Keep requests in queue (not recommended for model switch)
        
        Raises:
            ValueError: If model_name is not in model_configs
            RuntimeError: If engine not initialized
            TimeoutError: If drain timeout is exceeded
        """
        async with self._switching_lock:
            if self._engine is None:
                raise RuntimeError("Engine not initialized. Call initialize() first.")
            
            if model_name not in self.model_configs:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(self.model_configs.keys())}"
                )
            
            if model_name == self._current_model_name:
                logger.info(f"Model '{model_name}' already loaded, no switch needed.")
                return
            
            old_model = self._vllm_configs[self._current_model_name].model_config.model
            new_model = self._vllm_configs[model_name].model_config.model
            
            logger.info(
                f"Switching model from '{self._current_model_name}' "
                f"({old_model}) to '{model_name}' ({new_model})"
            )
            
            # Step 1: Drain pending requests
            logger.info("Draining pending requests...")
            try:
                await asyncio.wait_for(
                    self._engine.wait_for_requests_to_drain(drain_timeout),
                    timeout=drain_timeout + 5,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Drain timeout ({drain_timeout}s) exceeded. "
                    f"Proceeding with model switch anyway."
                )
            
            # Step 2: Sleep current model (offload to CPU)
            logger.info(f"Sleeping current model '{self._current_model_name}'...")
            await self._engine.sleep(level=sleep_level, mode=pause_mode)
            
            # Step 3: Reload model executor with new config
            logger.info(f"Reinitializing model executor for '{model_name}'...")
            await self._reload_model_executor(model_name)
            
            # Step 4: Wake up with new model
            logger.info(f"Waking up with new model '{model_name}'...")
            await self._engine.wake_up()
            
            self._current_model_name = model_name
            logger.info(f"Successfully switched to model: {new_model}")
    
    async def _reload_model_executor(self, model_name: str) -> None:
        """
        Reload the model executor with a new model config.
        
        This uses the collective_rpc mechanism to call model loading
        on the executor workers. The actual implementation depends on
        the worker having a load_model() method that supports dynamic
        model reloading.
        
        Args:
            model_name: Name of the model to load
        
        Raises:
            RuntimeError: If model reload fails
        """
        new_config = self._vllm_configs[model_name]
        
        # Use collective RPC to reload model on all workers
        # This leverages the existing model loading infrastructure
        try:
            # The worker needs to implement a load_model method that accepts
            # a VllmConfig and reloads the model
            await self._engine.collective_rpc(
                method="load_model",
                kwargs={"vllm_config": new_config},
                timeout=300.0,  # 5 minutes timeout for model loading
            )
        except Exception as e:
            logger.error(f"Failed to reload model executor: {e}")
            raise RuntimeError(f"Model reload failed: {e}") from e
    
    async def generate(
        self,
        prompt: PromptType | ProcessorInputs,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[Dict[str, Any]] = None,
        trace_headers: Optional[Dict[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Generate completion for a prompt using the current model.
        
        All arguments are passed through to AsyncLLM.generate().
        See AsyncLLM.generate() for detailed parameter documentation.
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters for generation
            request_id: Unique request identifier
            **kwargs: Additional arguments passed to AsyncLLM.generate()
        
        Yields:
            RequestOutput: Generation outputs
        
        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        async for output in self._engine.generate(
            prompt,
            sampling_params,
            request_id,
            prompt_text=prompt_text,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        ):
            yield output
    
    async def encode(
        self,
        prompt: PromptType | ProcessorInputs,
        pooling_params: PoolingParams,
        request_id: str,
        *,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Dict[str, str]] = None,
        priority: int = 0,
        tokenization_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Encode input for pooling/embedding models.
        
        All arguments are passed through to AsyncLLM.encode().
        See AsyncLLM.encode() for detailed parameter documentation.
        
        Args:
            prompt: Input prompt
            pooling_params: Pooling parameters
            request_id: Unique request identifier
            **kwargs: Additional arguments passed to AsyncLLM.encode()
        
        Yields:
            PoolingRequestOutput: Encoding outputs
        
        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        async for output in self._engine.encode(
            prompt,
            pooling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        ):
            yield output
    
    async def abort(
        self,
        request_id: str | list[str],
        internal: bool = False,
    ) -> None:
        """
        Abort one or more requests.
        
        Args:
            request_id: Request ID(s) to abort
            internal: Whether this is an internal abort
        """
        if self._engine is not None:
            await self._engine.abort(request_id, internal=internal)
    
    async def get_model_config(self) -> VllmConfig:
        """
        Get the configuration of the currently loaded model.
        
        Returns:
            VllmConfig: Current model configuration
        
        Raises:
            RuntimeError: If engine not initialized
        """
        if self._current_model_name is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._vllm_configs[self._current_model_name]
    
    async def is_sleeping(self) -> bool:
        """Check if the engine is currently in sleep mode."""
        if self._engine is None:
            return False
        return await self._engine.is_sleeping()
    
    def shutdown(self):
        """
        Shutdown the engine and clean up resources.
        
        This should be called when done using the multi-model manager.
        """
        if self._engine is not None:
            logger.info("Shutting down multi-model engine...")
            self._engine.shutdown()
            self._engine = None
            self._current_model_name = None
            logger.info("Shutdown complete.")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.shutdown()
