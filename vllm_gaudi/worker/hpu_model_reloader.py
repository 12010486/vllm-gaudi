# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Support for dynamic model reloading on HPU workers.

This module provides a mixin class that can be added to HPU workers
to enable dynamic model switching without recreating the worker.
"""

from typing import Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class HPUModelReloaderMixin:
    """
    Mixin for HPU workers to support dynamic model reloading.
    
    This should be mixed into the HPU worker class to add
    model switching capabilities. The worker can then reload
    a new model without being destroyed and recreated.
    
    Example:
        class MyHPUWorker(HPUModelReloaderMixin, BaseWorker):
            def _create_model_runner(self):
                # Implementation specific to your worker
                pass
    
    The mixin provides:
    - load_model(): Method to dynamically reload a model
    - Proper cleanup of old model and HPU resources
    - Config updates for the new model
    """
    
    def _clear_model_and_cache(self) -> None:
        """Clear the current model runner and free HPU cache."""
        if hasattr(self, 'model_runner') and self.model_runner is not None:
            logger.info("[HPUWorker] Clearing current model from memory...")

            # Explicitly delete model to free memory
            if hasattr(self.model_runner, 'model') and self.model_runner.model is not None:
                try:
                    import torch
                    for param in self.model_runner.model.parameters():
                        param.data = torch.empty(0, device='cpu')
                except Exception as e:
                    logger.warning(f"[HPUWorker] Error clearing model parameters: {e}")

                del self.model_runner.model
                self.model_runner.model = None

            del self.model_runner
            self.model_runner = None

        try:
            import habana_frameworks.torch as htorch
            if hasattr(htorch, 'hpu'):
                htorch.hpu.empty_cache()
                logger.info("[HPUWorker] Cleared HPU cache")
        except Exception as e:
            logger.warning(f"[HPUWorker] Could not clear HPU cache: {e}")

    def unload_model(self) -> None:
        """
        Unload the current model and free resources.

        This method should be called after the model has been put to sleep
        to release HPU memory without shutting down the worker process.
        """
        logger.info("[HPUWorker] Unloading current model")
        self._clear_model_and_cache()

    def load_model(self, vllm_config: VllmConfig) -> None:
        """
        Load a new model, replacing the current one.
        
        This method performs the following steps:
        1. Clears the current model from memory
        2. Clears HPU cache
        3. Updates worker configuration
        4. Loads the new model
        5. Reinitializes the model runner
        
        This method should be called via collective RPC from the
        engine when switching models.
        
        Args:
            vllm_config: New model configuration containing model_config,
                parallel_config, scheduler_config, etc.
        
        Note:
            The model should be in sleep state (weights on CPU) before
            calling this method for optimal memory management.
        """
        new_model = vllm_config.model_config.model
        logger.info(f"[HPUWorker] Reloading model: {new_model}")

        # Step 1: Clear current model
        self._clear_model_and_cache()

        # Step 2: Update all configs
        logger.info("[HPUWorker] Updating worker configuration...")
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.cache_config = vllm_config.cache_config
        self.load_config = vllm_config.load_config
        
        # Optional configs that may not exist
        if hasattr(vllm_config, 'lora_config'):
            self.lora_config = vllm_config.lora_config
        if hasattr(vllm_config, 'vision_language_config'):
            self.vision_language_config = vllm_config.vision_language_config
        if hasattr(vllm_config, 'speculative_config'):
            self.speculative_config = vllm_config.speculative_config
        if hasattr(vllm_config, 'prompt_adapter_config'):
            self.prompt_adapter_config = vllm_config.prompt_adapter_config
        if hasattr(vllm_config, 'observability_config'):
            self.observability_config = vllm_config.observability_config
        
        # Step 4: Create and load new model runner
        logger.info(f"[HPUWorker] Creating model runner for: {new_model}")
        self.model_runner = self._create_model_runner()
        
        logger.info(f"[HPUWorker] Loading model: {new_model}")
        self.model_runner.load_model()
        
        logger.info(f"[HPUWorker] Successfully reloaded model: {new_model}")
    
    def _create_model_runner(self):
        """
        Create a new model runner instance.
        
        This method should be implemented by the subclass to return
        an appropriate model runner for the worker type.
        
        Returns:
            Model runner instance
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclass must implement _create_model_runner() to create "
            "a model runner appropriate for this worker type."
        )


def add_model_reloader_to_worker(worker_class):
    """
    Decorator to add model reloading capability to a worker class.
    
    This can be used to dynamically add the HPUModelReloaderMixin
    to an existing worker class without modifying its definition.
    
    Example:
        @add_model_reloader_to_worker
        class MyWorker(BaseWorker):
            pass
    
    Args:
        worker_class: Worker class to enhance
    
    Returns:
        Enhanced worker class with model reloading capability
    """
    if not hasattr(worker_class, 'load_model') or not hasattr(worker_class, 'unload_model'):
        # Create a new class that inherits from both mixin and original
        enhanced_class = type(
            worker_class.__name__,
            (HPUModelReloaderMixin, worker_class),
            {}
        )
        return enhanced_class
    return worker_class

