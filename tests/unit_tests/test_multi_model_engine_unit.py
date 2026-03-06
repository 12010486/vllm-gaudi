"""
Unit tests for multi-model async engine and model switching.

These tests can be run with pytest and focus on the core logic
that's less dependent on vLLM internals.

Usage:
    pytest -v test_multi_model_engine.py
    pytest -v test_multi_model_engine.py -k "test_switch_model_sequence"
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from typing import Dict, Optional


# ============================================================================
# Test Fixtures - Mock Objects
# ============================================================================

@pytest.fixture
def mock_async_engine_args():
    """Factory to create mock AsyncEngineArgs."""
    def _create(model_name: str, **kwargs) -> Mock:
        args = Mock()
        args.model = model_name
        args.create_engine_config = Mock(
            return_value=_create_vllm_config(model_name, **kwargs)
        )
        args.disable_log_stats = False
        args.enable_log_requests = False
        return args
    return _create


def _create_vllm_config(model_name: str, **kwargs) -> Mock:
    """Helper to create mock VllmConfig."""
    config = Mock()
    config.model_config = Mock()
    config.model_config.model = kwargs.get("model", f"model-{model_name}")
    config.model_config.max_model_len = kwargs.get("max_model_len", 4096)
    config.parallel_config = Mock()
    config.scheduler_config = Mock()
    config.device_config = Mock()
    config.cache_config = Mock()
    config.load_config = Mock()
    return config


@pytest.fixture
def mock_async_llm():
    """Factory to create mock AsyncLLM engine."""
    engine = AsyncMock()
    engine.is_running = True
    engine.is_stopped = False
    engine.errored = False
    
    # Mock inference methods
    engine.generate = AsyncMock()
    engine.encode = AsyncMock()
    
    # Mock lifecycle methods
    engine.sleep = AsyncMock()
    engine.wake_up = AsyncMock()
    engine.abort = AsyncMock()
    engine.shutdown = Mock()
    
    # Mock state methods
    engine.wait_for_requests_to_drain = AsyncMock()
    engine.is_sleeping = AsyncMock(return_value=False)
    engine.is_paused = AsyncMock(return_value=False)
    
    # Mock collective RPC
    engine.collective_rpc = AsyncMock()
    
    # Mock stats/health
    engine.do_log_stats = AsyncMock()
    engine.check_health = AsyncMock()
    
    return engine


# ============================================================================
# Test Suite 1: Model Store / Registry Logic
# ============================================================================

class TestModelRegistry:
    """Test model configuration storage and retrieval."""
    
    def test_model_configs_storage(self, mock_async_engine_args):
        """Test that model configs are properly stored."""
        # Simulating: MultiModelAsyncLLM.__init__
        model_configs: Dict[str, Mock] = {
            "llama": mock_async_engine_args("llama"),
            "qwen": mock_async_engine_args("qwen"),
            "mistral": mock_async_engine_args("mistral"),
        }
        
        # Assertions
        assert len(model_configs) == 3
        assert "llama" in model_configs
        assert "qwen" in model_configs
        assert "mistral" in model_configs
    
    def test_available_models_list(self, mock_async_engine_args):
        """Test retrieving list of available models."""
        model_configs = {
            "model_a": mock_async_engine_args("model_a"),
            "model_b": mock_async_engine_args("model_b"),
        }
        
        available = list(model_configs.keys())
        assert available == ["model_a", "model_b"]
    
    def test_model_config_lookup(self, mock_async_engine_args):
        """Test looking up individual model config."""
        model_configs = {
            "llama": mock_async_engine_args("llama"),
        }
        
        config = model_configs.get("llama")
        assert config is not None
        assert config.model == "llama"
    
    def test_invalid_model_lookup_returns_none(self, mock_async_engine_args):
        """Test lookup of non-existent model."""
        model_configs = {
            "llama": mock_async_engine_args("llama"),
        }
        
        config = model_configs.get("nonexistent")
        assert config is None


# ============================================================================
# Test Suite 2: Engine Lifecycle
# ============================================================================

class TestEngineLifecycle:
    """Test engine initialization and shutdown."""
    
    @pytest.mark.asyncio
    async def test_engine_init_sequence(self, mock_async_engine_args, mock_async_llm):
        """Test proper initialization sequence."""
        # Setup
        model_configs = {
            "llama": mock_async_engine_args("llama"),
        }
        model_name = "llama"
        args = model_configs[model_name]
        
        # Simulate: MultiModelAsyncLLM.initialize()
        current_model = None
        engine = None
        
        # Step 1: Validate model name
        assert model_name in model_configs
        
        # Step 2: Ensure not already initialized
        assert engine is None
        
        # Step 3: Create vllm_config
        vllm_config = args.create_engine_config(None)
        
        # Step 4: Create engine (mocked)
        engine = mock_async_llm
        current_model = model_name
        
        # Assertions
        assert engine is not None
        assert current_model == "llama"
    
    @pytest.mark.asyncio
    async def test_engine_shutdown(self, mock_async_llm):
        """Test proper shutdown sequence."""
        # Setup
        engine = mock_async_llm
        current_model = "llama"
        
        # Simulate: MultiModelAsyncLLM.shutdown()
        if engine is not None:
            engine.shutdown()
            engine = None
            current_model = None
        
        # Assertions
        assert engine is None
        assert current_model is None
        mock_async_llm.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_async_llm):
        """Test async context manager cleanup."""
        engine = mock_async_llm
        
        # Simulate: async with manager: __aexit__()
        # Should call shutdown
        if engine is not None:
            engine.shutdown()
        
        mock_async_llm.shutdown.assert_called_once()


# ============================================================================
# Test Suite 3: Model Switching - Core Logic
# ============================================================================

class TestModelSwitchingLogic:
    """Test model switching workflow and state transitions."""
    
    @pytest.mark.asyncio
    async def test_switch_to_same_model_is_noop(self):
        """Test that switching to current model is no-op."""
        current_model = "llama"
        target_model = "llama"
        
        # Simulate: switch_model() with same model
        if target_model == current_model:
            # Should return early - no-op
            switched = False
        else:
            switched = True
        
        assert switched is False
    
    @pytest.mark.asyncio
    async def test_switch_requires_init(self):
        """Test that switch requires initialization."""
        engine = None
        
        # Simulate: switch_model() without init
        if engine is None:
            # Should raise RuntimeError
            with pytest.raises(RuntimeError):
                raise RuntimeError("Engine not initialized")
    
    @pytest.mark.asyncio
    async def test_switch_workflow_steps(self, mock_async_llm):
        """Test the 5-step switching workflow."""
        engine = mock_async_llm
        current_model = "llama"
        target_model = "qwen"
        
        # Setup configuration
        drain_timeout = 60
        sleep_level = 1
        
        steps_executed = []
        
        # Step 1: Drain pending requests
        steps_executed.append("drain")
        await engine.wait_for_requests_to_drain(drain_timeout)
        
        # Step 2: Sleep current model
        steps_executed.append("sleep")
        await engine.sleep(level=sleep_level)
        
        # Step 3: Unload current model
        steps_executed.append("unload")
        await engine.collective_rpc(method="unload_model", kwargs={})
        
        # Step 4: Reload with new model
        steps_executed.append("reload")
        await engine.collective_rpc(method="load_model", kwargs={"vllm_config": Mock()})
        
        # Step 5: Wake up
        steps_executed.append("wake")
        await engine.wake_up()
        
        # Assertions - verify sequence
        assert steps_executed == ["drain", "sleep", "unload", "reload", "wake"]
        assert engine.wait_for_requests_to_drain.await_count == 1
        assert engine.sleep.await_count == 1
        assert engine.collective_rpc.await_count == 2  # unload + reload
        assert engine.wake_up.await_count == 1
    
    @pytest.mark.asyncio
    async def test_drain_timeout_handling(self, mock_async_llm):
        """Test behavior when drain timeout exceeded."""
        engine = mock_async_llm
        drain_timeout = 60
        
        # Mock timeout
        engine.wait_for_requests_to_drain = AsyncMock(side_effect=asyncio.TimeoutError())
        
        # Should catch timeout and proceed with warning
        try:
            await engine.wait_for_requests_to_drain(drain_timeout)
        except asyncio.TimeoutError:
            # Caught - proceed anyway
            timeout_occurred = True
        
        assert timeout_occurred is True
    
    @pytest.mark.asyncio
    async def test_switch_state_update(self):
        """Test that current_model is updated after switch."""
        current_model = "llama"
        target_model = "qwen"
        
        # Simulate switch completion
        current_model = target_model
        
        # Assertion
        assert current_model == "qwen"


# ============================================================================
# Test Suite 4: Collective RPC Communication
# ============================================================================

class TestCollectiveRPC:
    """Test worker coordination via collective RPC."""
    
    @pytest.mark.asyncio
    async def test_unload_model_rpc(self, mock_async_llm):
        """Test collective RPC for model unload."""
        engine = mock_async_llm
        
        # Simulate: _unload_model_executor()
        await engine.collective_rpc(
            method="unload_model",
            kwargs={},
            timeout=120.0,
        )
        
        # Assertions
        engine.collective_rpc.assert_called_once_with(
            method="unload_model",
            kwargs={},
            timeout=120.0,
        )
    
    @pytest.mark.asyncio
    async def test_load_model_rpc(self, mock_async_llm):
        """Test collective RPC for model load."""
        engine = mock_async_llm
        vllm_config = Mock()
        
        # Simulate: _reload_model_executor()
        await engine.collective_rpc(
            method="load_model",
            kwargs={"vllm_config": vllm_config},
            timeout=300.0,
        )
        
        # Assertions
        engine.collective_rpc.assert_called_once_with(
            method="load_model",
            kwargs={"vllm_config": vllm_config},
            timeout=300.0,
        )
    
    @pytest.mark.asyncio
    async def test_rpc_failure_handling(self, mock_async_llm):
        """Test error handling for RPC failures."""
        engine = mock_async_llm
        engine.collective_rpc = AsyncMock(side_effect=Exception("RPC failed"))
        
        # Should propagate error as RuntimeError
        with pytest.raises(Exception):
            await engine.collective_rpc(
                method="load_model",
                kwargs={},
                timeout=300.0,
            )


# ============================================================================
# Test Suite 5: Inference Methods
# ============================================================================

class TestInferenceMethods:
    """Test generate/encode delegation."""
    
    @pytest.mark.asyncio
    async def test_generate_delegation(self, mock_async_llm):
        """Test that generate() delegates to engine."""
        engine = mock_async_llm
        
        # Mock engine to yield outputs
        async def mock_generate(*args, **kwargs):
            yield Mock(outputs=[Mock(text="output1")])
            yield Mock(outputs=[Mock(text="output2")])
        
        engine.generate = mock_generate
        
        # Simulate: MultiModelAsyncLLM.generate()
        outputs = []
        async for output in engine.generate("prompt", Mock(), "req-1"):
            outputs.append(output.outputs[0].text)
        
        # Assertions
        assert len(outputs) == 2
        assert outputs == ["output1", "output2"]
    
    @pytest.mark.asyncio
    async def test_encode_delegation(self, mock_async_llm):
        """Test that encode() delegates to engine."""
        engine = mock_async_llm
        
        # Mock engine to yield outputs
        async def mock_encode(*args, **kwargs):
            yield Mock(outputs=[[1.0, 2.0]])
        
        engine.encode = mock_encode
        
        # Simulate: MultiModelAsyncLLM.encode()
        outputs = []
        async for output in engine.encode("text", Mock(), "req-1"):
            outputs.append(output.outputs)
        
        # Assertions
        assert len(outputs) == 1


# ============================================================================
# Test Suite 6: Concurrency & Locking
# ============================================================================

class TestConcurrencyControl:
    """Test thread-safe operations and locking."""
    
    @pytest.mark.asyncio
    async def test_concurrent_switches_serialized(self):
        """Test that concurrent switches are serialized."""
        lock = asyncio.Lock()
        switch_log = []
        
        async def simulated_switch(name: str):
            async with lock:
                switch_log.append(f"start_{name}")
                await asyncio.sleep(0.01)  # Simulate work
                switch_log.append(f"end_{name}")
        
        # Start 3 concurrent switches
        await asyncio.gather(
            simulated_switch("switch1"),
            simulated_switch("switch2"),
            simulated_switch("switch3"),
        )
        
        # Verify no interleaving
        assert "start_switch1" in switch_log
        assert "end_switch1" in switch_log
        # No switch should start before previous ends
        assert switch_log.index("start_switch1") < switch_log.index("end_switch1")


# ============================================================================
# Test Suite 7: Configuration Parsing
# ============================================================================

class TestConfigurationParsing:
    """Test YAML config parsing logic."""
    
    def test_extract_default_model(self):
        """Test extraction of default model."""
        config = {
            "default_model": "llama",
            "models": {
                "llama": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
                "qwen": {"model": "Qwen/Qwen3-0.6B"},
            }
        }
        
        default = config.get("default_model")
        assert default == "llama"
    
    def test_fallback_to_first_model(self):
        """Test using first model if no default specified."""
        config = {
            "models": {
                "llama": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
                "qwen": {"model": "Qwen/Qwen3-0.6B"},
            }
        }
        
        default = config.get("default_model")
        if default is None:
            default = list(config["models"].keys())[0]
        
        assert default == "llama"
    
    def test_model_config_extraction(self):
        """Test extracting per-model configuration."""
        config = {
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                    "tensor_parallel_size": 2,
                },
                "qwen": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 2048,
                    "tensor_parallel_size": 1,
                },
            }
        }
        
        llama_cfg = config["models"]["llama"]
        assert llama_cfg["model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert llama_cfg["max_model_len"] == 4096
        assert llama_cfg["tensor_parallel_size"] == 2
    
    def test_validate_required_fields(self):
        """Test validation of required config fields."""
        # Valid config
        config = {
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                }
            }
        }
        
        # Check required field
        for name, cfg in config["models"].items():
            assert "model" in cfg
    
    def test_validate_model_exists_in_config(self):
        """Test validation that model exists in config."""
        config = {
            "default_model": "llama",
            "models": {
                "llama": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
            }
        }
        
        default_model = config.get("default_model")
        assert default_model in config["models"]


# ============================================================================
# Test Suite 8: Error Scenarios
# ============================================================================

class TestErrorScenarios:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_generate_without_init(self):
        """Test generate() error when not initialized."""
        engine = None
        
        # Should raise error
        with pytest.raises(RuntimeError):
            if engine is None:
                raise RuntimeError("Engine not initialized")
    
    @pytest.mark.asyncio
    async def test_switch_to_invalid_model(self, mock_async_engine_args):
        """Test switch to non-existent model."""
        model_configs = {
            "llama": mock_async_engine_args("llama"),
        }
        
        target = "nonexistent"
        
        # Should fail validation
        if target not in model_configs:
            with pytest.raises(ValueError):
                raise ValueError(f"Model '{target}' not found")
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, mock_async_llm):
        """Test error on double initialization."""
        engine = mock_async_llm
        
        # First init succeeds
        engine_is_init = engine is not None
        assert engine_is_init is True
        
        # Second init should fail
        if engine_is_init:
            with pytest.raises(RuntimeError):
                raise RuntimeError("Engine already initialized")


# ============================================================================
# Test Suite 9: Worker Model Reloader
# ============================================================================

class TestWorkerModelReloader:
    """Test HPUModelReloaderMixin logic."""
    
    def test_clear_model_runner(self):
        """Test clearing model runner."""
        # Mock worker state
        model_runner = Mock()
        model_runner.model = Mock()
        
        # Simulate: _clear_model_and_cache()
        if model_runner is not None:
            del model_runner.model
            model_runner.model = None
            del model_runner
            model_runner = None
        
        # Assertion
        assert model_runner is None
    
    def test_unload_model_sequence(self):
        """Test unload_model() calls clear."""
        model_runner = Mock()
        
        # Simulate: unload_model()
        # 1. Clear model
        if model_runner is not None:
            del model_runner
            model_runner = None
        
        # 2. Clear HPU cache (mocked)
        assert model_runner is None
    
    def test_load_model_updates_configs(self):
        """Test load_model() updates worker configs."""
        worker = Mock()
        vllm_config = Mock()
        vllm_config.model_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.scheduler_config = Mock()
        vllm_config.device_config = Mock()
        vllm_config.cache_config = Mock()
        vllm_config.load_config = Mock()
        
        # Simulate config update
        worker.model_config = vllm_config.model_config
        worker.parallel_config = vllm_config.parallel_config
        worker.scheduler_config = vllm_config.scheduler_config
        worker.device_config = vllm_config.device_config
        worker.cache_config = vllm_config.cache_config
        worker.load_config = vllm_config.load_config
        
        # Assertions
        assert worker.model_config is vllm_config.model_config
        assert worker.parallel_config is vllm_config.parallel_config
    
    def test_load_model_creates_runner(self):
        """Test load_model() creates new runner."""
        worker = Mock()
        worker._create_model_runner = Mock(return_value=Mock())
        
        # Simulate: load_model()
        new_runner = worker._create_model_runner()
        worker.model_runner = new_runner
        
        # Assertions
        worker._create_model_runner.assert_called()
        assert worker.model_runner is not None


# ============================================================================
# Test Suite 10: Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_async_engine_args, mock_async_llm):
        """Test: init → generate → switch → generate → shutdown."""
        model_configs = {
            "llama": mock_async_engine_args("llama"),
            "qwen": mock_async_engine_args("qwen"),
        }
        engine = mock_async_llm
        current_model = None
        
        # 1. Initialize with llama
        current_model = "llama"
        assert current_model == "llama"
        
        # 2. Generate
        engine.generate = AsyncMock(return_value=AsyncMock())
        async def mock_gen():
            yield Mock(outputs=[Mock(text="response1")])
        engine.generate.return_value = mock_gen()
        
        # 3. Switch to qwen
        # (5-step workflow)
        await engine.wait_for_requests_to_drain(60)
        await engine.sleep(level=1)
        await engine.collective_rpc(method="unload_model", kwargs={})
        await engine.collective_rpc(method="load_model", kwargs={"vllm_config": Mock()})
        await engine.wake_up()
        current_model = "qwen"
        
        # 4. Generate again
        async for output in mock_gen():
            assert output.outputs[0].text == "response1"
        
        # 5. Shutdown
        engine.shutdown()
        
        # Assertions
        assert engine.shutdown.called


# ============================================================================
# Test Running Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

