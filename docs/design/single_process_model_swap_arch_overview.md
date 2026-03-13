# Architecture Overview

This document provides an overview of the vLLM-gaudi architecture integration to allow a single-process model swap. Original v1 vllm architecture is multi-process based, and described in `vllm/docs/design/arch_overview.md`. Only modifications are presented.

## Entrypoints

vLLM provides a number of entrypoints for interacting with the system. The standard one for online inference is `OpenAI API Server`. We increased the HTTP entrypoints compatible with OpenAI API in order to control the single process models swap feature.

### OpenAI-Compatible Gaudi API Server
The server can be launched directly via:

```bash
export VLLM_SERVER_DEV_MODE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HPU_MULTI_MODEL_CONFIG=/path/to/multi_models.yaml
python -m vllm_gaudi.entrypoints.openai.multi_model_api_server
```

That code can be found in [vllm_gaudi/entrypoints/openai/multi_model_api_server.py](../../vllm_gaudi/entrypoints/openai/multi_model_api_server.py).

## New / Modified Components

| Component | Type | Role in Model Swap |
|---|---|---|
| `vllm_gaudi.engine.MultiModelAsyncLLM` | New manager wrapper | Owns multi-model configs, serializes swap requests, drains in-flight requests, and triggers in-process reconfigure |
| `install_engine_core_patch()` | Runtime patch installer | Injects `gaudi_reconfigure_engine()` into V1 `EngineCore` at import time |
| `EngineCore.gaudi_reconfigure_engine()` | Added utility method (patched) | Performs in-place runtime rebuild: pause/sleep, worker reload, KV cache re-init, scheduler/state reconstruction, resume |
| `HPUWorker.load_model(vllm_config_bytes=...)` | Extended worker load path | Accepts cloudpickled `VllmConfig` and reloads model runner/model with new config |
| `HPUWorker._rebuild_kv_cache_config_for_current_model(...)` | New helper | Rebuilds KV cache layer mappings from current model spec to prevent stale block-table/layer mapping state |

## Control Plane Delta: Switch Flow

### Caller side (`MultiModelAsyncLLM.switch_model`)

1. Acquire `_switching_lock` (single swap at a time).
2. Validate target model and skip no-op switches.
3. Drain pending requests (`wait_for_requests_to_drain`).
4. Serialize target `VllmConfig` with `cloudpickle`.
5. Invoke EngineCore utility: `call_utility_async("gaudi_reconfigure_engine", serialized_config)`.
6. Update local model sleep-state bookkeeping and active model pointer.

### EngineCore side (`gaudi_reconfigure_engine`)

1. Deserialize new config.
2. Pause scheduler with cache reset (`pause_scheduler(mode="abort", clear_cache=True)`).
3. Sleep executor at level 1 to release device memory pressure.
4. Broadcast worker reload via collective RPC (`load_model`).
5. Recompute and initialize KV cache (`_initialize_kv_caches`, `initialize_cache`).
6. Rebuild scheduler-dependent runtime objects:
   - `StructuredOutputManager`
   - scheduler instance
   - KV connector handshake metadata
   - multimodal receiver cache
   - request block hasher and batch queue helpers
7. Resume scheduler.

## State Rebuild Delta

The model swap path rebuilds runtime state that is model-shape or scheduler-policy dependent. This avoids carrying stale state across model boundaries (for example incompatible layer bindings or stale block-table assumptions).

Rebuilt state includes:

- KV cache configuration and block counts
- scheduler instance and block sizing
- structured output manager
- multimodal receiver cache
- request block hashing setup
- queueing/execution helper state (`batch_queue`, `step_fn`, abort queue)
