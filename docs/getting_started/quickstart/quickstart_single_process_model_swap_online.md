# Single-Process Model Swap (Online Quickstart)

This quickstart shows an end-to-end online flow for serving multiple small models sequentially on the same Gaudi card, in one process.

## When to Use

Use this mode when:

- You need to switch model A → model B without server restart.
- Your workload is sequential and model sizes fit the card budget.

Do not use this mode as a replacement for multi-process or multi-node orchestration.

## Prerequisites

- vLLM and vLLM Gaudi plugin installed.
- Multi-model config file, for example:

```yaml
default_model: llama
models:
  llama:
    model: meta-llama/Llama-3.1-8B-Instruct
    tensor_parallel_size: 1
    max_model_len: 4096
  qwen:
    model: Qwen/Qwen3-0.6B
    tensor_parallel_size: 1
    max_model_len: 4096
```

## Start Server

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_SERVER_DEV_MODE=1
export VLLM_HPU_MULTI_MODEL_CONFIG=/path/to/multi_models.yaml

python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
  --host 0.0.0.0 \
  --port 8080
```

## Online Flow (Smoke Test)

1) List available models:

```bash
curl -s http://localhost:8080/v1/models | jq
```

1) Generate with default model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Explain Intel Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

1) Switch model in-process:

```bash
curl -s http://localhost:8080/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "drain_timeout": 60
  }' | jq
```

1) Generate with switched model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain Intel Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

## Rollback

To disable this mode, unset multi-model env flag and use standard serving:

```bash
unset VLLM_HPU_MULTI_MODEL_CONFIG
vllm serve <your-model>
```
