# Single-Process Model Swap (Online Quickstart)

This quickstart shows an end-to-end online flow for serving multiple small models sequentially on the same Gaudi card, in one process.

## When to Use

Use this mode when:

- You run one process per HPU card.
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
export VLLM_GAUDI_MULTI_MODEL=1
export VLLM_GAUDI_MULTI_MODEL_CONFIG=/path/to/multi_models.yaml

python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
  --host 0.0.0.0 \
  --port 8080
```

Compatibility aliases are also supported:

- `VLLM_HPU_MULTI_MODEL`
- `VLLM_HPU_MULTI_MODEL_CONFIG`

## Online Flow (Smoke Test)

1) List available models:

```bash
curl -s http://localhost:8080/v1/models | jq
```

2) Generate with default model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Explain Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

3) Switch model in-process:

```bash
curl -s http://localhost:8080/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "drain_timeout": 60
  }' | jq
```

4) Generate with switched model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

## CI-friendly one-command flow

```bash
cd /home/scolabre/vllm-gaudi
bash tests/full_tests/ci_e2e_discoverable_tests.sh run_single_process_model_swap_online_e2e_test
```

## Operational Guidance

- Keep `tensor_parallel_size` and key runtime options consistent across model entries unless you have validated mixed setups.
- Start with two models, then increase only after repeated swap-cycle testing.
- Track swap duration and memory behavior during N consecutive cycles before production rollout.

## Rollback

To disable this mode, unset multi-model env flags and use standard serving:

```bash
unset VLLM_GAUDI_MULTI_MODEL
unset VLLM_GAUDI_MULTI_MODEL_CONFIG
vllm serve <your-model>
```
