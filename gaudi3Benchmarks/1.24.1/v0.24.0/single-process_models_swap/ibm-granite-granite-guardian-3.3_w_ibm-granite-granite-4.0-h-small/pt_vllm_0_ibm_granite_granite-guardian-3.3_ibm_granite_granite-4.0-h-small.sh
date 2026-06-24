#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
export VLLM_BUCKET_FILENAME=$(mktemp) && \
cat > "$VLLM_BUCKET_FILENAME" <<'BUCKETS'
(1, [256, 512, 1024, 2048, 4096, 8192], [0, 1, 2, 4, 8, 16])
(1, [512, 2048, 8192], [32, 64, 128, 192, 249])
(2, 1, [2, 4, 8, 16, 32, 64, 128, 256])
(8, 1, [8, 16, 32, 64, 128, 256, 512])
(16, 1, [16, 32, 64, 128, 256, 512, 1024])
(32, 1, [32, 64, 128, 256, 512, 1024, 2048])
BUCKETS

export VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml && \
cat > "$VLLM_HPU_MULTI_MODEL_CONFIG" << 'EOF'
default_model: ibm-granite/granite-guardian-3.3-8b
models:
  ibm-granite/granite-guardian-3.3-8b:
    model: ibm-granite/granite-guardian-3.3-8b
    tensor_parallel_size: 1
    max_num_seqs: 4
    dtype: bfloat16
    block_size: 128
    max_model_len: 131072
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.90
    override_generation_config:
      temperature: 0.0
      top_p: 1.0
      max_new_tokens: 512
  ibm-granite/granite-4.0-h-small:
    model: ibm-granite/granite-4.0-h-small
    tensor_parallel_size: 1
    max_num_seqs: 32
    dtype: bfloat16
    max_model_len: 131072
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.90
    max_num_batched_tokens: 8192
    enable_chunked_prefill: True
    tool_call_parser: hermes
    enable_auto_tool_choice: True
    override_generation_config:
      temperature: 0.0
EOF

VLLM_BUCKETING_FROM_FILE="$VLLM_BUCKET_FILENAME" \
VLLM_CONTIGUOUS_PA=fasle \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml \
python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
    --port 8090 \
    --disable-log-stats \
    --trust-remote-code	

# =========================
# BENCHMARK COMMAND - Model 1
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-guardian-3.3-8b \
    --dataset-name custom \
    --dataset-path /tmp/aegis-benchmark-granite-guardian-3.3-8b.jsonl \
    --base-url http://localhost:8090 \
    --num-prompts 40 \
    --max-concurrency 4 \
    --request-rate inf \
    --port 8090 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --temperature 0 \
    --trust-remote-code

# =========================
# ONLINE SWAP COMMAND 
# =========================
curl -s http://localhost:8090/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-4.0-h-small",
    "drain_timeout": 60
  }' | jq

# =========================
# BENCHMARK COMMAND - Model 2
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-4.0-h-small \
    --dataset-name random \
    --num-prompts 320 \
    --max-concurrency 32 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8090 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --ignore-eos \
    --trust-remote-code
	
# =========================
# TOOL VALIDATION CURL ONLY - Model 2
# =========================

cat > /tmp/tool_probe_8090.json <<'JSON'
{
  "model": "ibm-granite/granite-4.0-h-small",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2? Use the tool and return the final answer."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
          },
          "required": ["a", "b"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 256,
  "logprobs": true
}
JSON

curl -sS http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8090.json