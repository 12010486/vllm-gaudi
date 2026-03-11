# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Deploy a sample model  with Ray Serve LLM.

Ray Serve LLM is a scalable and production-grade model serving library built
on the Ray distributed computing framework and first-class support for the vLLM engine.

Key features:
- Automatic scaling, back-pressure, and load balancing across a Ray cluster.
- Unified multi-node multi-model deployment.
- Exposes an OpenAI-compatible HTTP API.
- Multi-LoRA support with shared base models.

Run `python3 ray_serve_sleep.py` to launch an endpoint.

Learn more in the official Ray Serve LLM documentation:
https://docs.ray.io/en/latest/serve/llm/serving-llms.html
"""

import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


def verify_hpu_cluster(min_hpu: float = 1.0) -> None:
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster = ray.cluster_resources()
    available = ray.available_resources()

    hpu_total = float(cluster.get("HPU", 0.0))
    gpu_total = float(cluster.get("GPU", 0.0))

    if hpu_total < min_hpu:
        raise RuntimeError(
            "HPU resources not found in Ray cluster. "
            f"cluster_resources={cluster}. "
            "Start Ray with HPU resources, e.g. "
            "ray start --head --resources='{\"HPU\": 1}'."
        )

    print("Ray cluster resources:", cluster)
    print("Ray available resources:", available)
    print(f"HPU total={hpu_total}, GPU total={gpu_total}")


verify_hpu_cluster(min_hpu=1.0)

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "llama",
        # Pre-downloading the model to local storage is recommended when
        # the model is large. Set model_source="/path/to/the/model".
        "model_source": "meta-llama//Llama-3.1-8B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        },
        "ray_actor_options": {
            "num_cpus": 1,
            "resources": {
                "HPU": 1,
            },
        },
    },
    # Customize engine arguments as required (for example, vLLM engine kwargs).
    engine_kwargs={
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "dtype": "auto",
        "max_num_seqs": 1,
        "max_model_len": 16384,
        "enable_prefix_caching": False,
    },
)

# Deploy the application.
llm_app = build_openai_app({"llm_configs": [llm_config]})
serve.run(llm_app)

