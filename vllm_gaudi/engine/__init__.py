# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model engine support for vLLM on Gaudi."""

from vllm_gaudi.engine.engine_core_patch import install_engine_core_patch
from vllm_gaudi.engine.multi_model_async_llm import MultiModelAsyncLLM

install_engine_core_patch()

__all__ = ["MultiModelAsyncLLM"]
