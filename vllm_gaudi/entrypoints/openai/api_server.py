#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM OpenAI API Server with Gaudi Extensions

This module extends the standard vLLM OpenAI API server with Gaudi-specific
functionality (model swap, sleep/wake) without requiring patches to vLLM.

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python -m vllm_gaudi.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

Or as a console script (if installed via setup.py):
  vllm-gaudi-api-server --model meta-llama/Llama-3.1-8B-Instruct
"""

import sys
from argparse import Namespace

import uvloop

import vllm.envs as envs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client,
    init_app_state,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config
from vllm.entrypoints.utils import cli_env_setup
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import decorate_logs

# Import Gaudi extension
from vllm_gaudi.extension.openai_gaudi_routes import (
    register_gaudi_openai_routes,
)

logger = init_logger("vllm_gaudi.entrypoints.openai.api_server")


async def run_server(args: Namespace, **uvicorn_kwargs) -> None:
    """Run the vLLM OpenAI API server with Gaudi extensions."""

    decorate_logs("APIServer")
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(
    listen_address,
    sock,
    args,
    client_config=None,
    **uvicorn_kwargs,
) -> None:
    """Run a single API server worker with Gaudi extensions."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)

        app = build_app(args, supported_tasks)

        # Register Gaudi-specific routes BEFORE initializing app state
        register_gaudi_openai_routes(app)
        logger.info("Registered Gaudi OpenAI extension routes")

        await init_app_state(engine_client, app.state, args, supported_tasks)

        logger.info(
            "Starting vLLM API server %d on %s",
            engine_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()


def main() -> None:
    """Main entry point for the Gaudi OpenAI API server."""

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (Gaudi)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    try:
        uvloop.run(run_server(args))
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        sys.exit(0)
    except Exception as exc:
        logger.error("Server error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

