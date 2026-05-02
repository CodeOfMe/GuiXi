"""
OpenAI Function-Calling Tools for GuiXi.

Defines TOOLS schema and dispatch function for LLM agent integration.
"""

import json
from typing import Any

from .api import (
    api_infer,
    api_batch_infer,
    api_cache_stats,
    api_clear_cache,
    api_compress,
    api_stream_infer,
)

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "guixi_infer",
            "description": "Run LLM inference with bandwidth optimization. Supports compression, caching, and streaming for reduced bandwidth usage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Input prompt for the model",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 100,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0-2.0)",
                        "default": 0.7,
                    },
                    "compression": {
                        "type": "string",
                        "description": "Compression mode: 'none', 'lz4', 'zstd'",
                        "enum": ["none", "lz4", "zstd"],
                        "default": "lz4",
                    },
                    "cache_policy": {
                        "type": "string",
                        "description": "Cache policy: 'none', 'read', 'write', 'read_write'",
                        "enum": ["none", "read", "write", "read_write"],
                        "default": "read",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guixi_batch_infer",
            "description": "Run batched inference for multiple prompts. Reduces bandwidth through request batching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of input prompts",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens per prompt",
                        "default": 100,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature",
                        "default": 0.7,
                    },
                    "compression": {
                        "type": "string",
                        "description": "Compression mode",
                        "enum": ["none", "lz4", "zstd"],
                        "default": "lz4",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Number of prompts per batch",
                        "default": 10,
                    },
                },
                "required": ["prompts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guixi_compress",
            "description": "Compress a token sequence using specified compression mode. Returns compression ratio and savings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tokens": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of token IDs to compress",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Compression mode",
                        "enum": ["none", "lz4", "zstd"],
                        "default": "lz4",
                    },
                },
                "required": ["tokens"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guixi_cache_stats",
            "description": "Get cache statistics including hit rate, size, and bandwidth savings.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guixi_clear_cache",
            "description": "Clear all cached entries in the semantic cache.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "guixi_stream_infer",
            "description": "Stream inference results token by token. Returns a stream of tokens for real-time display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Input prompt",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 100,
                    },
                    "compression": {
                        "type": "string",
                        "description": "Compression mode",
                        "enum": ["none", "lz4", "zstd"],
                        "default": "lz4",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any] | str) -> dict:
    """
    Dispatch tool call to appropriate API function.

    Args:
        name: Tool name (e.g., 'guixi_infer')
        arguments: Tool arguments as dict or JSON string

    Returns:
        ToolResult as dict, ready for LLM consumption
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "guixi_infer":
        import asyncio

        result = asyncio.run(
            api_infer(
                prompt=arguments["prompt"],
                max_tokens=arguments.get("max_tokens", 100),
                temperature=arguments.get("temperature", 0.7),
                compression=arguments.get("compression", "lz4"),
                cache_policy=arguments.get("cache_policy", "read"),
            )
        )
        return result.to_dict()

    if name == "guixi_batch_infer":
        import asyncio

        result = asyncio.run(
            api_batch_infer(
                prompts=arguments["prompts"],
                max_tokens=arguments.get("max_tokens", 100),
                temperature=arguments.get("temperature", 0.7),
                compression=arguments.get("compression", "lz4"),
                batch_size=arguments.get("batch_size", 10),
            )
        )
        return result.to_dict()

    if name == "guixi_compress":
        import asyncio

        result = asyncio.run(
            api_compress(
                tokens=arguments["tokens"],
                mode=arguments.get("mode", "lz4"),
            )
        )
        return result.to_dict()

    if name == "guixi_cache_stats":
        import asyncio

        result = asyncio.run(api_cache_stats())
        return result.to_dict()

    if name == "guixi_clear_cache":
        import asyncio

        result = asyncio.run(api_clear_cache())
        return result.to_dict()

    if name == "guixi_stream_infer":
        import asyncio

        result = asyncio.run(
            api_stream_infer(
                prompt=arguments["prompt"],
                max_tokens=arguments.get("max_tokens", 100),
                compression=arguments.get("compression", "lz4"),
            )
        )
        return result.to_dict()

    raise ValueError(f"Unknown tool: {name}")


def get_tool(name: str) -> dict | None:
    """
    Get a specific tool schema by name.

    Args:
        name: Tool name

    Returns:
        Tool schema dict or None if not found
    """
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def list_tool_names() -> list[str]:
    """
    List all available tool names.

    Returns:
        List of tool names
    """
    return [tool["function"]["name"] for tool in TOOLS]
