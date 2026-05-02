"""
Unified Python API for GuiXi.

Provides a clean, consistent interface for using GuiXi in Python code.
All functions return ToolResult for consistent error handling and metadata.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import (
    GuiXiClient,
    InferenceRequest,
)
from .cache import CachePolicy, SemanticCache
from .compress import CompressionMode as CompMode


@dataclass
class ToolResult:
    """
    Standardized result container for all API functions.

    All API functions return this type for consistent error handling,
    metadata propagation, and response format.
    """

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


async def api_infer(
    *,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    compression: str = "lz4",
    cache_policy: str = "read",
    server_url: str = "ws://localhost:8080",
) -> ToolResult:
    """
    Perform a single inference with bandwidth optimization.

    Args:
        prompt: Input prompt for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        compression: Compression mode ('none', 'lz4', 'zstd')
        cache_policy: Cache behavior ('none', 'read', 'write', 'read_write')
        server_url: Server WebSocket URL

    Returns:
        ToolResult with tokens, text, and stats

    Example:
        >>> result = await api_infer("What is AI?", max_tokens=50)
        >>> if result.success:
        ...     print(f"Generated: {result.data['text']}")
    """
    try:
        comp_mode = CompMode(compression.lower()) if compression != "none" else CompMode.NONE
        cache = CachePolicy(cache_policy.lower())

        client = GuiXiClient(server_url, compression=comp_mode)
        await client.connect()

        request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            compression=comp_mode,
            cache_policy=cache,
        )

        response = await client.infer(request)

        await client.disconnect()

        return ToolResult(
            success=True,
            data=response.to_dict(),
            metadata={
                "compression": compression,
                "cache_hit": response.cached,
                "latency_ms": 0,
            },
        )

    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            metadata={"prompt_length": len(prompt)},
        )


async def api_batch_infer(
    *,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
    compression: str = "lz4",
    batch_size: int = 10,
) -> ToolResult:
    """
    Perform batched inference for multiple prompts.

    Args:
        prompts: List of input prompts
        max_tokens: Maximum tokens per prompt
        temperature: Sampling temperature
        compression: Compression mode
        batch_size: Number of prompts per batch

    Returns:
        ToolResult with list of responses and aggregated stats

    Example:
        >>> results = await api_batch_infer(["What is AI?", "What is ML?"])
        >>> for r in results.data:
        ...     print(f"Response: {r['text']}")
    """
    try:
        comp_mode = CompMode(compression.lower()) if compression != "none" else CompMode.NONE

        client = GuiXiClient(compression=comp_mode)
        await client.connect()

        requests = [
            InferenceRequest(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                compression=comp_mode,
            )
            for p in prompts
        ]

        responses = []
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_results = await client.batch_infer(batch)
            responses.extend([r.to_dict() for r in batch_results])

        await client.disconnect()

        total_tokens = sum(len(r.get("tokens", [])) for r in responses)

        return ToolResult(
            success=True,
            data={"responses": responses, "count": len(responses)},
            metadata={
                "total_prompts": len(prompts),
                "total_tokens": total_tokens,
                "batch_size": batch_size,
            },
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def api_stream_infer(
    *,
    prompt: str,
    max_tokens: int = 100,
    compression: str = "lz4",
) -> ToolResult:
    """
    Stream inference results token by token.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        compression: Compression mode

    Yields:
        ToolResult for each token

    Example:
        >>> async for token_result in api_stream_infer("Hello"):
        ...     print(token_result.data, end="", flush=True)
    """
    try:
        comp_mode = CompMode(compression.lower()) if compression != "none" else CompMode.NONE

        client = GuiXiClient(compression=comp_mode)
        await client.connect()

        tokens = []
        async for token in client.stream(prompt, max_tokens=max_tokens):
            tokens.append(token)
            yield ToolResult(
                success=True,
                data=token,
                metadata={"index": len(tokens), "total": max_tokens},
            )

        await client.disconnect()

    except Exception as e:
        yield ToolResult(success=False, error=str(e))


async def api_cache_stats() -> ToolResult:
    """
    Get cache statistics.

    Returns:
        ToolResult with cache metrics

    Example:
        >>> stats = await api_cache_stats()
        >>> print(f"Hit rate: {stats.data['hit_rate']:.2%}")
    """
    try:
        cache = SemanticCache()

        stats = cache.get_stats()

        return ToolResult(
            success=True,
            data=stats,
            metadata={"timestamp": time.time()},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def api_clear_cache() -> ToolResult:
    """
    Clear all cached entries.

    Returns:
        ToolResult with operation status

    Example:
        >>> result = await api_clear_cache()
        >>> print(f"Cleared: {result.data['success']}")
    """
    try:
        cache = SemanticCache()
        await cache.clear()

        return ToolResult(
            success=True,
            data={"cleared": True, "timestamp": time.time()},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def api_compress(
    *,
    data: List[int],
    mode: str = "lz4",
) -> ToolResult:
    """
    Compress a token sequence.

    Args:
        data: List of token IDs
        mode: Compression mode ('lz4', 'zstd', 'none')

    Returns:
        ToolResult with compressed bytes and stats

    Example:
        >>> result = await api_compress([1, 2, 3, 4, 5])
        >>> print(f"Compressed {result.data['ratio']:.2f}x")
    """
    try:
        from .compress import CompressionEngine

        comp_mode = CompMode(mode.lower())
        engine = CompressionEngine(comp_mode)

        original_bytes = len(data) * 4
        compressed = engine.compress(data)

        info = engine.get_compression_info(
            b"".join(__import__("struct").pack("<I", t) for t in data),
            compressed,
        )

        return ToolResult(
            success=True,
            data={
                "compressed_size": len(compressed),
                "original_size": original_bytes,
                "ratio": info["ratio"],
                "savings": info["savings_percent"],
            },
            metadata={"mode": mode, "token_count": len(data)},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


def api_create_client(
    *,
    server_url: str = "ws://localhost:8080",
    compression: str = "lz4",
) -> ToolResult:
    """
    Create a GuiXi client instance.

    Args:
        server_url: Server WebSocket URL
        compression: Default compression mode

    Returns:
        ToolResult with client instance

    Example:
        >>> result = api_create_client("ws://server:8080")
        >>> client = result.data
    """
    try:
        comp_mode = CompMode(compression.lower()) if compression != "none" else CompMode.NONE
        client = GuiXiClient(server_url, compression=comp_mode)

        return ToolResult(
            success=True,
            data={"client": client, "url": server_url},
            metadata={"compression": compression},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


def sync_api_infer(*, prompt: str, **kwargs) -> ToolResult:
    """
    Synchronous wrapper for api_infer.

    Args:
        prompt: Input prompt for inference.

    Example:
        >>> result = sync_api_infer("What is AI?")
        >>> print(result.data)
    """
    return asyncio.run(api_infer(prompt, **kwargs))
