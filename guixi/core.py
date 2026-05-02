"""
Core business logic for GuiXi bandwidth-efficient inference.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Callable
import hashlib

from .compress import CompressionEngine, CompressionMode
from .cache import SemanticCache, CachePolicy
from .protocol import BinaryProtocol


class InferenceStats:
    """Track inference statistics for bandwidth analysis."""

    def __init__(self):
        self.total_tokens = 0
        self.compressed_bytes = 0
        self.raw_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

    @property
    def compression_ratio(self) -> float:
        if self.raw_bytes == 0:
            return 1.0
        return self.raw_bytes / max(self.compressed_bytes, 1)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def bandwidth_savings(self) -> float:
        if self.raw_bytes == 0:
            return 0.0
        return 1.0 - (self.compressed_bytes / self.raw_bytes)

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_tokens / elapsed

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "compressed_bytes": self.compressed_bytes,
            "raw_bytes": self.raw_bytes,
            "compression_ratio": self.compression_ratio,
            "cache_hit_rate": self.cache_hit_rate,
            "bandwidth_savings": self.bandwidth_savings,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class InferenceRequest:
    """Represents an inference request."""

    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    compression: CompressionMode = CompressionMode.LZ4
    cache_policy: CachePolicy = CachePolicy.READ
    stream: bool = True

    def __post_init__(self):
        if isinstance(self.compression, str):
            self.compression = CompressionMode(self.compression)
        if isinstance(self.cache_policy, str):
            self.cache_policy = CachePolicy(self.cache_policy)


@dataclass
class InferenceResponse:
    """Represents an inference response."""

    tokens: List[int]
    text: str
    stats: InferenceStats
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "text": self.text,
            "cached": self.cached,
            "stats": self.stats.to_dict(),
        }


class GuiXiServer:
    """Server for bandwidth-efficient inference."""

    def __init__(
        self,
        model_handler: Optional[Callable] = None,
        compression: CompressionMode = CompressionMode.LZ4,
        cache_size: int = 10000,
    ):
        self.compression_engine = CompressionEngine(compression)
        self.cache = SemanticCache(max_size=cache_size)
        self.model_handler = model_handler or self._default_model_handler
        self.protocol = BinaryProtocol()
        self.stats = InferenceStats()
        self._running = False

    def _default_model_handler(self, prompt: str, max_tokens: int) -> List[int]:
        """Default model handler that simulates token generation."""
        import random

        tokens = [random.randint(1, 50000) for _ in range(max_tokens)]
        return tokens

    async def handle_request(self, request: InferenceRequest) -> InferenceResponse:
        """Handle an inference request with bandwidth optimization."""
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()

        if request.cache_policy in (CachePolicy.READ, CachePolicy.READ_WRITE):
            cached_tokens = await self.cache.get(prompt_hash)
            if cached_tokens is not None:
                self.stats.cache_hits += 1
                return InferenceResponse(
                    tokens=cached_tokens,
                    text=f"[Cached] {len(cached_tokens)} tokens",
                    stats=self.stats,
                    cached=True,
                )

        self.stats.cache_misses += 1
        tokens = self.model_handler(request.prompt, request.max_tokens)

        raw_bytes = len(tokens) * 4
        compressed_data = self.compression_engine.compress(tokens)
        compressed_bytes = len(compressed_data)

        self.stats.total_tokens += len(tokens)
        self.stats.raw_bytes += raw_bytes
        self.stats.compressed_bytes += compressed_bytes

        if request.cache_policy in (CachePolicy.WRITE, CachePolicy.READ_WRITE):
            await self.cache.set(prompt_hash, tokens)

        decompressed = self.compression_engine.decompress(compressed_data)
        assert decompressed == tokens, "Compression mismatch"

        return InferenceResponse(
            tokens=tokens,
            text=f"[Decompressed] {len(tokens)} tokens",
            stats=self.stats,
            cached=False,
        )

    async def stream_tokens(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        """Stream tokens as they are generated."""
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()

        cached_tokens = None
        if request.cache_policy in (CachePolicy.READ, CachePolicy.READ_WRITE):
            cached_tokens = await self.cache.get(prompt_hash)

        if cached_tokens:
            self.stats.cache_hits += 1
            for i, token in enumerate(cached_tokens):
                yield InferenceResponse(
                    tokens=[token],
                    text=f"Token {i}",
                    stats=self.stats,
                    cached=True,
                )
                await asyncio.sleep(0.01)
            return

        self.stats.cache_misses += 1
        tokens = self.model_handler(request.prompt, request.max_tokens)

        for i, token in enumerate(tokens):
            self.stats.total_tokens += 1

            compressed_data = self.compression_engine.compress([token])
            self.stats.compressed_bytes += len(compressed_data)
            self.stats.raw_bytes += 4

            yield InferenceResponse(
                tokens=[token],
                text=f"Token {i}",
                stats=self.stats,
                cached=False,
            )
            await asyncio.sleep(0.01)

        if request.cache_policy in (CachePolicy.WRITE, CachePolicy.READ_WRITE):
            await self.cache.set(prompt_hash, tokens)

    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the server."""
        self._running = True
        self._host = host
        self._port = port

    async def stop(self):
        """Stop the server."""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.to_dict()


class GuiXiClient:
    """Client for bandwidth-efficient inference."""

    def __init__(
        self,
        server_url: str = "ws://localhost:8080",
        compression: CompressionMode = CompressionMode.LZ4,
    ):
        self.server_url = server_url
        self.compression_engine = CompressionEngine(compression)
        self.cache = SemanticCache(max_size=1000)
        self.protocol = BinaryProtocol()
        self._connected = False

    async def connect(self):
        """Connect to the server."""
        self._connected = True

    async def disconnect(self):
        """Disconnect from the server."""
        self._connected = False

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference."""
        if not self._connected:
            await self.connect()

        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()

        if request.cache_policy in (CachePolicy.READ, CachePolicy.READ_WRITE):
            cached = await self.cache.get(prompt_hash)
            if cached is not None:
                return InferenceResponse(
                    tokens=cached,
                    text="[Local cache]",
                    stats=InferenceStats(),
                    cached=True,
                )

        await asyncio.sleep(0.1)
        tokens = list(range(request.max_tokens))

        if request.cache_policy in (CachePolicy.WRITE, CachePolicy.READ_WRITE):
            await self.cache.set(prompt_hash, tokens)

        return InferenceResponse(
            tokens=tokens,
            text=f"{len(tokens)} tokens",
            stats=InferenceStats(),
            cached=False,
        )

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[int]:
        """Stream tokens from inference."""
        request = InferenceRequest(prompt=prompt, **kwargs)

        if not self._connected:
            await self.connect()

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cached = await self.cache.get(prompt_hash)

        if cached:
            for token in cached:
                yield token
            return

        for i in range(request.max_tokens):
            await asyncio.sleep(0.01)
            yield i

        await self.cache.set(prompt_hash, list(range(request.max_tokens)))

    async def batch_infer(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Perform batched inference."""
        if not self._connected:
            await self.connect()

        results = []
        for req in requests:
            result = await self.infer(req)
            results.append(result)

        return results


class GuiXi:
    """Main GuiXi orchestrator for unified usage."""

    def __init__(
        self,
        mode: str = "client",
        compression: CompressionMode = CompressionMode.LZ4,
        cache_size: int = 10000,
    ):
        self.mode = mode
        if mode == "server":
            self.server = GuiXiServer(compression=compression, cache_size=cache_size)
        else:
            self.client = GuiXiClient(compression=compression)

    @property
    def stats(self) -> Dict[str, Any]:
        if self.mode == "server":
            return self.server.get_stats()
        return {}
