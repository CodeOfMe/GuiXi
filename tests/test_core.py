"""Test suite for GuiXi core functionality."""

import pytest

from guixi.compress import CompressionEngine, CompressionMode
from guixi.cache import SemanticCache
from guixi.protocol import BinaryProtocol
from guixi.sync import DeltaSync, DeltaUpdate
from guixi.core import (
    InferenceRequest,
    InferenceStats,
    GuiXiClient,
    GuiXiServer,
)


class TestCompressionEngine:
    """Test compression functionality."""

    def test_no_compression(self):
        """Test no compression mode."""
        engine = CompressionEngine(CompressionMode.NONE)
        tokens = [1, 2, 3, 4, 5]
        compressed = engine.compress(tokens)
        decompressed = engine.decompress(compressed)
        assert decompressed == tokens

    def test_lz4_compression(self):
        """Test LZ4 compression."""
        engine = CompressionEngine(CompressionMode.LZ4)
        # Use compressible data (repeated pattern)
        tokens = [i % 100 for i in range(1000)]
        compressed = engine.compress(tokens)
        decompressed = engine.decompress(compressed)
        assert decompressed == tokens
        # Compressed payload (excluding 12-byte header) should be smaller
        assert len(compressed) - 12 < len(tokens) * 4

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        engine = CompressionEngine(CompressionMode.LZ4)
        tokens = [i % 100 for i in range(1000)]
        compressed = engine.compress(tokens)
        info = engine.get_compression_info(
            b"".join(__import__("struct").pack("<I", t) for t in tokens), compressed
        )
        assert info["ratio"] > 1.0


class TestSemanticCache:
    """Test semantic cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache set and get."""
        cache = SemanticCache()
        tokens = [1, 2, 3, 4, 5]
        await cache.set("hash123", tokens)
        result = await cache.get("hash123")
        assert result == tokens

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        cache = SemanticCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = SemanticCache()
        await cache.set("hash1", [1, 2, 3])
        await cache.get("hash1")
        await cache.get("hash2")
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clearing."""
        cache = SemanticCache()
        await cache.set("hash1", [1, 2, 3])
        await cache.clear()
        stats = cache.get_stats()
        assert stats["size"] == 0


class TestBinaryProtocol:
    """Test binary protocol."""

    def test_encode_decode_tokens(self):
        """Test token encoding and decoding."""
        protocol = BinaryProtocol()
        tokens = [100, 200, 300, 400]
        encoded = protocol.encode_tokens(tokens)
        decoded, cached = protocol.decode_tokens(encoded)
        assert decoded == tokens
        assert not cached

    def test_encode_decode_delta(self):
        """Test delta encoding and decoding."""
        protocol = BinaryProtocol()
        additions = [10, 20, 30]
        deletions = [1, 2]
        encoded = protocol.encode_delta(additions, deletions, 5)
        dec_add, dec_del, pos = protocol.decode_delta(encoded)
        assert dec_add == additions
        assert dec_del == deletions
        assert pos == 5


class TestDeltaSync:
    """Test delta synchronization."""

    def test_compute_delta_additions(self):
        """Test delta computation for additions."""
        sync = DeltaSync()
        old = [1, 2, 3]
        new = [1, 2, 3, 4, 5]
        delta = sync.compute_delta(old, new)
        assert 4 in delta.additions
        assert 5 in delta.additions

    def test_compute_delta_deletions(self):
        """Test delta computation for deletions."""
        sync = DeltaSync()
        old = [1, 2, 3, 4, 5]
        new = [1, 2, 3]
        delta = sync.compute_delta(old, new)
        assert len(delta.deletions) > 0

    def test_apply_delta(self):
        """Test applying delta to state."""
        sync = DeltaSync()
        base = [1, 2, 3]
        delta = DeltaUpdate(additions=[4, 5], position=3)
        result = sync.apply_delta(base, delta)
        assert 4 in result
        assert 5 in result


class TestInferenceStats:
    """Test inference statistics."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = InferenceStats()
        assert stats.total_tokens == 0
        assert stats.compression_ratio == 1.0
        assert stats.cache_hit_rate == 0.0

    def test_stats_update(self):
        """Test stats update."""
        stats = InferenceStats()
        stats.total_tokens += 100
        stats.raw_bytes += 400
        stats.compressed_bytes += 100
        stats.cache_hits += 10
        stats.cache_misses += 5

        assert stats.compression_ratio == 4.0
        assert stats.cache_hit_rate == pytest.approx(0.667, rel=0.01)


class TestGuiXiServer:
    """Test GuiXi server."""

    @pytest.mark.asyncio
    async def test_server_creation(self):
        """Test server creation."""
        server = GuiXiServer()
        assert server is not None
        assert server.compression_engine is not None

    @pytest.mark.asyncio
    async def test_handle_request(self):
        """Test handling inference request."""
        server = GuiXiServer()
        request = InferenceRequest(prompt="Test prompt", max_tokens=10)
        response = await server.handle_request(request)
        assert response is not None
        assert len(response.tokens) == 10


class TestGuiXiClient:
    """Test GuiXi client."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Test client creation."""
        client = GuiXiClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_infer(self):
        """Test inference."""
        client = GuiXiClient()
        await client.connect()
        request = InferenceRequest(prompt="Test", max_tokens=5)
        response = await client.infer(request)
        assert response is not None
        await client.disconnect()
