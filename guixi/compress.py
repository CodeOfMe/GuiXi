"""
Compression engine for token stream compression.

Supports multiple compression modes:
- LZ4: Fast compression, low latency
- ZSTD: High compression ratio
- NONE: No compression for comparison
"""

import struct
from enum import Enum
from typing import List

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class CompressionMode(Enum):
    """Compression mode selection."""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    LZ4_HUFFMAN = "lz4_huffman"


class CompressionEngine:
    """
    Token stream compression engine.

    Provides real-time compression of token sequences using
    various algorithms optimized for different latency/compression tradeoffs.
    """

    MAGIC_HEADER = b"GXXC"

    def __init__(self, mode: CompressionMode = CompressionMode.LZ4):
        self.mode = mode
        self._validate_mode()

    def _validate_mode(self):
        """Validate compression mode availability."""
        if self.mode == CompressionMode.LZ4 and not LZ4_AVAILABLE:
            raise ImportError("lz4 not installed. Run: pip install lz4")
        if self.mode in (CompressionMode.ZSTD, CompressionMode.LZ4_HUFFMAN):
            if not ZSTD_AVAILABLE:
                raise ImportError("zstandard not installed. Run: pip install zstandard")

    def compress(self, tokens: List[int]) -> bytes:
        """
        Compress a list of tokens into bytes.

        Args:
            tokens: List of integer token IDs

        Returns:
            Compressed bytes with header
        """
        token_bytes = b"".join(struct.pack("<I", t) for t in tokens)

        if self.mode == CompressionMode.NONE:
            compressed = token_bytes
        elif self.mode == CompressionMode.LZ4:
            compressed = lz4.frame.compress(token_bytes)
        elif self.mode in (CompressionMode.ZSTD, CompressionMode.LZ4_HUFFMAN):
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(token_bytes)
        else:
            compressed = token_bytes

        header = self._build_header(len(compressed), len(tokens))
        return header + compressed

    def decompress(self, data: bytes) -> List[int]:
        """
        Decompress bytes back to token list.

        Args:
            data: Compressed bytes with header

        Returns:
            List of integer token IDs
        """
        if data[:4] != self.MAGIC_HEADER:
            raise ValueError("Invalid compression header")

        header_size = 12
        compressed_size = struct.unpack("<I", data[4:8])[0]

        compressed_data = data[header_size : header_size + compressed_size]

        if self.mode == CompressionMode.NONE:
            token_bytes = compressed_data
        elif self.mode == CompressionMode.LZ4:
            token_bytes = lz4.frame.decompress(compressed_data)
        elif self.mode in (CompressionMode.ZSTD, CompressionMode.LZ4_HUFFMAN):
            dctx = zstd.ZstdDecompressor()
            token_bytes = dctx.decompress(compressed_data)
        else:
            token_bytes = compressed_data

        tokens = []
        for i in range(0, len(token_bytes), 4):
            tokens.append(struct.unpack("<I", token_bytes[i : i + 4])[0])

        return tokens

    def _build_header(self, compressed_size: int, token_count: int) -> bytes:
        """Build compression header."""
        return self.MAGIC_HEADER + struct.pack("<II", compressed_size, token_count)

    def get_compression_info(self, original: bytes, compressed: bytes) -> dict:
        """Get detailed compression information."""
        ratio = len(original) / max(len(compressed), 1)
        savings = 1 - (len(compressed) / max(len(original), 1))
        return {
            "original_size": len(original),
            "compressed_size": len(compressed),
            "ratio": ratio,
            "savings_percent": savings * 100,
            "mode": self.mode.value,
        }


class AdaptiveCompressor:
    """
    Adaptive compression that adjusts based on content type.

    Automatically selects the best compression method based on
    the characteristics of the token stream.
    """

    def __init__(self):
        self.engine = CompressionEngine(CompressionMode.LZ4)
        self.mode = CompressionMode.LZ4

    def compress(self, tokens: List[int]) -> bytes:
        """Compress with adaptive mode selection."""
        entropy = self._calculate_entropy(tokens)

        if entropy < 2.0:
            self.mode = CompressionMode.ZSTD
        elif entropy < 4.0:
            self.mode = CompressionMode.LZ4
        else:
            self.mode = CompressionMode.NONE

        self.engine.mode = self.mode
        return self.engine.compress(tokens)

    def _calculate_entropy(self, tokens: List[int]) -> float:
        """Calculate Shannon entropy of token distribution."""
        if not tokens:
            return 0.0

        from collections import Counter

        counter = Counter(tokens)
        total = len(tokens)

        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * (p**0.5)

        return entropy * 10

    def decompress(self, data: bytes) -> List[int]:
        """Decompress."""
        return self.engine.decompress(data)
