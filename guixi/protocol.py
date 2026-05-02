"""
Binary protocol for bandwidth-efficient message encoding.

Optimized wire format for LLM inference:
- 8-byte header with magic, version, flags, length
- Compressed payload
- Support for streaming deltas
"""

import struct
import time
from enum import IntEnum
from typing import Any, Dict, List, Tuple

from .compress import CompressionEngine, CompressionMode


class MessageType(IntEnum):
    """Message type identifiers."""

    TOKEN = 0x01
    DELTA = 0x02
    CONTROL = 0x03
    ERROR = 0x04
    HEARTBEAT = 0x05
    CACHE_HIT = 0x06
    CACHE_MISS = 0x07
    STATS = 0x08


class ProtocolVersion:
    """Protocol version constants."""

    CURRENT = 1
    MIN_SUPPORTED = 1


class BinaryProtocol:
    """
    Binary protocol for LLM inference messaging.

    Wire format:
    ┌─────────────────────────────────────────┐
    │ Header (12 bytes)                       │
    ├─────────────────────────────────────────┤
    │ Magic: 0x47585849 ('GXXI') │ 4 bytes   │
    │ Version: uint8              │ 1 byte   │
    │ Flags: uint8                 │ 1 byte   │
    │ Message Type: uint8          │ 1 byte   │
    │ Payload Length: uint32       │ 4 bytes  │
    │ Reserved: uint32             │ 4 bytes  │
    ├─────────────────────────────────────────┤
    │ Payload (variable)                      │
    └─────────────────────────────────────────┘

    Flags:
    - Bit 0: Compression enabled
    - Bit 1: Encrypted
    - Bit 2: Streaming mode
    - Bit 3: Cached response
    """

    MAGIC = 0x47585849
    HEADER_SIZE = 16
    MAX_PAYLOAD_SIZE = 10 * 1024 * 1024

    FLAG_COMPRESSION = 0x01
    FLAG_ENCRYPTED = 0x02
    FLAG_STREAMING = 0x04
    FLAG_CACHED = 0x08

    def __init__(self, compression: CompressionMode = CompressionMode.LZ4):
        self.compression = CompressionEngine(compression)
        self.compression_enabled = True

    def encode_message(
        self,
        msg_type: MessageType,
        payload: bytes,
        flags: int = 0,
    ) -> bytes:
        """
        Encode a message with header.

        Args:
            msg_type: Message type identifier
            payload: Raw payload bytes
            flags: Flag bits

        Returns:
            Complete message with header
        """
        compressed = False
        if self.compression_enabled and msg_type == MessageType.TOKEN and len(payload) > 16:
            compressed_payload = self._maybe_compress(payload)
            if len(compressed_payload) < len(payload):
                payload = compressed_payload
                compressed = True

        if compressed:
            flags |= self.FLAG_COMPRESSION

        header = struct.pack(
            "<IIII",
            self.MAGIC,
            ProtocolVersion.CURRENT,
            (flags << 24) | (msg_type & 0xFF),
            len(payload),
        )

        return header + payload

    def decode_message(self, data: bytes) -> Tuple[MessageType, bytes, int]:
        """
        Decode a message from bytes.

        Args:
            data: Complete message bytes

        Returns:
            (message_type, payload, flags)
        """
        if len(data) < self.HEADER_SIZE:
            raise ValueError(f"Message too short: {len(data)} bytes")

        magic, version, type_flags, payload_len = struct.unpack("<IIII", data[: self.HEADER_SIZE])

        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")

        if version < ProtocolVersion.MIN_SUPPORTED:
            raise ValueError(f"Unsupported protocol version: {version}")

        msg_type = MessageType(type_flags & 0xFF)
        flags = (type_flags >> 24) & 0xFF

        if len(data) < self.HEADER_SIZE + payload_len:
            raise ValueError(f"Payload truncated: expected {payload_len}")

        payload = data[self.HEADER_SIZE : self.HEADER_SIZE + payload_len]

        if flags & self.FLAG_COMPRESSION and payload:
            payload = self._maybe_decompress(payload, msg_type)

        return msg_type, payload, flags

    def _maybe_compress(self, payload: bytes) -> bytes:
        """Compress payload if enabled. Returns full compressed data with header."""
        compressed = self.compression.compress(
            list(struct.unpack(f"<{len(payload) // 4}I", payload))
        )
        return compressed

    def _maybe_decompress(self, payload: bytes, msg_type: MessageType) -> bytes:
        """Decompress payload if compressed."""
        if msg_type == MessageType.TOKEN:
            try:
                tokens = self.compression.decompress(payload)
                return struct.pack(f"<{len(tokens)}I", *tokens)
            except Exception:
                pass
        return payload

    def encode_tokens(self, tokens: List[int], cached: bool = False) -> bytes:
        """Encode token stream message."""
        flags = self.FLAG_STREAMING
        if cached:
            flags |= self.FLAG_CACHED

        payload = struct.pack(f"<{len(tokens)}I", *tokens)
        return self.encode_message(MessageType.TOKEN, payload, flags)

    def decode_tokens(self, data: bytes) -> Tuple[List[int], bool]:
        """Decode token stream message."""
        msg_type, payload, flags = self.decode_message(data)
        tokens = list(struct.unpack(f"<{len(payload) // 4}I", payload))
        cached = bool(flags & self.FLAG_CACHED)
        return tokens, cached

    def encode_delta(
        self,
        additions: List[int],
        deletions: List[int],
        position: int,
    ) -> bytes:
        """Encode delta update message."""
        payload = struct.pack(
            "<III",
            position,
            len(additions),
            len(deletions),
        )
        payload += struct.pack(f"<{len(additions)}I", *additions)
        payload += struct.pack(f"<{len(deletions)}I", *deletions)
        return self.encode_message(MessageType.DELTA, payload)

    def decode_delta(self, data: bytes) -> Tuple[List[int], List[int], int]:
        """Decode delta update message."""
        _, payload, _ = self.decode_message(data)

        position, num_additions, num_deletions = struct.unpack("<III", payload[:12])
        offset = 12

        additions = list(
            struct.unpack(f"<{num_additions}I", payload[offset : offset + num_additions * 4])
        )
        offset += num_additions * 4

        deletions = list(
            struct.unpack(f"<{num_deletions}I", payload[offset : offset + num_deletions * 4])
        )

        return additions, deletions, position

    def encode_control(self, command: str, params: Dict[str, Any] = None) -> bytes:
        """Encode control message."""
        import json

        payload = json.dumps({"command": command, "params": params or {}}).encode()
        return self.encode_message(MessageType.CONTROL, payload)

    def decode_control(self, data: bytes) -> Tuple[str, Dict[str, Any]]:
        """Decode control message."""
        import json

        _, payload, _ = self.decode_message(data)
        msg = json.loads(payload)
        return msg["command"], msg.get("params", {})

    def encode_error(self, error_code: int, message: str) -> bytes:
        """Encode error message."""
        import json

        payload = json.dumps({"code": error_code, "message": message}).encode()
        return self.encode_message(MessageType.ERROR, payload)

    def decode_error(self, data: bytes) -> Tuple[int, str]:
        """Decode error message."""
        import json

        _, payload, _ = self.decode_message(data)
        msg = json.loads(payload)
        return msg["code"], msg["message"]

    def encode_stats(self, stats: Dict[str, Any]) -> bytes:
        """Encode statistics message."""
        import json

        payload = json.dumps(stats).encode()
        return self.encode_message(MessageType.STATS, payload)

    def decode_stats(self, data: bytes) -> Dict[str, Any]:
        """Decode statistics message."""
        import json

        _, payload, _ = self.decode_message(data)
        return json.loads(payload)

    def encode_heartbeat(self) -> bytes:
        """Encode heartbeat message."""
        return self.encode_message(MessageType.HEARTBEAT, struct.pack("<d", time.time()))

    @staticmethod
    def create_stats_request() -> bytes:
        """Create stats request message."""
        protocol = BinaryProtocol()
        return protocol.encode_control("get_stats")

