"""
Semantic cache for prompt/response caching with embedding similarity.

Uses embedding-based similarity search to find and reuse cached responses
for semantically similar prompts, achieving high cache hit rates even
without exact matches.
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class CachePolicy(Enum):
    """Cache behavior policy."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass
class CacheEntry:
    """A cached prompt-response pair."""

    prompt_hash: str
    response_tokens: List[int]
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def touch(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class SemanticCache:
    """
    High-performance semantic cache with embedding similarity.

    Features:
    - Exact hash matching for O(1) lookups
    - Cosine similarity search for semantic matches
    - LRU eviction with size limits
    - Async operations for non-blocking access
    """

    def __init__(
        self,
        max_size: int = 10000,
        similarity_threshold: float = 0.95,
        embedding_dim: int = 384,
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim

        self._exact_cache: Dict[str, CacheEntry] = {}
        self._semantic_index: Dict[str, List[float]] = {}
        self._access_order: List[str] = []

        self._hits = 0
        self._misses = 0
        self._total_bytes = 0

    async def get(self, prompt_hash: str) -> Optional[List[int]]:
        """
        Get cached response for a prompt hash.

        Args:
            prompt_hash: SHA256 hash of the prompt

        Returns:
            Cached tokens if found, None otherwise
        """
        if prompt_hash in self._exact_cache:
            entry = self._exact_cache[prompt_hash]
            entry.touch()
            self._hits += 1
            return entry.response_tokens

        self._misses += 1
        return None

    async def get_similar(
        self, embedding: List[float], limit: int = 5
    ) -> List[Tuple[CacheEntry, float]]:
        """
        Find cached entries with similar embeddings.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of (CacheEntry, similarity_score) tuples
        """
        if not NUMPY_AVAILABLE or not self._semantic_index:
            return []

        query = np.array(embedding)
        results = []

        for hash_key, cached_embedding in self._semantic_index.items():
            if hash_key not in self._exact_cache:
                continue

            cached = np.array(cached_embedding)
            similarity = self._cosine_similarity(query, cached)

            if similarity >= self.similarity_threshold:
                results.append((self._exact_cache[hash_key], float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def set(self, prompt_hash: str, tokens: List[int]) -> None:
        """
        Cache a prompt-response pair.

        Args:
            prompt_hash: SHA256 hash of the prompt
            tokens: Response tokens to cache
        """
        if prompt_hash in self._exact_cache:
            entry = self._exact_cache[prompt_hash]
            self._total_bytes -= entry.size_bytes
        elif len(self._exact_cache) >= self.max_size:
            await self._evict_lru()

        entry_size = len(tokens) * 4
        entry = CacheEntry(
            prompt_hash=prompt_hash,
            response_tokens=tokens,
            size_bytes=entry_size,
        )

        self._exact_cache[prompt_hash] = entry
        self._total_bytes += entry_size

        if prompt_hash not in self._access_order:
            self._access_order.append(prompt_hash)

    async def set_with_embedding(
        self, prompt_hash: str, tokens: List[int], embedding: List[float]
    ) -> None:
        """
        Cache with embedding for semantic search.

        Args:
            prompt_hash: SHA256 hash of the prompt
            tokens: Response tokens
            embedding: Embedding vector for similarity search
        """
        await self.set(prompt_hash, tokens)
        self._semantic_index[prompt_hash] = embedding

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        oldest = self._access_order.pop(0)
        if oldest in self._exact_cache:
            entry = self._exact_cache[oldest]
            self._total_bytes -= entry.size_bytes
            del self._exact_cache[oldest]

        if oldest in self._semantic_index:
            del self._semantic_index[oldest]

    async def clear(self) -> None:
        """Clear all cached entries."""
        self._exact_cache.clear()
        self._semantic_index.clear()
        self._access_order.clear()
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._exact_cache),
            "max_size": self.max_size,
            "total_bytes": self._total_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "semantic_entries": len(self._semantic_index),
        }

    @staticmethod
    def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Generate hash for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
