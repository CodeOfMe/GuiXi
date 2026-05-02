"""
Delta synchronization for efficient state updates.

Implements differential synchronization algorithms to only transmit
changed portions of state between client and server, dramatically
reducing bandwidth for stateful inference sessions.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class DeltaUpdate:
    """
    Represents a delta update between two states.

    Attributes:
        additions: New items added at position
        deletions: Indices of items removed
        modifications: (index, new_value) pairs for changed items
        position: Base position for the delta
        timestamp: When the delta was created
    """

    additions: List[int] = field(default_factory=list)
    deletions: List[int] = field(default_factory=list)
    modifications: List[Tuple[int, int]] = field(default_factory=list)
    position: int = 0
    timestamp: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        """Check if delta has no changes."""
        return (
            len(self.additions) == 0 and len(self.deletions) == 0 and len(self.modifications) == 0
        )

    def size(self) -> int:
        """Get the number of changes in this delta."""
        return len(self.additions) + len(self.deletions) + len(self.modifications)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "additions": self.additions,
            "deletions": self.deletions,
            "modifications": self.modifications,
            "position": self.position,
            "timestamp": self.timestamp,
        }


class DeltaSync:
    """
    Delta synchronization for token streams.

    Implements the TextSync algorithm for efficient synchronization
    of token sequences across network boundaries.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.client_state: List[int] = []
        self.server_state: List[int] = []
        self.revision = 0

    def compute_delta(self, old_state: List[int], new_state: List[int]) -> DeltaUpdate:
        """
        Compute delta between two states using LCS algorithm.

        Args:
            old_state: Previous state
            new_state: New state

        Returns:
            DeltaUpdate describing changes
        """
        if not old_state:
            return DeltaUpdate(
                additions=new_state.copy(),
                position=0,
            )

        lcs = self._longest_common_subsequence(old_state, new_state)

        additions = []
        deletions = []
        modifications = []

        old_idx = 0
        new_idx = 0
        lcs_idx = 0

        while old_idx < len(old_state) or new_idx < len(new_state):
            if (
                lcs_idx < len(lcs)
                and old_idx < len(old_state)
                and new_idx < len(new_state)
                and old_state[old_idx] == lcs[lcs_idx]
                and new_state[new_idx] == lcs[lcs_idx]
            ):
                old_idx += 1
                new_idx += 1
                lcs_idx += 1
            elif (
                lcs_idx < len(lcs)
                and new_idx < len(new_state)
                and new_state[new_idx] == lcs[lcs_idx]
            ):
                additions.append(new_state[new_idx])
                new_idx += 1
            elif (
                lcs_idx < len(lcs)
                and old_idx < len(old_state)
                and old_state[old_idx] != lcs[lcs_idx]
            ):
                deletions.append(old_idx)
                old_idx += 1
            elif new_idx < len(new_state):
                additions.append(new_state[new_idx])
                new_idx += 1
            elif old_idx < len(old_state):
                deletions.append(old_idx)
                old_idx += 1

        return DeltaUpdate(
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            position=min(len(old_state), len(new_state)),
        )

    def apply_delta(self, state: List[int], delta: DeltaUpdate) -> List[int]:
        """
        Apply a delta update to a state.

        Args:
            state: Current state
            delta: Delta to apply

        Returns:
            Updated state
        """
        result = state.copy()

        for idx in sorted(delta.deletions, reverse=True):
            if 0 <= idx < len(result):
                del result[idx]

        for i, value in enumerate(delta.additions):
            pos = delta.position + i
            if pos >= len(result):
                result.append(value)
            else:
                result.insert(pos, value)

        for idx, value in delta.modifications:
            if 0 <= idx < len(result):
                result[idx] = value

        return result

    def patch(self, base: List[int], delta: DeltaUpdate) -> List[int]:
        """
        Apply delta and return patched result.

        This is an alias for apply_delta for semantic clarity.
        """
        return self.apply_delta(base, delta)

    def _longest_common_subsequence(self, a: List[int], b: List[int]) -> List[int]:
        """
        Compute LCS using dynamic programming.

        Args:
            a: First sequence
            b: Second sequence

        Returns:
            Longest common subsequence
        """
        if not NUMPY_AVAILABLE:
            return self._lcs_naive(a, b)

        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return []

        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                lcs.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1, j] > dp[i, j - 1]:
                i -= 1
            else:
                j -= 1

        return list(reversed(lcs))

    def _lcs_naive(self, a: List[int], b: List[int]) -> List[int]:
        """Naive LCS implementation for environments without numpy."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return []

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                lcs.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i, j - 1]:
                i -= 1
            else:
                j -= 1

        return list(reversed(lcs))

    def sync(self, client_state: List[int]) -> Tuple[List[int], DeltaUpdate]:
        """
        Synchronize client state with server.

        Args:
            client_state: Current client state

        Returns:
            (synced_state, delta_to_send)
        """
        delta = self.compute_delta(self.server_state, client_state)
        self.server_state = client_state
        self.revision += 1
        return self.apply_delta(self.server_state, delta), delta

    def get_state_hash(self, state: List[int]) -> str:
        """Get a hash of the current state for comparison."""
        import hashlib

        state_str = ",".join(map(str, state))
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
