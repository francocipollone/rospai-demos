"""Time-indexed following of an action chunk (no ROS dependencies)."""

import numpy as np

_CHUNK_NDIM = 2


class ChunkFollower:
    """Serve actions from the latest chunk, indexed by time elapsed since its observation."""

    def __init__(self, action_dim: int, dt_ns: int):
        """Create a follower for ``[H, action_dim]`` chunks with ``dt_ns`` nanoseconds between actions."""
        self._action_dim = action_dim
        self._dt_ns = dt_ns
        self._chunk: np.ndarray | None = None
        self._t_obs_ns: int | None = None

    def push(self, t_obs_ns: int, chunk: np.ndarray) -> bool:
        """Replace the current chunk with a newer valid one; return False and keep the old one otherwise."""
        chunk = np.asarray(chunk)
        if (
            chunk.dtype != np.float32
            or chunk.ndim != _CHUNK_NDIM
            or chunk.shape[0] < 1
            or chunk.shape[1] != self._action_dim
        ):
            return False
        if not np.isfinite(chunk).all():
            return False
        if self._t_obs_ns is not None and t_obs_ns <= self._t_obs_ns:
            return False
        self._chunk = chunk.copy()
        self._t_obs_ns = t_obs_ns
        return True

    def _index(self, now_ns: int) -> int:
        horizon = self._chunk.shape[0]
        return min(max((now_ns - self._t_obs_ns) // self._dt_ns, 0), horizon - 1)

    def action_at(self, now_ns: int) -> np.ndarray | None:
        """Return the action for ``now_ns`` (clamped to the chunk), or None before any chunk."""
        if self._chunk is None:
            return None
        return self._chunk[self._index(now_ns)]

    def remaining(self, now_ns: int) -> int:
        """Return the number of chunk actions left from ``now_ns`` on, 0 when there is no chunk."""
        if self._chunk is None:
            return 0
        return self._chunk.shape[0] - self._index(now_ns)

    def reset(self) -> None:
        """Drop the current chunk."""
        self._chunk = None
        self._t_obs_ns = None

    @property
    def t_obs_ns(self) -> int | None:
        """Observation time of the current chunk, or None."""
        return self._t_obs_ns
