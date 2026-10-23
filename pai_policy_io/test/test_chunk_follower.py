"""Tests for the time-indexed action chunk follower."""

import numpy as np

from pai_policy_io.chunk_follower import ChunkFollower

DT = 20_000_000
T_OLD = 2_000
LAST = 4
MID = 2
REMAINING = 3


def test_action_at_indexes_by_time():
    """The action index follows elapsed time, clamped to the chunk."""
    f = ChunkFollower(6, DT)
    chunk = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 6))
    assert f.push(1_000_000_000, chunk)
    assert f.action_at(1_000_000_000)[0] == 0 and f.action_at(1_040_000_000)[0] == MID
    assert f.action_at(1_000_000_000 + 10 * DT)[0] == LAST
    assert f.action_at(900_000_000)[0] == 0 and f.remaining(1_040_000_000) == REMAINING


def test_push_rejects_wrong_dim_nan_and_older():
    """Invalid or older chunks are rejected and the current one is kept."""
    f = ChunkFollower(6, DT)
    good = np.zeros((3, 6), np.float32)
    assert f.push(T_OLD, good)
    assert not f.push(3_000, np.zeros((3, 5), np.float32))
    assert not f.push(3_000, np.full((3, 6), np.nan, np.float32))
    assert not f.push(1_000, np.ones((3, 6), np.float32))
    assert f.t_obs_ns == T_OLD and f.action_at(T_OLD)[0] == 0


def test_push_replaces_with_newer():
    """A newer chunk replaces the current one."""
    f = ChunkFollower(2, DT)
    assert f.push(1_000, np.zeros((2, 2), np.float32))
    assert f.push(T_OLD, np.ones((4, 2), np.float32))
    assert f.t_obs_ns == T_OLD and f.action_at(T_OLD)[0] == 1 and f.remaining(T_OLD) == LAST


def test_action_at_before_any_chunk_is_none():
    """No chunk means no action and nothing remaining."""
    f = ChunkFollower(6, DT)
    assert f.action_at(0) is None and f.remaining(0) == 0 and f.t_obs_ns is None
    f.push(1, np.zeros((1, 6), np.float32))
    f.reset()
    assert f.action_at(1) is None and f.t_obs_ns is None
