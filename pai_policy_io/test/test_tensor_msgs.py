"""Tests for numpy <-> Tensor message conversion."""

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from pai_policy_io.tensor_msgs import (
    array_to_tensor,
    frame_to_tensorlist,
    string_to_tensor,
    tensor_to_array,
    tensorlist_to_frame,
)
from pai_tensor_msgs.msg import Tensor

IMG_BYTES = 18
VEC_BYTES = 24


def test_array_round_trip_uint8_hwc():
    """A uint8 HWC image survives a round trip."""
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    t = array_to_tensor("observation.images.wrist", img)
    assert t.dtype == Tensor.DTYPE_UINT8 and list(t.shape) == [2, 3, 3] and len(t.data) == IMG_BYTES
    np.testing.assert_array_equal(tensor_to_array(t), img)


def test_float32_vector_round_trip():
    """A float32 vector packs to 4 bytes per element and round trips."""
    vec = np.arange(6, dtype=np.float32)
    t = array_to_tensor("observation.state", vec)
    assert t.dtype == Tensor.DTYPE_FLOAT32 and len(t.data) == VEC_BYTES
    out = tensor_to_array(t)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, vec)


def test_string_round_trip():
    """A string tensor decodes back to the original text."""
    t = string_to_tensor("task", "place cubes on tray")
    assert t.dtype == Tensor.DTYPE_STRING and list(t.shape) == [19]
    assert tensor_to_array(t) == "place cubes on tray"


def test_bad_length_raises():
    """Truncated data is rejected."""
    t = array_to_tensor("x", np.arange(6, dtype=np.float32))
    t.data = bytes(t.data)[:-1]
    with pytest.raises(ValueError):
        tensor_to_array(t)


def test_unsupported_dtype_raises():
    """Unsupported dtypes raise TypeError."""
    with pytest.raises(TypeError):
        array_to_tensor("x", np.zeros(2, np.float16))


def test_frame_to_tensorlist_casts_and_stamps():
    """Casts are applied and the stamp is copied; the frame round trips."""
    stamp = Time(sec=3, nanosec=4)
    frame = {"observation.state": np.zeros(6, np.float64), "task": "go"}
    msg = frame_to_tensorlist(frame, stamp, casts={"observation.state": np.float32})
    assert msg.header.stamp == stamp and msg.tensors[0].dtype == Tensor.DTYPE_FLOAT32
    back = tensorlist_to_frame(msg)
    assert back["task"] == "go" and back["observation.state"].dtype == np.float32
