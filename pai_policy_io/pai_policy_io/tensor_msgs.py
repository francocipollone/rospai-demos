"""Conversion between numpy arrays and ``pai_tensor_msgs`` messages."""

import numpy as np
from builtin_interfaces.msg import Time

from pai_tensor_msgs.msg import Tensor, TensorList

NP_TO_DTYPE: dict[np.dtype, int] = {
    np.dtype(np.uint8): Tensor.DTYPE_UINT8,
    np.dtype(np.int32): Tensor.DTYPE_INT32,
    np.dtype(np.int64): Tensor.DTYPE_INT64,
    np.dtype(np.float32): Tensor.DTYPE_FLOAT32,
    np.dtype(np.float64): Tensor.DTYPE_FLOAT64,
}
DTYPE_TO_NP: dict[int, np.dtype] = {code: dtype for dtype, code in NP_TO_DTYPE.items()}


def array_to_tensor(name: str, array: np.ndarray) -> Tensor:
    """Pack a numpy array into a ``Tensor``; raise ``TypeError`` for unsupported dtypes."""
    array = np.ascontiguousarray(array)
    if array.dtype not in NP_TO_DTYPE:
        raise TypeError(f"unsupported dtype {array.dtype} for tensor '{name}'")
    tensor = Tensor()
    tensor.name = name
    tensor.dtype = NP_TO_DTYPE[array.dtype]
    tensor.shape = [int(n) for n in array.shape]
    tensor.data = array.tobytes()
    return tensor


def string_to_tensor(name: str, text: str) -> Tensor:
    """Pack a string into a ``DTYPE_STRING`` ``Tensor`` whose shape is the UTF-8 byte count."""
    raw = text.encode("utf-8")
    tensor = Tensor()
    tensor.name = name
    tensor.dtype = Tensor.DTYPE_STRING
    tensor.shape = [len(raw)]
    tensor.data = raw
    return tensor


def tensor_to_array(tensor: Tensor) -> np.ndarray | str:
    """Unpack a ``Tensor`` into a numpy array, or a ``str`` for ``DTYPE_STRING``."""
    raw = bytes(tensor.data)
    shape = tuple(int(n) for n in tensor.shape)
    if tensor.dtype == Tensor.DTYPE_STRING:
        if len(raw) != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"tensor '{tensor.name}': {len(raw)} data bytes do not match shape {shape}")
        return raw.decode("utf-8")
    if tensor.dtype not in DTYPE_TO_NP:
        raise ValueError(f"tensor '{tensor.name}': unknown dtype code {tensor.dtype}")
    dtype = DTYPE_TO_NP[tensor.dtype]
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"tensor '{tensor.name}': expected {expected} data bytes for shape {shape}, got {len(raw)}")
    return np.frombuffer(raw, dtype).reshape(shape).copy()


def frame_to_tensorlist(frame: dict[str, np.ndarray | str], stamp: Time, casts: dict[str, np.dtype]) -> TensorList:
    """Pack a name-to-value frame into a stamped ``TensorList``, casting keys listed in ``casts`` first."""
    msg = TensorList()
    msg.header.stamp = stamp
    for name, value in frame.items():
        if isinstance(value, str):
            msg.tensors.append(string_to_tensor(name, value))
            continue
        array = np.asarray(value)
        if name in casts:
            array = array.astype(casts[name])
        msg.tensors.append(array_to_tensor(name, array))
    return msg


def tensorlist_to_frame(msg: TensorList) -> dict[str, np.ndarray | str]:
    """Unpack a ``TensorList`` into a name-to-value frame."""
    return {tensor.name: tensor_to_array(tensor) for tensor in msg.tensors}
