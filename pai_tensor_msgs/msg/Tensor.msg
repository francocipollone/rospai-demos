# One named, dense, row-major tensor. See docs/design/policy-io-and-inference.md §5.
uint8 DTYPE_UINT8=0
uint8 DTYPE_INT32=1
uint8 DTYPE_INT64=2
uint8 DTYPE_FLOAT32=3
uint8 DTYPE_FLOAT64=4
uint8 DTYPE_STRING=5

string name      # e.g. observation.images.wrist, observation.state, action
uint8 dtype      # one of DTYPE_*
int64[] shape    # e.g. [480, 480, 3]; [] for a scalar
uint8[] data     # C-contiguous, native endian; len == prod(shape) * itemsize
