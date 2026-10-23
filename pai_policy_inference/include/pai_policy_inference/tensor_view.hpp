#ifndef PAI_POLICY_INFERENCE__TENSOR_VIEW_HPP_
#define PAI_POLICY_INFERENCE__TENSOR_VIEW_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pai_tensor_msgs/msg/tensor.hpp"

namespace pai_policy_inference
{

enum class DType : uint8_t { UINT8 = 0, INT32 = 1, INT64 = 2, FLOAT32 = 3, FLOAT64 = 4, STRING = 5 };

/// "uint8" -> UINT8, ... Throws std::invalid_argument for unknown names.
DType dtype_from_string(const std::string & name);

/// Bytes per element (STRING -> 1).
size_t itemsize(DType dtype);

/// Product of the dimensions; 1 for an empty shape. Throws std::invalid_argument
/// for a negative dimension or when the product overflows int64_t.
int64_t element_count(const std::vector<int64_t> & shape);

/// Non-owning view; `data` points into the message it was made from.
struct TensorView
{
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
  const uint8_t * data;
  size_t nbytes;
};

struct OwnedTensor
{
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

/// Zero-copy view of a message. Throws std::invalid_argument when the dtype is
/// unknown or data.size() != element_count(shape) * itemsize(dtype).
TensorView view_of(const pai_tensor_msgs::msg::Tensor & msg);

pai_tensor_msgs::msg::Tensor to_msg(const OwnedTensor & tensor);

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__TENSOR_VIEW_HPP_
