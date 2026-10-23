#include "pai_policy_inference/tensor_view.hpp"

#include <stdexcept>

namespace pai_policy_inference
{

DType dtype_from_string(const std::string & name)
{
  if (name == "uint8") {return DType::UINT8;}
  if (name == "int32") {return DType::INT32;}
  if (name == "int64") {return DType::INT64;}
  if (name == "float32") {return DType::FLOAT32;}
  if (name == "float64") {return DType::FLOAT64;}
  if (name == "string") {return DType::STRING;}
  throw std::invalid_argument("unknown dtype: " + name);
}

size_t itemsize(DType dtype)
{
  switch (dtype) {
    case DType::UINT8: return 1;
    case DType::INT32: return 4;
    case DType::INT64: return 8;
    case DType::FLOAT32: return 4;
    case DType::FLOAT64: return 8;
    case DType::STRING: return 1;
  }
  throw std::invalid_argument("invalid dtype value");
}

int64_t element_count(const std::vector<int64_t> & shape)
{
  int64_t count = 1;
  for (const auto dim : shape) {
    count *= dim;
  }
  return count;
}

TensorView view_of(const pai_tensor_msgs::msg::Tensor & msg)
{
  if (msg.dtype > static_cast<uint8_t>(DType::STRING)) {
    throw std::invalid_argument(
            "tensor '" + msg.name + "': unknown dtype " + std::to_string(msg.dtype));
  }
  const auto dtype = static_cast<DType>(msg.dtype);
  const int64_t count = element_count(msg.shape);
  if (count < 0) {
    throw std::invalid_argument("tensor '" + msg.name + "': negative dimension in shape");
  }
  const size_t expected = static_cast<size_t>(count) * itemsize(dtype);
  if (msg.data.size() != expected) {
    throw std::invalid_argument(
            "tensor '" + msg.name + "': data has " + std::to_string(msg.data.size()) +
            " bytes, expected " + std::to_string(expected));
  }
  return TensorView{msg.name, dtype, msg.shape, msg.data.data(), msg.data.size()};
}

pai_tensor_msgs::msg::Tensor to_msg(const OwnedTensor & tensor)
{
  pai_tensor_msgs::msg::Tensor msg;
  msg.name = tensor.name;
  msg.dtype = static_cast<uint8_t>(tensor.dtype);
  msg.shape = tensor.shape;
  msg.data.assign(tensor.data.begin(), tensor.data.end());
  return msg;
}

}  // namespace pai_policy_inference
