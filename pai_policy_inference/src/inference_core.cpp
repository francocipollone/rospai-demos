#include "pai_policy_inference/inference_core.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pai_policy_inference/tensor_view.hpp"

namespace pai_policy_inference
{
namespace
{

std::string dtype_name(DType dtype)
{
  switch (dtype) {
    case DType::UINT8: return "uint8";
    case DType::INT32: return "int32";
    case DType::INT64: return "int64";
    case DType::FLOAT32: return "float32";
    case DType::FLOAT64: return "float64";
    case DType::STRING: return "string";
  }
  return "unknown";
}

std::string describe(DType dtype, const std::vector<int64_t> & shape)
{
  std::string out = dtype_name(dtype) + "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i ? "," : "") + (shape[i] < 0 ? std::string("?") : std::to_string(shape[i]));
  }
  return out + "]";
}

/// `actual` matches `expected` when ranks are equal and every non-dynamic dim is equal.
bool shape_matches(const std::vector<int64_t> & expected, const std::vector<int64_t> & actual)
{
  if (expected.size() != actual.size()) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] >= 0 && expected[i] != actual[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

InferenceCore::InferenceCore(std::unique_ptr<InferenceBackend> backend, PolicySignature signature)
: backend_(std::move(backend)), signature_(std::move(signature))
{
  if (!backend_) {
    throw std::runtime_error("inference core: backend is null");
  }
  if (!signature_.execution) {
    throw std::runtime_error("inference core: signature has no execution section");
  }
  inputs_ = backend_->inputs();
  for (const auto & input : inputs_) {
    const auto it = signature_.observations.find(input.name);
    if (it == signature_.observations.end()) {
      throw std::runtime_error(
              "model input '" + input.name + "' is not an observation in the policy signature");
    }
    const DType sig_dtype = dtype_from_string(it->second.dtype);
    if (sig_dtype != input.dtype || !shape_matches(input.shape, it->second.shape)) {
      throw std::runtime_error(
              "model input '" + input.name + "' is " + describe(input.dtype, input.shape) +
              " but the policy signature says " + describe(sig_dtype, it->second.shape));
    }
  }

  action_shape_ = {static_cast<int64_t>(signature_.execution->horizon), action_dim(signature_)};
  const auto outputs = backend_->outputs();
  if (outputs.size() != 1 || outputs[0].name != "action") {
    throw std::runtime_error("model must have exactly one output named 'action'");
  }
  if (outputs[0].dtype != DType::FLOAT32 || !shape_matches(outputs[0].shape, action_shape_)) {
    throw std::runtime_error(
            "model output 'action' is " + describe(outputs[0].dtype, outputs[0].shape) +
            " but the policy signature requires " + describe(DType::FLOAT32, action_shape_));
  }
}

pai_tensor_msgs::msg::TensorList InferenceCore::infer(const pai_tensor_msgs::msg::TensorList & obs)
{
  std::vector<TensorView> views;
  views.reserve(inputs_.size());
  for (const auto & input : inputs_) {
    const auto it = std::find_if(
      obs.tensors.begin(), obs.tensors.end(),
      [&](const auto & t) {return t.name == input.name;});
    if (it == obs.tensors.end()) {
      throw std::invalid_argument("observation missing tensor '" + input.name + "'");
    }
    TensorView view = view_of(*it);
    const auto & spec = signature_.observations.at(input.name);
    if (view.dtype != input.dtype || !shape_matches(spec.shape, view.shape)) {
      throw std::invalid_argument(
              "tensor '" + input.name + "': expected " + describe(input.dtype, spec.shape) +
              ", got " + describe(view.dtype, view.shape));
    }
    views.push_back(std::move(view));
  }

  auto outputs = backend_->run(views);
  auto action = std::find_if(
    outputs.begin(), outputs.end(), [](const auto & t) {return t.name == "action";});
  if (action == outputs.end()) {
    throw std::runtime_error("backend did not return an 'action' tensor");
  }
  if (action->dtype != DType::FLOAT32 || action->shape != action_shape_) {
    throw std::runtime_error(
            "model output 'action' is " + describe(action->dtype, action->shape) +
            ", expected " + describe(DType::FLOAT32, action_shape_));
  }

  pai_tensor_msgs::msg::TensorList out;
  out.header = obs.header;
  out.tensors.push_back(to_msg(*action));
  return out;
}

}  // namespace pai_policy_inference
