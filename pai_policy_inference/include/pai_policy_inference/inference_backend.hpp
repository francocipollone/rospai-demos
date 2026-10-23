#ifndef PAI_POLICY_INFERENCE__INFERENCE_BACKEND_HPP_
#define PAI_POLICY_INFERENCE__INFERENCE_BACKEND_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "pai_policy_inference/tensor_view.hpp"

namespace pai_policy_inference
{

/// Name, dtype and shape of one model input or output; shape entries of -1 are dynamic.
struct IoSpec
{
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
};

/// A model runner. Implementations must not retain the input views past run().
class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<IoSpec> inputs() const = 0;
  virtual std::vector<IoSpec> outputs() const = 0;
  /// `inputs` must be in the order of inputs().
  virtual std::vector<OwnedTensor> run(const std::vector<TensorView> & inputs) = 0;
};

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__INFERENCE_BACKEND_HPP_
