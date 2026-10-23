#ifndef PAI_POLICY_INFERENCE__ORT_BACKEND_HPP_
#define PAI_POLICY_INFERENCE__ORT_BACKEND_HPP_

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "pai_policy_inference/inference_backend.hpp"

namespace pai_policy_inference
{

struct OrtOptions
{
  std::string provider = "cpu";  // "cpu" | "cuda"
  int intra_op_threads = 0;      // 0 = ONNX Runtime default
};

/// ONNX Runtime backend. The ORT headers stay in ort_backend.cpp (pimpl).
class OrtBackend : public InferenceBackend
{
public:
  /// Throws std::runtime_error when the provider is unavailable in this ONNX
  /// Runtime build or the model cannot be loaded.
  OrtBackend(const std::filesystem::path & model, const OrtOptions & options);
  ~OrtBackend() override;

  std::vector<IoSpec> inputs() const override;
  std::vector<IoSpec> outputs() const override;
  /// Inputs are wrapped without copying; only the outputs are copied.
  std::vector<OwnedTensor> run(const std::vector<TensorView> & inputs) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__ORT_BACKEND_HPP_
