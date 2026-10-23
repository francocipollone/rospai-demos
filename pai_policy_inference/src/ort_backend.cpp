#include "pai_policy_inference/ort_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace pai_policy_inference
{
namespace
{

ONNXTensorElementDataType to_onnx(DType dtype)
{
  switch (dtype) {
    case DType::UINT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DType::INT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::FLOAT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DType::STRING: break;
  }
  throw std::invalid_argument("string tensors are not supported as ONNX Runtime inputs");
}

DType from_onnx(ONNXTensorElementDataType type, const std::string & name)
{
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DType::UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DType::INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DType::INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DType::FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return DType::FLOAT64;
    default: break;
  }
  throw std::runtime_error(
          "model tensor '" + name + "' has unsupported element type " +
          std::to_string(static_cast<int>(type)));
}

void require_provider(const std::string & provider)
{
  const char * ort_name = nullptr;
  if (provider == "cpu") {
    ort_name = "CPUExecutionProvider";
  } else if (provider == "cuda") {
    ort_name = "CUDAExecutionProvider";
  }
  bool available = false;
  if (ort_name != nullptr) {
    const auto providers = Ort::GetAvailableProviders();
    available = std::find(providers.begin(), providers.end(), ort_name) != providers.end();
  }
  if (!available) {
    throw std::runtime_error(
            "execution provider '" + provider + "' is not available in this ONNX Runtime build");
  }
}

}  // namespace

struct OrtBackend::Impl
{
  Impl(const std::filesystem::path & model, const OrtOptions & options)
  : env(ORT_LOGGING_LEVEL_WARNING, "pai_policy_inference"),
    memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    require_provider(options.provider);
    try {
      Ort::SessionOptions session_options;
      if (options.intra_op_threads > 0) {
        session_options.SetIntraOpNumThreads(options.intra_op_threads);
      }
      if (options.provider == "cuda") {
        session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
      }
      session = std::make_unique<Ort::Session>(env, model.c_str(), session_options);

      Ort::AllocatorWithDefaultOptions allocator;
      for (size_t i = 0; i < session->GetInputCount(); ++i) {
        input_specs.push_back(
          describe(
            session->GetInputNameAllocated(i, allocator).get(),
            session->GetInputTypeInfo(i)));
      }
      for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        output_specs.push_back(
          describe(
            session->GetOutputNameAllocated(i, allocator).get(),
            session->GetOutputTypeInfo(i)));
      }
    } catch (const Ort::Exception & e) {
      throw std::runtime_error(
              "failed to load model '" + model.string() + "' with provider '" +
              options.provider + "': " + e.what());
    }
  }

  static IoSpec describe(const std::string & name, const Ort::TypeInfo & type_info)
  {
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      throw std::runtime_error("model tensor '" + name + "' is not a tensor");
    }
    const auto info = type_info.GetTensorTypeAndShapeInfo();
    return IoSpec{name, from_onnx(info.GetElementType(), name), info.GetShape()};
  }

  Ort::Env env;
  Ort::MemoryInfo memory_info;
  std::unique_ptr<Ort::Session> session;
  std::vector<IoSpec> input_specs;
  std::vector<IoSpec> output_specs;
};

OrtBackend::OrtBackend(const std::filesystem::path & model, const OrtOptions & options)
: impl_(std::make_unique<Impl>(model, options)) {}

OrtBackend::~OrtBackend() = default;

std::vector<IoSpec> OrtBackend::inputs() const {return impl_->input_specs;}

std::vector<IoSpec> OrtBackend::outputs() const {return impl_->output_specs;}

std::vector<OwnedTensor> OrtBackend::run(const std::vector<TensorView> & inputs)
{
  const auto & specs = impl_->input_specs;
  if (inputs.size() != specs.size()) {
    throw std::invalid_argument(
            "expected " + std::to_string(specs.size()) + " inputs, got " +
            std::to_string(inputs.size()));
  }
  std::vector<const char *> input_names;
  std::vector<Ort::Value> values;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto & view = inputs[i];
    if (view.name != specs[i].name) {
      throw std::invalid_argument(
              "input " + std::to_string(i) + " is '" + view.name + "', expected '" +
              specs[i].name + "'");
    }
    // ORT only reads inputs; the const_cast avoids copying the image bytes.
    values.push_back(
      Ort::Value::CreateTensor(
        impl_->memory_info, const_cast<void *>(static_cast<const void *>(view.data)),
        view.nbytes, view.shape.data(), view.shape.size(), to_onnx(view.dtype)));
    input_names.push_back(specs[i].name.c_str());
  }
  std::vector<const char *> output_names;
  for (const auto & spec : impl_->output_specs) {
    output_names.push_back(spec.name.c_str());
  }

  std::vector<Ort::Value> results;
  try {
    results = impl_->session->Run(
      Ort::RunOptions{nullptr}, input_names.data(), values.data(), values.size(),
      output_names.data(), output_names.size());
  } catch (const Ort::Exception & e) {
    throw std::runtime_error(std::string("ONNX Runtime run failed: ") + e.what());
  }

  std::vector<OwnedTensor> outputs;
  for (size_t i = 0; i < results.size(); ++i) {
    const auto info = results[i].GetTensorTypeAndShapeInfo();
    OwnedTensor tensor;
    tensor.name = impl_->output_specs[i].name;
    tensor.dtype = from_onnx(info.GetElementType(), tensor.name);
    tensor.shape = info.GetShape();
    const size_t nbytes = info.GetElementCount() * itemsize(tensor.dtype);
    tensor.data.resize(nbytes);
    std::memcpy(tensor.data.data(), results[i].GetTensorRawData(), nbytes);
    outputs.push_back(std::move(tensor));
  }
  return outputs;
}

}  // namespace pai_policy_inference
