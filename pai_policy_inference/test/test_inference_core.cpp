#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pai_policy_inference/inference_core.hpp"
#include "pai_policy_inference/ort_backend.hpp"
#include "pai_policy_inference/signature.hpp"
#include "pai_tensor_msgs/msg/tensor_list.hpp"

using namespace std::chrono_literals;
using pai_policy_inference::DType;
using pai_policy_inference::HashGate;
using pai_policy_inference::InferenceCore;
using pai_policy_inference::LatestSlot;
using pai_policy_inference::load_signature;
using pai_policy_inference::OrtBackend;
using pai_policy_inference::OrtOptions;
using pai_tensor_msgs::msg::Tensor;
using pai_tensor_msgs::msg::TensorList;

namespace
{
const char * const kModel = FIXTURE_DIR "/add_one.onnx";
const char * const kSignature = FIXTURE_DIR "/add_one_signature.yaml";

Tensor state_tensor(float value)
{
  Tensor t;
  t.name = "observation.state";
  t.dtype = Tensor::DTYPE_FLOAT32;
  t.shape = {6};
  t.data.resize(6 * sizeof(float));
  for (int i = 0; i < 6; ++i) {
    std::memcpy(t.data.data() + i * sizeof(float), &value, sizeof(float));
  }
  return t;
}

Tensor image_tensor()
{
  Tensor t;
  t.name = "observation.images.wrist";
  t.dtype = Tensor::DTYPE_UINT8;
  t.shape = {4, 4, 3};
  t.data.assign(48, 0);
  return t;
}

TensorList make_obs(float state_value)
{
  TensorList obs;
  obs.header.stamp.sec = 12;
  obs.header.stamp.nanosec = 34;
  obs.tensors.push_back(state_tensor(state_value));
  obs.tensors.push_back(image_tensor());
  return obs;
}

InferenceCore make_core()
{
  return InferenceCore(std::make_unique<OrtBackend>(kModel, OrtOptions{}), load_signature(kSignature));
}
}  // namespace

TEST(OrtBackend, ReportsIo)
{
  OrtBackend b(kModel, {});
  ASSERT_EQ(b.inputs().size(), 2u);
  ASSERT_EQ(b.outputs().size(), 1u);
  EXPECT_EQ(b.outputs()[0].name, "action");
  EXPECT_EQ(b.outputs()[0].dtype, DType::FLOAT32);
  EXPECT_EQ(b.outputs()[0].shape, (std::vector<int64_t>{2, 6}));
}

TEST(OrtBackend, UnavailableProviderThrows)
{
  EXPECT_THROW(OrtBackend(kModel, {"nope", 0}), std::runtime_error);
  EXPECT_THROW(OrtBackend(kModel, {"cuda", 0}), std::runtime_error);  // CPU-only build
}

TEST(InferenceCore, RunsAndCopiesStamp)
{
  auto core = make_core();
  auto out = core.infer(make_obs(1.5f));
  EXPECT_EQ(out.header.stamp.sec, 12);
  EXPECT_EQ(out.header.stamp.nanosec, 34u);
  ASSERT_EQ(out.tensors.size(), 1u);
  EXPECT_EQ(out.tensors[0].name, "action");
  EXPECT_EQ(out.tensors[0].shape, (std::vector<int64_t>{2, 6}));
  ASSERT_EQ(out.tensors[0].data.size(), 12 * sizeof(float));
  const float * a = reinterpret_cast<const float *>(out.tensors[0].data.data());
  EXPECT_FLOAT_EQ(a[0], 2.5f);
  EXPECT_FLOAT_EQ(a[11], 2.5f);
}

TEST(InferenceCore, MissingInputThrows)
{
  auto core = make_core();
  TensorList obs;
  obs.tensors.push_back(state_tensor(1.0f));
  EXPECT_THROW(core.infer(obs), std::invalid_argument);
}

TEST(InferenceCore, WrongDtypeThrows)
{
  auto core = make_core();
  auto obs = make_obs(1.0f);
  obs.tensors[0].dtype = Tensor::DTYPE_FLOAT64;
  obs.tensors[0].data.assign(6 * sizeof(double), 0);
  EXPECT_THROW(core.infer(obs), std::invalid_argument);
}

TEST(InferenceCore, SignatureMismatchAtConstruction)
{
  auto sig = load_signature(kSignature);
  sig.observations.at("observation.state").shape = {5};
  EXPECT_THROW(
    InferenceCore(std::make_unique<OrtBackend>(kModel, OrtOptions{}), sig), std::runtime_error);
}

TEST(LatestSlot, KeepsOnlyNewest)
{
  LatestSlot<int> s;
  s.put(1);
  s.put(2);
  EXPECT_EQ(s.take_blocking(10ms), 2);
  EXPECT_EQ(s.dropped(), 1u);
  EXPECT_FALSE(s.take_blocking(10ms));
}

TEST(HashGate, EnforcesAndOverrides)
{
  EXPECT_EQ(HashGate("sha256:a", true).check("sha256:a"), "");
  EXPECT_THROW(HashGate("sha256:a", true).check("sha256:b"), std::runtime_error);
  EXPECT_NE(HashGate("sha256:a", false).check("sha256:b"), "");
}
