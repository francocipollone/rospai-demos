#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pai_policy_inference/tensor_view.hpp"
#include "pai_tensor_msgs/msg/tensor.hpp"

using pai_policy_inference::DType;
using pai_policy_inference::element_count;
using pai_policy_inference::itemsize;
using pai_policy_inference::OwnedTensor;
using pai_policy_inference::to_msg;
using pai_policy_inference::view_of;
using pai_tensor_msgs::msg::Tensor;

TEST(TensorView, ItemsizeAndCount)
{
  EXPECT_EQ(itemsize(DType::FLOAT32), 4u);
  EXPECT_EQ(element_count({480, 480, 3}), 691200);
  EXPECT_EQ(element_count({}), 1);
}

TEST(TensorView, ElementCountRejectsNegativeDim)
{
  EXPECT_THROW(element_count({-1, -1}), std::invalid_argument);
  EXPECT_THROW(element_count({3, -2}), std::invalid_argument);
}

TEST(TensorView, ElementCountRejectsOverflow)
{
  EXPECT_THROW(element_count({4611686018427387905LL, 4}), std::invalid_argument);
}

TEST(TensorView, ElementCountZeroSizeShape)
{
  EXPECT_EQ(element_count({0, 5}), 0);
  EXPECT_EQ(element_count({5, 0}), 0);
}

TEST(TensorView, ViewOfRejectsNegativeShape)
{
  Tensor t;
  t.dtype = 3;
  t.shape = {-1, -1};
  t.data.resize(4);
  EXPECT_THROW(view_of(t), std::invalid_argument);
}

TEST(TensorView, DtypeFromString)
{
  EXPECT_EQ(pai_policy_inference::dtype_from_string("uint8"), DType::UINT8);
  EXPECT_EQ(pai_policy_inference::dtype_from_string("float32"), DType::FLOAT32);
  EXPECT_THROW(pai_policy_inference::dtype_from_string("bogus"), std::invalid_argument);
}

TEST(TensorView, ViewOfValidatesLength)
{
  Tensor t;
  t.name = "x";
  t.dtype = 3;
  t.shape = {6};
  t.data.resize(24);
  const auto v = view_of(t);
  EXPECT_EQ(v.nbytes, 24u);
  EXPECT_EQ(v.data, t.data.data());  // zero-copy
  t.data.resize(23);
  EXPECT_THROW(view_of(t), std::invalid_argument);
}

TEST(TensorView, ToMsgRoundTrip)
{
  OwnedTensor o{"action", DType::FLOAT32, {2, 3}, std::vector<uint8_t>(24, 7)};
  auto m = to_msg(o);
  EXPECT_EQ(m.name, "action");
  EXPECT_EQ(m.dtype, 3);
  EXPECT_EQ(m.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(m.data.size(), 24u);
  EXPECT_EQ(view_of(m).nbytes, 24u);
}
