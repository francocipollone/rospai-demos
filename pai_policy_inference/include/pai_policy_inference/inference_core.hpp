#ifndef PAI_POLICY_INFERENCE__INFERENCE_CORE_HPP_
#define PAI_POLICY_INFERENCE__INFERENCE_CORE_HPP_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "pai_policy_inference/inference_backend.hpp"
#include "pai_policy_inference/signature.hpp"
#include "pai_tensor_msgs/msg/tensor_list.hpp"

namespace pai_policy_inference
{

/// Robot-agnostic: binds observation tensors to backend inputs by name and
/// returns the "action" chunk. Not safe for concurrent infer() calls.
class InferenceCore
{
public:
  /// Throws std::runtime_error when the backend I/O does not match the signature.
  InferenceCore(std::unique_ptr<InferenceBackend> backend, PolicySignature signature);

  /// Throws std::invalid_argument for a missing or mismatched observation tensor.
  pai_tensor_msgs::msg::TensorList infer(const pai_tensor_msgs::msg::TensorList & obs);

private:
  std::unique_ptr<InferenceBackend> backend_;
  PolicySignature signature_;
  std::vector<IoSpec> inputs_;
  std::vector<int64_t> action_shape_;
};

/// Single-slot mailbox: put() replaces an untaken value (counted in dropped()).
template<typename T>
class LatestSlot
{
public:
  void put(T value)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (value_) {
        ++dropped_;
      }
      value_ = std::move(value);
    }
    cv_.notify_one();
  }

  std::optional<T> take_blocking(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] {return value_.has_value();})) {
      return std::nullopt;
    }
    std::optional<T> out = std::move(value_);
    value_.reset();
    return out;
  }

  size_t dropped() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::optional<T> value_;
  size_t dropped_ = 0;
};

/// Compares the pipeline hash a policy was exported with against the one the
/// robot computes.
class HashGate
{
public:
  HashGate(std::string expected, bool enforce)
  : expected_(std::move(expected)), enforce_(enforce) {}

  /// Returns "" when equal. On mismatch throws std::runtime_error when
  /// enforcing, otherwise returns a message for the caller to log at ERROR.
  std::string check(const std::string & received) const
  {
    if (received == expected_) {
      return "";
    }
    std::string message = "pipeline hash mismatch: policy expects '" + expected_ +
      "' but the robot computed '" + received + "'";
    if (enforce_) {
      throw std::runtime_error(message);
    }
    return message + " (enforcement disabled, continuing)";
  }

private:
  std::string expected_;
  bool enforce_;
};

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__INFERENCE_CORE_HPP_
