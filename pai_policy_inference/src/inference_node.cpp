#include "pai_policy_inference/inference_node.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pai_policy_inference/ort_backend.hpp"
#include "pai_policy_inference/signature.hpp"
#include "pai_tensor_msgs/msg/pipeline_info.hpp"
#include "rclcpp/wait_set.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pai_policy_inference
{

namespace
{

std::string format_seconds(double seconds)
{
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.1f", seconds);
  return buffer;
}

}  // namespace

InferenceNode::InferenceNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("inference_node", options)
{
  const auto model_dir = declare_parameter<std::string>("model_dir", "");
  const auto backend = declare_parameter<std::string>("backend", "onnxruntime");
  const auto provider = declare_parameter<std::string>("provider", "cpu");
  const auto intra_op_threads = declare_parameter<int>("intra_op_threads", 0);
  const auto observation_topic =
    declare_parameter<std::string>("observation_topic", "/policy/observation");
  const auto action_chunk_topic =
    declare_parameter<std::string>("action_chunk_topic", "/policy/action_chunk");
  const auto pipeline_info_topic =
    declare_parameter<std::string>("pipeline_info_topic", "/policy/pipeline_info");
  const auto enforce_pipeline_hash = declare_parameter<bool>("enforce_pipeline_hash", true);
  const auto pipeline_info_timeout_s = declare_parameter<double>("pipeline_info_timeout_s", 10.0);

  if (model_dir.empty()) {
    throw std::runtime_error("model_dir parameter is required");
  }
  if (backend != "onnxruntime") {
    throw std::runtime_error("unsupported backend '" + backend + "' (only onnxruntime)");
  }

  const std::filesystem::path dir(model_dir);
  const PolicySignature signature = load_signature(dir / "policy_signature.yaml");
  OrtOptions ort_options;
  ort_options.provider = provider;
  ort_options.intra_op_threads = intra_op_threads;
  core_ = std::make_unique<InferenceCore>(
    std::make_unique<OrtBackend>(dir / "model.onnx", ort_options), signature);

  // Wait for the robot's PipelineInfo with a wait set, so no executor is needed.
  auto info_sub = create_subscription<pai_tensor_msgs::msg::PipelineInfo>(
    pipeline_info_topic, rclcpp::QoS(1).reliable().transient_local(),
    [](pai_tensor_msgs::msg::PipelineInfo::ConstSharedPtr) {});
  rclcpp::WaitSet wait_set;
  wait_set.add_subscription(info_sub);
  const auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(pipeline_info_timeout_s));
  const auto result = wait_set.wait(timeout);
  pai_tensor_msgs::msg::PipelineInfo info;
  rclcpp::MessageInfo message_info;
  if (result.kind() != rclcpp::WaitResultKind::Ready ||
    !info_sub->take(info, message_info))
  {
    throw std::runtime_error(
            "no PipelineInfo received on " + pipeline_info_topic + " within " +
            format_seconds(pipeline_info_timeout_s) + " s");
  }
  wait_set.remove_subscription(info_sub);
  info_sub.reset();

  const HashGate gate(signature.pipeline_hash, enforce_pipeline_hash);
  const std::string mismatch = gate.check(info.pipeline_hash);
  if (!mismatch.empty()) {
    RCLCPP_ERROR(get_logger(), "%s", mismatch.c_str());
  } else {
    RCLCPP_INFO(
      get_logger(), "model %s: horizon=%d action_dim=%ld hash OK", model_dir.c_str(),
      signature.execution->horizon, static_cast<long>(action_dim(signature)));  // NOLINT
  }

  observation_sub_ = create_subscription<TensorList>(
    observation_topic, rclcpp::QoS(1).reliable(),
    [this](TensorList::ConstSharedPtr msg) {slot_.put(std::move(msg));});
  action_pub_ = create_publisher<TensorList>(action_chunk_topic, rclcpp::QoS(1).reliable());

  worker_ = std::thread([this] {worker_loop();});
}

InferenceNode::~InferenceNode()
{
  stop_ = true;
  if (worker_.joinable()) {
    worker_.join();
  }
}

void InferenceNode::worker_loop()
{
  size_t last_logged_drops = 0;
  while (!stop_ && rclcpp::ok()) {
    auto obs = slot_.take_blocking(std::chrono::milliseconds(100));
    if (!obs) {
      continue;
    }
    try {
      action_pub_->publish(core_->infer(**obs));
    } catch (const std::invalid_argument & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "skipping observation: %s", e.what());
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "inference failed: %s", e.what());
    }
    const size_t dropped = slot_.dropped();
    if (dropped / 100 > last_logged_drops / 100) {
      RCLCPP_DEBUG(get_logger(), "dropped %zu stale observations so far", dropped);
    }
    last_logged_drops = dropped;
  }
}

}  // namespace pai_policy_inference

RCLCPP_COMPONENTS_REGISTER_NODE(pai_policy_inference::InferenceNode)
