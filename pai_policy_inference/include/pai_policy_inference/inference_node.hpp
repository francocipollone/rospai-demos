#ifndef PAI_POLICY_INFERENCE__INFERENCE_NODE_HPP_
#define PAI_POLICY_INFERENCE__INFERENCE_NODE_HPP_

#include <atomic>
#include <memory>
#include <thread>

#include "pai_policy_inference/inference_core.hpp"
#include "pai_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pai_policy_inference
{

/// Thin ROS plumbing around InferenceCore: waits for the robot's PipelineInfo,
/// gates on the pipeline hash, then runs the latest observation through the
/// model on a worker thread and publishes the action chunk.
class InferenceNode : public rclcpp::Node
{
public:
  using TensorList = pai_tensor_msgs::msg::TensorList;

  /// Throws std::runtime_error on any startup failure.
  explicit InferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~InferenceNode() override;

private:
  void worker_loop();

  std::unique_ptr<InferenceCore> core_;
  LatestSlot<TensorList::ConstSharedPtr> slot_;
  rclcpp::Subscription<TensorList>::SharedPtr observation_sub_;
  rclcpp::Publisher<TensorList>::SharedPtr action_pub_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__INFERENCE_NODE_HPP_
