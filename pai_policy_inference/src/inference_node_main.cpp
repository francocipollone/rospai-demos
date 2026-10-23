#include <exception>
#include <memory>

#include "pai_policy_inference/inference_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<pai_policy_inference::InferenceNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("inference_node"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
