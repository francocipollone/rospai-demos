#ifndef PAI_POLICY_INFERENCE__SIGNATURE_HPP_
#define PAI_POLICY_INFERENCE__SIGNATURE_HPP_

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace pai_policy_inference
{

struct TensorSpec
{
  std::string dtype;
  std::vector<int64_t> shape;  // joint vectors: {joints.size()}
};

struct ActionSpec
{
  std::string dtype;
  std::vector<std::string> joints;
  std::vector<std::string> kinds;
};

struct ExecutionSpec
{
  int horizon;
  int execute_steps;
  std::string strategy;
  double max_staleness_s;
};

struct PolicySignature
{
  int signature_version;
  std::string robot;
  int fps;
  std::map<std::string, TensorSpec> observations;
  ActionSpec action;
  std::optional<ExecutionSpec> execution;
  std::string pipeline_hash;
};

/// Load a policy_signature.yaml. Throws std::runtime_error naming the
/// missing or invalid key (including a missing `execution` section).
PolicySignature load_signature(const std::filesystem::path & yaml_path);

/// Number of action dimensions (action.joints.size()).
int64_t action_dim(const PolicySignature & signature);

}  // namespace pai_policy_inference

#endif  // PAI_POLICY_INFERENCE__SIGNATURE_HPP_
