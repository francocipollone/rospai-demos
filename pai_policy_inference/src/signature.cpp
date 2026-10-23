#include "pai_policy_inference/signature.hpp"

#include <stdexcept>

#include "yaml-cpp/yaml.h"

namespace pai_policy_inference
{

namespace
{

YAML::Node require(const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::runtime_error(
            path.empty() ? "missing key: " + key : "missing key: " + path + "." + key);
  }
  return node;
}

template<typename T>
T require_as(const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = require(parent, key, path);
  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    throw std::runtime_error(
            "invalid value for key: " + (path.empty() ? key : path + "." + key));
  }
}

TensorSpec parse_observation(const YAML::Node & node, const std::string & path)
{
  TensorSpec spec;
  spec.dtype = require_as<std::string>(node, "dtype", path);
  if (node["shape"]) {
    spec.shape = require_as<std::vector<int64_t>>(node, "shape", path);
  } else if (node["joints"]) {
    const auto joints = require_as<std::vector<std::string>>(node, "joints", path);
    spec.shape = {static_cast<int64_t>(joints.size())};
  } else {
    throw std::runtime_error("missing key: " + path + ".shape (or joints)");
  }
  return spec;
}

}  // namespace

PolicySignature load_signature(const std::filesystem::path & yaml_path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path.string());
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("cannot load " + yaml_path.string() + ": " + e.what());
  }
  if (!root.IsMap()) {
    throw std::runtime_error(yaml_path.string() + ": top level is not a mapping");
  }

  PolicySignature sig;
  sig.signature_version = require_as<int>(root, "signature_version", "");
  sig.robot = require_as<std::string>(root, "robot", "");
  sig.fps = require_as<int>(root, "fps", "");

  const YAML::Node observations = require(root, "observations", "");
  if (!observations.IsMap()) {
    throw std::runtime_error("invalid value for key: observations");
  }
  for (const auto & entry : observations) {
    const auto name = entry.first.as<std::string>();
    sig.observations[name] = parse_observation(entry.second, "observations." + name);
  }

  const YAML::Node action = require(root, "action", "");
  sig.action.dtype = require_as<std::string>(action, "dtype", "action");
  sig.action.joints = require_as<std::vector<std::string>>(action, "joints", "action");
  sig.action.kinds = require_as<std::vector<std::string>>(action, "kinds", "action");

  if (!root["execution"]) {
    throw std::runtime_error("execution section missing: not an exported model");
  }
  const YAML::Node exec = root["execution"];
  ExecutionSpec spec;
  spec.horizon = require_as<int>(exec, "horizon", "execution");
  spec.execute_steps = require_as<int>(exec, "execute_steps", "execution");
  spec.strategy = require_as<std::string>(exec, "strategy", "execution");
  spec.max_staleness_s = require_as<double>(exec, "max_staleness_s", "execution");
  sig.execution = spec;

  if (root["pipeline_hash"]) {
    sig.pipeline_hash = require_as<std::string>(root, "pipeline_hash", "");
  }
  return sig;
}

int64_t action_dim(const PolicySignature & signature)
{
  return static_cast<int64_t>(signature.action.joints.size());
}

}  // namespace pai_policy_inference
