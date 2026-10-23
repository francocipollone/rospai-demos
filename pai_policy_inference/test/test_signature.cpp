#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pai_policy_inference/signature.hpp"

using pai_policy_inference::action_dim;
using pai_policy_inference::load_signature;

TEST(Signature, LoadsExample)
{
  auto sig = load_signature(FIXTURE_DIR "/policy_signature.yaml");
  EXPECT_EQ(sig.fps, 50);
  EXPECT_EQ(
    sig.observations.at("observation.images.wrist").shape,
    (std::vector<int64_t>{480, 480, 3}));
  EXPECT_EQ(sig.observations.at("observation.state").shape, (std::vector<int64_t>{6}));
  EXPECT_EQ(action_dim(sig), 6);
  ASSERT_TRUE(sig.execution);
  EXPECT_EQ(sig.execution->horizon, 100);
  EXPECT_EQ(sig.execution->strategy, "blend");
  EXPECT_EQ(sig.pipeline_hash, "sha256:0000");
}

TEST(Signature, MissingExecutionThrows)
{
  std::ifstream in(FIXTURE_DIR "/policy_signature.yaml");
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  const auto pos = text.find("execution:");
  ASSERT_NE(pos, std::string::npos);
  const auto end = text.find("pipeline_hash:");
  const std::string stripped = text.substr(0, pos) + text.substr(end);

  const auto tmp = std::filesystem::temp_directory_path() / "pai_sig_no_execution.yaml";
  {
    std::ofstream out(tmp);
    out << stripped;
  }
  try {
    load_signature(tmp);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("execution"), std::string::npos);
  }
  std::filesystem::remove(tmp);
}
