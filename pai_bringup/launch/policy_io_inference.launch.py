# Copyright 2025 Yadunund Vijay.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Rosetta's policy runner (TensorList bridge) and the inference node."""
    rosetta_config = PathJoinSubstitution([FindPackageShare("pai_data_collection"), "config", "rosetta"])

    declared_arguments = [
        DeclareLaunchArgument(
            "model_dir",
            description="Directory with model.onnx and policy_signature.yaml.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (/clock) time.",
        ),
        DeclareLaunchArgument(
            "provider",
            default_value="cpu",
            description="ONNX Runtime execution provider: cpu or cuda.",
        ),
        DeclareLaunchArgument(
            "intra_op_threads",
            default_value="0",
            description="ONNX Runtime intra-op threads; 0 uses the ONNX Runtime default (all cores).",
        ),
        DeclareLaunchArgument(
            "pipeline_info_timeout_s",
            default_value="10.0",
            description="Seconds the inference node waits for PipelineInfo at start-up before exiting.",
        ),
        DeclareLaunchArgument(
            "contract_path",
            default_value=PathJoinSubstitution([rosetta_config, "so_arm101_policy_io.yaml"]),
            description="Rosetta SI contract describing the observation/action pipeline.",
        ),
        DeclareLaunchArgument(
            "enforce_pipeline_hash",
            default_value="true",
            description="Refuse to start when the pipeline hash does not match the model.",
        ),
    ]

    policy_runner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rosetta"), "launch", "policy_runner_launch.py"])
        ),
        launch_arguments={
            "params_file": PathJoinSubstitution([rosetta_config, "policy_runner_tensorlist.yaml"]),
            "contract_path": LaunchConfiguration("contract_path"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    inference_node = Node(
        package="pai_policy_inference",
        executable="inference_node",
        output="screen",
        parameters=[
            {
                "model_dir": LaunchConfiguration("model_dir"),
                "provider": LaunchConfiguration("provider"),
                "intra_op_threads": ParameterValue(LaunchConfiguration("intra_op_threads"), value_type=int),
                "pipeline_info_timeout_s": ParameterValue(
                    LaunchConfiguration("pipeline_info_timeout_s"), value_type=float
                ),
                "enforce_pipeline_hash": ParameterValue(LaunchConfiguration("enforce_pipeline_hash"), value_type=bool),
                "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
            }
        ],
    )

    return LaunchDescription([*declared_arguments, policy_runner, inference_node])
