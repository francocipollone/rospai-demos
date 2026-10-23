"""Rosetta ``PolicyRunner`` bridging Rosetta frames to the Policy I/O ``TensorList`` topics (phase-1)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosetta.policies.protocols import RunnerFeedback, RunnerResult

from pai_policy_io.chunk_follower import ChunkFollower
from pai_policy_io.contract_signature import contract_pipeline_hash
from pai_policy_io.signature import signature_to_dict
from pai_policy_io.tensor_msgs import frame_to_tensorlist, tensorlist_to_frame
from pai_tensor_msgs.msg import PipelineInfo, TensorList

_WARN_THROTTLE_S = 5.0
_PARAMETERS = (
    ("observation_topic", "/policy/observation", "Topic the sampled observation TensorList is published on"),
    ("action_chunk_topic", "/policy/action_chunk", "Topic the action-chunk TensorList is received on"),
    ("pipeline_info_topic", "/policy/pipeline_info", "Topic the PipelineInfo is published on (latched)"),
    ("robot", "so_arm101", "Robot name recorded in the policy signature"),
)


def _stamp_ns(header) -> int:
    return header.stamp.sec * 10**9 + header.stamp.nanosec


class TensorListBridgeRunner:
    """Publish each sampled Rosetta frame as a ``TensorList`` and replay incoming action chunks."""

    def __init__(self):
        """Create an unconfigured runner."""
        self._node = None
        self._sig = None
        self._follower: ChunkFollower | None = None
        self._frames = None
        self._obs_pub = None
        self._casts: dict[str, np.dtype] = {}
        self._published = 0
        self.observation_topic = ""
        self.action_chunk_topic = ""
        self.pipeline_info_topic = ""
        self.robot = ""

    def setup(self, node, contract) -> None:
        """Declare parameters, derive the signature from the contract and publish the pipeline info."""
        self._node = node
        for name, default, description in _PARAMETERS:
            if not node.has_parameter(name):
                node.declare_parameter(name, default, ParameterDescriptor(description=description, read_only=True))
            setattr(self, name, node.get_parameter(name).value)
        contract_path = Path(node.get_parameter("contract_path").value)
        self._sig, pipeline_hash = contract_pipeline_hash(contract_path, self.robot)
        self._casts = {
            key: np.dtype(spec.dtype) for key, spec in self._sig.observations.items() if spec.dtype != "uint8"
        }
        self._follower = ChunkFollower(len(self._sig.action.joints), 10**9 // self._sig.fps)
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._obs_pub = node.create_publisher(TensorList, self.observation_topic, reliable)
        info_pub = node.create_publisher(PipelineInfo, self.pipeline_info_topic, latched)
        info = PipelineInfo()
        info.header.stamp = node.get_clock().now().to_msg()
        info.signature_yaml = yaml.safe_dump(signature_to_dict(self._sig))
        info.pipeline_hash = pipeline_hash
        info_pub.publish(info)

    def run(self, frames, *, task: str, stop_event: threading.Event) -> RunnerResult:
        """Publish observations and actions from timer and subscription callbacks until stopped."""
        node = self._node
        self._frames = frames
        self._follower.reset()
        self._published = 0
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        sub = node.create_subscription(
            TensorList,
            self.action_chunk_topic,
            self._on_chunk,
            reliable,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        timer = node.create_timer(1.0 / self._sig.fps, self._tick, callback_group=MutuallyExclusiveCallbackGroup())
        try:
            stop_event.wait()
        finally:
            node.destroy_timer(timer)
            node.destroy_subscription(sub)
        return RunnerResult(True, "Stopped")

    def _tick(self) -> None:
        frames = self._frames
        if frames is None or not frames.warmed_up:
            return
        now = self._node.get_clock().now()
        frame = {k: v for k, v in frames.sample_frame().items() if k in self._sig.observations}
        self._obs_pub.publish(frame_to_tensorlist(frame, now.to_msg(), self._casts))
        action = self._follower.action_at(now.nanoseconds)
        if action is not None:
            frames.publish_frame({"action": action.astype(np.float64)})
            self._published += 1

    def _on_chunk(self, msg: TensorList) -> None:
        try:
            arrays = tensorlist_to_frame(msg)
        except ValueError as err:
            self._warn(f"dropping malformed action chunk: {err}")
            return
        if "action" not in arrays:
            self._warn("dropping action chunk without an 'action' tensor")
            return
        if not self._follower.push(_stamp_ns(msg.header), arrays["action"]):
            self._warn("dropping invalid or stale action chunk")

    def _warn(self, text: str) -> None:
        self._node.get_logger().warning(text, throttle_duration_sec=_WARN_THROTTLE_S)

    def feedback(self) -> RunnerFeedback:
        """Return the remaining chunk length and the number of actions published."""
        if self._follower is None or self._node is None:
            return RunnerFeedback()
        now_ns = self._node.get_clock().now().nanoseconds
        return RunnerFeedback(queue_depth=self._follower.remaining(now_ns), published_actions=self._published)

    def request_stop(self) -> None:
        """No-op: ``run`` only waits on ``stop_event``."""

    def teardown(self) -> None:
        """Nothing to release."""
