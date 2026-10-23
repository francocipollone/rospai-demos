"""Tests for the Rosetta TensorList bridge runner, driven through fake node and frame objects."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from rosetta.contract.schema import load_contract

from pai_policy_io.contract_signature import contract_pipeline_hash
from pai_policy_io.rosetta_bridge import TensorListBridgeRunner
from pai_policy_io.tensor_msgs import array_to_tensor
from pai_tensor_msgs.msg import Tensor, TensorList

CONTRACT = Path(get_package_share_directory("pai_data_collection")) / "config/rosetta/so_arm101_policy_io.yaml"
NOW_NS = 5_000_000_000
OBS_TOPIC = "/policy/observation"


class FakeLogger:
    """Logger accepting rclpy-style keyword arguments."""

    def __init__(self):
        """Create a logger with no recorded warnings."""
        self.warnings = []
        self.errors = []

    def error(self, msg, **_kwargs):
        """Record an error."""
        self.errors.append(msg)

    def warning(self, msg, **_kwargs):
        """Record a warning."""
        self.warnings.append(msg)

    info = debug = warning


class FakeClock:
    """Clock frozen at ``NOW_NS``."""

    def now(self):
        """Return self so ``nanoseconds`` and ``to_msg`` are available."""
        return self

    @property
    def nanoseconds(self):
        """Current time in nanoseconds."""
        return NOW_NS

    def to_msg(self):
        """Current time as a ``builtin_interfaces/Time``."""
        return Time(sec=NOW_NS // 10**9, nanosec=NOW_NS % 10**9)


class FakePublisher:
    """Publisher appending to the node's per-topic list."""

    def __init__(self, sink):
        """Publish into ``sink``."""
        self._sink = sink

    def publish(self, msg):
        """Record ``msg``."""
        self._sink.append(msg)


class FakeNode:
    """Minimal stand-in for the hosting rclpy node."""

    def __init__(self, **params):
        """Create a node holding ``params``."""
        self.params = dict(params)
        self.published = {}
        self.qos = {}
        self.timer_callback = None
        self.sub_callback = None
        self.destroyed = []
        self.errors = []
        self.logger = FakeLogger()

    def declare_parameter(self, name, value=None, descriptor=None):
        """Record a parameter."""
        self.params[name] = value

    def has_parameter(self, name):
        """Whether ``name`` is declared."""
        return name in self.params

    def get_parameter(self, name):
        """Return an object with ``.value``."""
        return type("Param", (), {"value": self.params[name]})()

    def get_logger(self):
        """Return the fake logger."""
        return self.logger

    def get_clock(self):
        """Return the frozen clock."""
        return FakeClock()

    def create_publisher(self, _msg_type, topic, qos):
        """Return a recording publisher."""
        self.qos[topic] = qos
        pub = FakePublisher(self.published.setdefault(topic, []))
        pub.topic = topic
        return pub

    def destroy_publisher(self, pub):
        """Record destruction."""
        self.destroyed.append(pub.topic)

    def create_subscription(self, _msg_type, _topic, callback, _qos, callback_group=None):
        """Remember the subscription callback."""
        self.sub_callback = callback
        return "subscription"

    def create_timer(self, _period, callback, callback_group=None):
        """Remember the timer callback."""
        self.timer_callback = callback
        return "timer"

    def destroy_subscription(self, sub):
        """Record destruction."""
        self.destroyed.append(sub)

    def destroy_timer(self, timer):
        """Record destruction."""
        self.destroyed.append(timer)

    def fire_timer(self):
        """Invoke the timer callback."""
        self.timer_callback()

    def deliver(self, msg):
        """Invoke the subscription callback."""
        self.sub_callback(msg)


class FakeFrames:
    """FrameIO returning a fixed frame and recording published actions."""

    def __init__(self, warmed_up=True):
        """Create the frame source."""
        self.warmed_up = warmed_up
        self.published = []
        self.resets = 0
        self.fail_sampling = False

    def reset_state(self):
        """Count resets."""
        self.resets += 1

    def sample_frame(self):
        """Return two images, a float64 state and a key outside the signature."""
        if self.fail_sampling:
            raise RuntimeError("sampling failed")
        image = np.zeros((480, 480, 3), np.uint8)
        return {
            "observation.images.wrist": image,
            "observation.images.static": image,
            "observation.state": np.zeros(6, np.float64),
            "task": "ignored",
        }

    def publish_frame(self, action_frame):
        """Record the action frame."""
        self.published.append(action_frame)
        return action_frame


def make_chunk(stamp, horizon, value, dim=6):
    """Build an action-chunk TensorList."""
    msg = TensorList()
    msg.header.stamp = stamp
    msg.tensors.append(array_to_tensor("action", np.full((horizon, dim), value, np.float32)))
    return msg


@pytest.fixture
def node():
    """A fake node carrying the contract path."""
    return FakeNode(contract_path=str(CONTRACT))


def start(runner, node, frames):
    """Run the runner on a thread and wait until its timer exists."""
    stop = threading.Event()
    box = {}
    thread = threading.Thread(target=lambda: box.update(result=runner.run(frames, task="", stop_event=stop)))
    thread.start()
    deadline = time.monotonic() + 2
    while node.timer_callback is None and time.monotonic() < deadline:
        time.sleep(0.01)
    return stop, thread, box


def test_setup_publishes_pipeline_info_once(node):
    """Setup publishes one PipelineInfo carrying the contract hash and signature."""
    runner = TensorListBridgeRunner()
    runner.setup(node, load_contract(str(CONTRACT)))
    info = node.published[runner.pipeline_info_topic]
    assert len(info) == 1
    assert info[0].pipeline_hash == contract_pipeline_hash(CONTRACT, "so_arm101")[1]
    assert "observation.state" in info[0].signature_yaml


def test_tick_publishes_observation_in_signature_dtypes_and_follows_chunk(node):
    """A tick publishes a signature-typed observation; a chunk is then followed by action."""
    runner, frames = TensorListBridgeRunner(), FakeFrames()
    runner.setup(node, load_contract(str(CONTRACT)))
    stop, thread, box = start(runner, node, frames)
    node.fire_timer()
    obs = node.published[OBS_TOPIC][-1]
    assert {t.name for t in obs.tensors} == {
        "observation.images.wrist",
        "observation.images.static",
        "observation.state",
    }
    assert next(t for t in obs.tensors if t.name == "observation.state").dtype == Tensor.DTYPE_FLOAT32
    assert frames.published == []
    node.deliver(make_chunk(stamp=obs.header.stamp, horizon=4, value=0.5))
    node.fire_timer()
    np.testing.assert_allclose(frames.published[-1]["action"], 0.5)
    assert frames.published[-1]["action"].dtype == np.float64
    assert runner.feedback().published_actions == 1
    stop.set()
    thread.join(2)
    assert box["result"].success and box["result"].message == "Stopped"
    assert sorted(node.destroyed) == ["subscription", "timer"]
    assert frames.resets == 1


def test_tick_before_warmup_publishes_nothing(node):
    """Nothing is sampled or published until the frames are warmed up."""
    runner, frames = TensorListBridgeRunner(), FakeFrames(warmed_up=False)
    runner.setup(node, load_contract(str(CONTRACT)))
    stop, thread, _ = start(runner, node, frames)
    node.fire_timer()
    assert OBS_TOPIC not in node.published or node.published[OBS_TOPIC] == []
    assert frames.published == []
    stop.set()
    thread.join(2)


def test_chunk_with_wrong_dim_is_ignored(node):
    """A chunk whose action dimension mismatches the signature is rejected with a warning."""
    runner, frames = TensorListBridgeRunner(), FakeFrames()
    runner.setup(node, load_contract(str(CONTRACT)))
    stop, thread, _ = start(runner, node, frames)
    node.deliver(make_chunk(stamp=Time(sec=5), horizon=4, value=0.5, dim=3))
    assert runner.feedback().queue_depth == 0
    assert node.logger.warnings
    node.fire_timer()
    assert frames.published == []
    stop.set()
    thread.join(2)


def test_tick_exception_is_logged_not_raised(node):
    """An exception while sampling is logged and the tick is skipped."""
    runner, frames = TensorListBridgeRunner(), FakeFrames()
    runner.setup(node, load_contract(str(CONTRACT)))
    stop, thread, _ = start(runner, node, frames)
    frames.fail_sampling = True
    node.fire_timer()
    assert node.logger.errors and frames.published == []
    stop.set()
    thread.join(2)


def test_teardown_destroys_publishers_and_is_safe_before_setup(node):
    """Teardown destroys both publishers; calling it without setup or twice does nothing."""
    TensorListBridgeRunner().teardown()
    runner = TensorListBridgeRunner()
    runner.setup(node, load_contract(str(CONTRACT)))
    runner.teardown()
    runner.teardown()
    assert sorted(node.destroyed) == sorted([OBS_TOPIC, runner.pipeline_info_topic])


def test_concurrent_chunks_ticks_and_feedback_do_not_raise(node):
    """Pushing chunks while ticking and polling feedback from other threads never raises."""
    runner, frames = TensorListBridgeRunner(), FakeFrames()
    runner.setup(node, load_contract(str(CONTRACT)))
    stop, thread, _ = start(runner, node, frames)
    errors = []

    def hammer(fn):
        try:
            for _ in range(300):
                fn()
        except Exception as err:
            errors.append(err)

    chunks = iter(range(1, 10**6))
    workers = [
        threading.Thread(target=hammer, args=(f,))
        for f in (
            lambda: node.deliver(make_chunk(Time(sec=next(chunks)), 4, 0.5)),
            runner.feedback,
            node.fire_timer,
        )
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(5)
    assert errors == [] and node.logger.errors == []
    stop.set()
    thread.join(2)
