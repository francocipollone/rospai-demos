"""Tests for the LeRobot ACT to ONNX exporter (tiny randomly-initialised ACT, synthetic stats)."""

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402

from pai_policy_io.export.lerobot_act import (  # noqa: E402
    ExportEquivalenceError,
    ExportOptions,
    build_wrapper_from_policy,
    check_equivalence,
    export_act_from_policy,
    export_onnx,
)
from pai_policy_io.signature import (  # noqa: E402
    SIGNATURE_VERSION,
    ActionSpec,
    ExecutionSpec,
    PolicySignature,
    SignatureError,
    TensorSpec,
    load_signature,
)

IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
IMAGE_HW = 8
DIM = 6
CHUNK = 4
EXECUTE_STEPS = 2
TIGHT_ATOL = 1e-4
RAD2DEG = 180.0 / math.pi
JOINTS = tuple(f"joint_{i}" for i in range(DIM))
# Stats in degrees, like the legacy LeRobot dataset.
STATE_MEAN = np.array([10.0, -20.0, 30.0, 5.0, -45.0, 15.0], dtype=np.float32)
STATE_STD = np.array([20.0, 15.0, 25.0, 10.0, 30.0, 8.0], dtype=np.float32)
IMAGE_MEAN = np.array([0.4, 0.5, 0.6], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.2, 0.25, 0.3], dtype=np.float32).reshape(3, 1, 1)


def _stats():
    return {
        STATE_KEY: {"mean": STATE_MEAN, "std": STATE_STD},
        IMAGE_KEY: {"mean": IMAGE_MEAN, "std": IMAGE_STD},
        "action": {"mean": STATE_MEAN + 1.0, "std": STATE_STD * 1.5},
    }


def _signature(joints=JOINTS, image_shape=(IMAGE_HW, IMAGE_HW, 3)):
    observations = {
        IMAGE_KEY: TensorSpec("uint8", image_shape, layout="HWC_RGB"),
        STATE_KEY: TensorSpec("float32", (len(joints),), units="si", joints=joints),
    }
    action = ActionSpec("float32", "si", "absolute_joint", joints, ("continuous",) * len(joints))
    return PolicySignature(SIGNATURE_VERSION, "test_robot", 50, observations, action)


@pytest.fixture(scope="module")
def tiny_policy():
    """A tiny randomly-initialised ACT on CPU with its pre/post processors."""
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            STATE_KEY: PolicyFeature(FeatureType.STATE, (DIM,)),
            IMAGE_KEY: PolicyFeature(FeatureType.VISUAL, (3, IMAGE_HW, IMAGE_HW)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (DIM,))},
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_vae=False,
        device="cpu",
    )
    policy = ACTPolicy(config).eval()
    pre, post = make_act_pre_post_processors(config, dataset_stats=_stats())
    return policy, pre, post


def _random_frame(rng):
    state_deg = STATE_MEAN + STATE_STD * rng.standard_normal(DIM).astype(np.float32)
    return {
        IMAGE_KEY: rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8),
        STATE_KEY: (state_deg / RAD2DEG).astype(np.float32),
    }


def _wrapper_out(wrapper, frame):
    with torch.no_grad():
        return wrapper(torch.from_numpy(frame[IMAGE_KEY]), torch.from_numpy(frame[STATE_KEY])).numpy()


def test_wrapper_matches_lerobot_reference_on_random_frames(tiny_policy, tmp_path):
    """The exported graph reproduces the LeRobot pre/model/post path."""
    policy, pre, post = tiny_policy
    wrapper = build_wrapper_from_policy(policy, pre, post, _signature(), ExportOptions(legacy_units=None))
    rng = np.random.default_rng(1)
    frames = [_random_frame(rng) for _ in range(3)]
    export_onnx(wrapper, _signature(), tmp_path / "model.onnx", opset=17)
    err = check_equivalence(tmp_path / "model.onnx", policy, pre, post, frames, ExportOptions())
    assert err < TIGHT_ATOL
    assert _wrapper_out(wrapper, frames[0]).shape == (CHUNK, DIM)


def test_legacy_deg_wrapper_takes_radians_and_returns_radians(tiny_policy):
    """Legacy wrapper on radians == plain wrapper on degrees, output converted back to radians."""
    policy, pre, post = tiny_policy
    legacy = build_wrapper_from_policy(policy, pre, post, _signature(), ExportOptions(legacy_units="deg"))
    plain = build_wrapper_from_policy(policy, pre, post, _signature(), ExportOptions(legacy_units=None))
    frame_rad = _random_frame(np.random.default_rng(2))
    frame_deg = {**frame_rad, STATE_KEY: (frame_rad[STATE_KEY] * np.float32(RAD2DEG)).astype(np.float32)}
    out_legacy = _wrapper_out(legacy, frame_rad)
    out_plain_deg = _wrapper_out(plain, frame_deg)
    np.testing.assert_allclose(out_legacy, out_plain_deg / RAD2DEG, rtol=1e-5, atol=1e-6)
    # The legacy action really is in radians: the action stats are tens of degrees.
    assert np.abs(out_plain_deg).max() > np.abs(out_legacy).max() * 10


def test_export_act_writes_artifact_with_execution_and_hash(tiny_policy, tmp_path):
    """The artifact holds a checked model and the signature with execution and pipeline hash."""
    policy, pre, post = tiny_policy
    opts = ExportOptions(execute_steps=EXECUTE_STEPS, check_frames=2)
    errors = []
    out = export_act_from_policy(
        policy, pre, post, _signature(), "sha256:abc", tmp_path / "artifact", opts, on_checked=errors.append
    )
    assert len(errors) == 1 and errors[0] < TIGHT_ATOL
    sig = load_signature(out / "policy_signature.yaml")
    assert sig.execution == ExecutionSpec(CHUNK, EXECUTE_STEPS, "blend", 0.2) and sig.pipeline_hash == "sha256:abc"
    onnx.checker.check_model(str(out / "model.onnx"))
    model = onnx.load(out / "model.onnx")
    names = [i.name for i in model.graph.input]
    assert names == [IMAGE_KEY, STATE_KEY]
    assert [o.name for o in model.graph.output] == ["action"]
    assert sorted(p.name for p in out.iterdir()) == ["model.onnx", "policy_signature.yaml"]


@pytest.mark.parametrize(
    "signature",
    [_signature(joints=JOINTS[:5]), _signature(image_shape=(IMAGE_HW * 2, IMAGE_HW, 3))],
    ids=["action_dim", "image_shape"],
)
def test_export_refuses_action_dim_mismatch(tiny_policy, tmp_path, signature):
    """A signature that disagrees with the policy features is refused before anything is written."""
    policy, pre, post = tiny_policy
    with pytest.raises(SignatureError):
        export_act_from_policy(policy, pre, post, signature, "sha256:abc", tmp_path / "artifact", ExportOptions())
    assert not (tmp_path / "artifact").exists()


def test_equivalence_failure_raises(tiny_policy, tmp_path, monkeypatch):
    """An ONNX output off by one is rejected and leaves no artifact behind."""
    policy, pre, post = tiny_policy
    original_run = ort.InferenceSession.run

    def shifted_run(self, *args, **kwargs):
        return [out + 1.0 for out in original_run(self, *args, **kwargs)]

    monkeypatch.setattr(ort.InferenceSession, "run", shifted_run)
    frames = [_random_frame(np.random.default_rng(3))]
    wrapper = build_wrapper_from_policy(policy, pre, post, _signature(), ExportOptions())
    export_onnx(wrapper, _signature(), tmp_path / "model.onnx", opset=17)
    with pytest.raises(ExportEquivalenceError):
        check_equivalence(tmp_path / "model.onnx", policy, pre, post, frames, ExportOptions())
    opts = replace(ExportOptions(), check_frames=1)
    with pytest.raises(ExportEquivalenceError):
        export_act_from_policy(policy, pre, post, _signature(), "sha256:abc", tmp_path / "artifact", opts)
    assert not (tmp_path / "artifact").exists()
    assert not list(tmp_path.glob(".artifact.*")), "staging directory left behind"
