"""Export a LeRobot ACT policy to ONNX with normalisation and legacy unit wrapping baked in (design spec 5.3, 7.2).

The exported graph takes the signature tensors exactly (no batch dimension): uint8 HWC RGB images in signature
order, then the float32 SI state. It returns the float32 SI action chunk ``[horizon, action_dim]``. Every export
is checked against the LeRobot reference path (pre-processor, ``predict_action_chunk``, post-processor) before
the artifact is written.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from pai_policy_io.signature import ExecutionSpec, PolicySignature, SignatureError, dump_signature

_LOG = logging.getLogger(__name__)

MODEL_FILE = "model.onnx"
SIGNATURE_FILE = "policy_signature.yaml"
LEGACY_UNITS = (None, "deg")
EPS = 1e-8  # LeRobot's _NormalizationMixin.eps
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0
_SUPPORTED_MODES = (NormalizationMode.IDENTITY, NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX)


class ExportEquivalenceError(RuntimeError):
    """Raised when the exported ONNX model disagrees with the LeRobot reference path."""


@dataclass(frozen=True)
class ExportOptions:
    """Options for exporting an ACT policy."""

    legacy_units: str | None = None  # "deg": state rad->deg before the model, action deg->rad after
    execute_steps: int = 30
    strategy: str = "blend"
    max_staleness_s: float = 0.2
    opset: int = 17
    check_frames: int = 8  # equivalence frames; 0 disables (export still refused unless unsafe_skip_check)
    atol: float = 1e-3
    rtol: float = 1e-3
    dataset: str | None = None  # LeRobot repo id or local root for check frames; None -> random frames within stats
    unsafe_skip_check: bool = False  # allow check_frames == 0


def _normalize(x: torch.Tensor, mode: NormalizationMode, stats: dict[str, torch.Tensor], inverse: bool) -> torch.Tensor:
    """Apply LeRobot's (un)normalisation formulas (``_NormalizationMixin._apply_transform``)."""
    if mode == NormalizationMode.IDENTITY or not stats:
        return x
    if mode == NormalizationMode.MEAN_STD:
        mean, std = stats["mean"], stats["std"]
        if inverse:
            return x * std + mean
        return (x - mean) / (std + EPS)
    if mode == NormalizationMode.MIN_MAX:
        min_val = stats["min"]
        denom = stats["max"] - min_val
        denom = torch.where(denom == 0, torch.tensor(EPS, dtype=denom.dtype), denom)
        if inverse:
            return (x + 1) / 2 * denom + min_val
        return 2 * (x - min_val) / denom - 1
    raise NotImplementedError(f"normalisation mode {mode} is not supported by the ONNX exporter")


class ActOnnxWrapper(torch.nn.Module):
    """ACT policy with image conversion, normalisation and legacy unit wrapping, in signature I/O."""

    def __init__(
        self,
        policy: ACTPolicy,
        stats: dict[str, dict[str, torch.Tensor]],
        image_keys: list[str],
        state_key: str,
        legacy_units: str | None,
    ) -> None:
        """Wrap ``policy``; ``stats`` maps feature key -> stat name -> tensor (visual stats ``[C,1,1]``)."""
        super().__init__()
        if legacy_units not in LEGACY_UNITS:
            raise ValueError(f"unknown legacy_units '{legacy_units}', expected one of {LEGACY_UNITS}")
        norm_map = policy.config.normalization_mapping
        self._modes = {
            FeatureType.VISUAL: norm_map.get(FeatureType.VISUAL, NormalizationMode.IDENTITY),
            FeatureType.STATE: norm_map.get(FeatureType.STATE, NormalizationMode.IDENTITY),
            FeatureType.ACTION: norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY),
        }
        for mode in self._modes.values():
            if mode not in _SUPPORTED_MODES:
                raise NotImplementedError(f"normalisation mode {mode} is not supported by the ONNX exporter")
        self.policy = policy
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.legacy_units = legacy_units
        self.stats = {
            key: {name: tensor.detach().to(torch.float32).cpu() for name, tensor in stats.get(key, {}).items()}
            for key in [*self.image_keys, state_key, ACTION]
        }

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Inputs: images uint8 ``[H,W,3]`` (signature order), then state float32 ``[D]``; returns ``[horizon, D]``."""
        *images, state = inputs
        batch = {}
        for key, image in zip(self.image_keys, images, strict=True):
            x = (image.to(torch.float32) / 255).permute(2, 0, 1).unsqueeze(0)
            batch[key] = _normalize(x, self._modes[FeatureType.VISUAL], self.stats[key], inverse=False)
        x = state.unsqueeze(0)
        if self.legacy_units == "deg":
            x = x * RAD_TO_DEG
        batch[self.state_key] = _normalize(x, self._modes[FeatureType.STATE], self.stats[self.state_key], False)
        actions = self.policy.predict_action_chunk(batch)
        actions = _normalize(actions, self._modes[FeatureType.ACTION], self.stats[ACTION], inverse=True)
        if self.legacy_units == "deg":
            actions = actions * DEG_TO_RAD
        return actions.squeeze(0)


def _normalization_step(processor, step_type):
    for step in processor.steps:
        if isinstance(step, step_type):
            return step
    raise ValueError(f"processor has no {step_type.__name__}")


def _unflatten(flat: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in flat.items():
        key, stat_name = flat_key.rsplit(".", 1)
        stats.setdefault(key, {})[stat_name] = tensor
    return stats


def extract_stats(preprocessor) -> dict[str, dict[str, torch.Tensor]]:
    """Normalisation stats (key -> stat -> tensor) from the pre-processor's ``NormalizerProcessorStep``."""
    return _unflatten(_normalization_step(preprocessor, NormalizerProcessorStep).state_dict())


def _wrapper_stats(preprocessor, postprocessor) -> dict[str, dict[str, torch.Tensor]]:
    """Observation stats from the pre-processor, action stats from the post-processor (as LeRobot applies them)."""
    stats = {key: value for key, value in extract_stats(preprocessor).items() if key != ACTION}
    post = _unflatten(_normalization_step(postprocessor, UnnormalizerProcessorStep).state_dict())
    if ACTION in post:
        stats[ACTION] = post[ACTION]
    return stats


def signature_image_keys(signature: PolicySignature) -> list[str]:
    """Image observation keys (3-D ``[H,W,3]`` tensors) in signature order."""
    return [key for key, spec in signature.observations.items() if len(spec.shape) == 3]  # noqa: PLR2004


def validate_signature(policy: ACTPolicy, signature: PolicySignature) -> list[str]:
    """Check the signature against the policy features; return the image keys in signature order."""
    config = policy.config
    action_dim = config.output_features[ACTION].shape[0]
    if len(signature.action.joints) != action_dim:
        raise SignatureError(
            f"signature has {len(signature.action.joints)} action joints, policy action dim is {action_dim}"
        )
    image_keys = signature_image_keys(signature)
    expected = {*config.image_features, OBS_STATE}
    if set(signature.observations) != expected:
        raise SignatureError(
            f"signature observations {sorted(signature.observations)} != policy inputs {sorted(expected)}"
        )
    for key in image_keys:
        spec = signature.observations[key]
        channels, height, width = config.image_features[key].shape
        if spec.dtype != "uint8" or tuple(spec.shape) != (height, width, channels):
            raise SignatureError(
                f"'{key}' is {spec.dtype} {spec.shape}, policy expects uint8 {(height, width, channels)}"
            )
    state = signature.observations[OBS_STATE]
    state_dim = config.robot_state_feature.shape[0]
    if state.dtype != "float32" or tuple(state.shape) != (state_dim,):
        raise SignatureError(f"'{OBS_STATE}' is {state.dtype} {state.shape}, policy expects float32 ({state_dim},)")
    return image_keys


def build_wrapper_from_policy(
    policy: ACTPolicy, preprocessor, postprocessor, signature: PolicySignature, opts: ExportOptions
) -> ActOnnxWrapper:
    """Validate the signature and wrap an in-memory policy (CPU, eval mode)."""
    image_keys = validate_signature(policy, signature)
    policy.to("cpu").eval()
    stats = _wrapper_stats(preprocessor, postprocessor)
    return ActOnnxWrapper(policy, stats, image_keys, OBS_STATE, opts.legacy_units).eval()


def load_policy(pretrained: str) -> tuple[ACTPolicy, object, object]:
    """Load an ACT checkpoint and its processors on CPU."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(pretrained)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(pretrained, config=config).to("cpu").eval()
    device_override = {"device": "cpu"}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )
    return policy, preprocessor, postprocessor


def build_wrapper(pretrained: str, signature: PolicySignature, opts: ExportOptions):
    """Load ``pretrained`` and return ``(wrapper, policy, preprocessor, postprocessor)``."""
    policy, preprocessor, postprocessor = load_policy(pretrained)
    wrapper = build_wrapper_from_policy(policy, preprocessor, postprocessor, signature, opts)
    return wrapper, policy, preprocessor, postprocessor


def export_onnx(wrapper: ActOnnxWrapper, signature: PolicySignature, out_path: Path, opset: int) -> None:
    """Trace ``wrapper`` to ONNX with the signature tensors as inputs and ``action`` as output."""
    input_names = [*wrapper.image_keys, wrapper.state_key]
    example_inputs = tuple(
        torch.zeros(signature.observations[key].shape, dtype=getattr(torch, signature.observations[key].dtype))
        for key in input_names
    )
    # The graph has fixed signature shapes, so the tracer's shape warnings do not apply; the TorchScript exporter
    # (dynamo=False, onnxscript is not installed) is deprecated but still supported.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(out_path),
            input_names=input_names,
            output_names=["action"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(out_path))


def reference_chunk(policy, preprocessor, postprocessor, frame: dict[str, np.ndarray], legacy_units) -> np.ndarray:
    """The LeRobot path for one SI frame, as lerobot_rosetta's policy server runs it; returns the SI chunk."""
    state = torch.from_numpy(np.asarray(frame[OBS_STATE], dtype=np.float32)).unsqueeze(0)
    if legacy_units == "deg":
        state = state * RAD_TO_DEG
    observation = {OBS_STATE: state}
    for key in policy.config.image_features:
        # prepare_image: uint8 (C,H,W) -> float32 / 255, contiguous, then the batch dimension.
        image = torch.from_numpy(np.ascontiguousarray(frame[key])).permute(2, 0, 1)
        observation[key] = (image.type(torch.float32) / 255).contiguous().unsqueeze(0)
    observation = preprocessor(observation)
    chunk = policy.predict_action_chunk(observation)
    actions = torch.stack([postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1).squeeze(0)
    if legacy_units == "deg":
        actions = actions * DEG_TO_RAD
    return actions.detach().cpu().numpy()


def check_equivalence(
    onnx_path: Path, policy, preprocessor, postprocessor, frames: list[dict[str, np.ndarray]], opts: ExportOptions
) -> float:
    """Compare ONNX Runtime (CPU) against the LeRobot path; return the max abs error, raise beyond tolerance."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    max_err = 0.0
    for index, frame in enumerate(frames):
        expected = reference_chunk(policy, preprocessor, postprocessor, frame, opts.legacy_units)
        (actual,) = session.run(["action"], {name: frame[name] for name in input_names})
        if actual.shape != expected.shape:
            raise ExportEquivalenceError(f"frame {index}: ONNX action shape {actual.shape} != {expected.shape}")
        err = np.abs(actual - expected)
        max_err = max(max_err, float(err.max()))
        if np.any(err > opts.atol + opts.rtol * np.abs(expected)):
            raise ExportEquivalenceError(
                f"frame {index}: max abs error {float(err.max()):.3e} beyond atol {opts.atol} rtol {opts.rtol}"
            )
    return max_err


def random_frames(signature: PolicySignature, stats, n: int, legacy_units: str | None, seed: int = 0):
    """``n`` SI frames: random uint8 images and states drawn within the state stats (model units -> SI)."""
    rng = np.random.default_rng(seed)
    state_stats = {name: t.numpy().astype(np.float32) for name, t in stats.get(OBS_STATE, {}).items()}
    dim = signature.observations[OBS_STATE].shape[0]
    frames = []
    for _ in range(n):
        if "mean" in state_stats and "std" in state_stats:
            state = state_stats["mean"] + state_stats["std"] * rng.standard_normal(dim).astype(np.float32)
        elif "min" in state_stats and "max" in state_stats:
            state = rng.uniform(state_stats["min"], state_stats["max"]).astype(np.float32)
        else:
            state = rng.standard_normal(dim).astype(np.float32)
        if legacy_units == "deg":
            state = state * np.float32(DEG_TO_RAD)
        frame = {OBS_STATE: state.astype(np.float32)}
        for key in signature_image_keys(signature):
            frame[key] = rng.integers(0, 256, signature.observations[key].shape, dtype=np.uint8)
        frames.append(frame)
    return frames


def dataset_frames(dataset: str, signature: PolicySignature, n: int, legacy_units: str | None):
    """``n`` SI frames evenly spaced through a LeRobot dataset (repo id or local root)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset)
    ds = LeRobotDataset(root.name, root=root) if root.is_dir() else LeRobotDataset(dataset)
    frames = []
    for index in np.linspace(0, len(ds) - 1, n).astype(int):
        item = ds[int(index)]
        state = item[OBS_STATE].to(torch.float32)
        if legacy_units == "deg":
            state = state * DEG_TO_RAD
        frame = {OBS_STATE: state.numpy().astype(np.float32)}
        for key in signature_image_keys(signature):
            frame[key] = (item[key] * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        frames.append(frame)
    return frames


def export_act_from_policy(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    signature: PolicySignature,
    pipeline_hash: str,
    out_dir: Path,
    opts: ExportOptions,
    *,
    on_checked: Callable[[float], None] | None = None,
) -> Path:
    """Export an in-memory ACT to ``out_dir/{model.onnx, policy_signature.yaml}`` after the equivalence check.

    The model is exported and checked in a staging directory next to ``out_dir``; only a model that passes the
    check is moved into ``out_dir``, so a failed export never leaves (or replaces) an artifact. ``on_checked``
    receives the max abs error (rad) once the check has passed.
    """
    if opts.check_frames <= 0 and not opts.unsafe_skip_check:
        raise ExportEquivalenceError("equivalence check disabled (check_frames=0) without unsafe_skip_check")
    wrapper = build_wrapper_from_policy(policy, preprocessor, postprocessor, signature, opts)
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        model_path = staging / MODEL_FILE
        export_onnx(wrapper, signature, model_path, opts.opset)
        if opts.check_frames > 0:
            if opts.dataset is None:
                frames = random_frames(signature, wrapper.stats, opts.check_frames, opts.legacy_units)
            else:
                frames = dataset_frames(opts.dataset, signature, opts.check_frames, opts.legacy_units)
            err = check_equivalence(model_path, policy, preprocessor, postprocessor, frames, opts)
            if on_checked is not None:
                on_checked(err)
        else:
            _LOG.warning("equivalence check SKIPPED (unsafe_skip_check)")
        execution = ExecutionSpec(policy.config.chunk_size, opts.execute_steps, opts.strategy, opts.max_staleness_s)
        dump_signature(replace(signature, execution=execution, pipeline_hash=pipeline_hash), staging / SIGNATURE_FILE)
        out_dir.mkdir(exist_ok=True)
        for name in (MODEL_FILE, SIGNATURE_FILE):
            os.replace(staging / name, out_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir


def export_act(
    pretrained: str,
    signature: PolicySignature,
    pipeline_hash: str,
    out_dir: Path,
    opts: ExportOptions,
    *,
    on_checked: Callable[[float], None] | None = None,
) -> Path:
    """Load a LeRobot ACT checkpoint (hub id or path) on CPU and export it (see ``export_act_from_policy``)."""
    policy, preprocessor, postprocessor = load_policy(pretrained)
    return export_act_from_policy(
        policy, preprocessor, postprocessor, signature, pipeline_hash, out_dir, opts, on_checked=on_checked
    )
