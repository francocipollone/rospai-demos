"""Policy signature model, YAML I/O and pipeline hash (design spec 5.7-5.8)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SIGNATURE_VERSION = 1

ACTION_KINDS = ("continuous", "binary", "quaternion", "rotation_6d")
ACTION_TYPES = ("absolute_joint", "delta_joint", "ee_pose")
STRATEGIES = ("replace", "blend", "temporal_ensemble")


class SignatureError(ValueError):
    """Raised when a policy signature is malformed."""


@dataclass(frozen=True)
class TensorSpec:
    """Observation tensor description: an image (shape + layout) or a joint vector (joints)."""

    dtype: str
    shape: tuple[int, ...]
    layout: str | None = None
    units: str | None = None
    joints: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ActionSpec:
    """Action vector description: one kind per joint."""

    dtype: str
    units: str
    action_type: str
    joints: tuple[str, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionSpec:
    """How an exported model's action chunks are executed."""

    horizon: int
    execute_steps: int
    strategy: str
    max_staleness_s: float


@dataclass(frozen=True)
class PolicySignature:
    """Machine-readable policy signature (policy_signature.yaml)."""

    signature_version: int
    robot: str
    fps: int
    observations: dict[str, TensorSpec]
    action: ActionSpec
    execution: ExecutionSpec | None = None
    pipeline_hash: str | None = None


def _tensor_to_dict(spec: TensorSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": spec.dtype}
    if spec.joints is None:
        out["shape"] = list(spec.shape)
    if spec.layout is not None:
        out["layout"] = spec.layout
    if spec.units is not None:
        out["units"] = spec.units
    if spec.joints is not None:
        out["joints"] = list(spec.joints)
    return out


def _action_to_dict(action: ActionSpec) -> dict[str, Any]:
    return {
        "dtype": action.dtype,
        "units": action.units,
        "action_type": action.action_type,
        "joints": list(action.joints),
        "kinds": list(action.kinds),
    }


def _io_dict(sig: PolicySignature) -> dict[str, Any]:
    return {
        "observations": {name: _tensor_to_dict(spec) for name, spec in sig.observations.items()},
        "action": _action_to_dict(sig.action),
    }


def signature_to_dict(sig: PolicySignature) -> dict[str, Any]:
    """Convert a signature to a YAML-ready dict, omitting unset sections."""
    out: dict[str, Any] = {"signature_version": sig.signature_version, "robot": sig.robot, "fps": sig.fps}
    out.update(_io_dict(sig))
    if sig.execution is not None:
        ex = sig.execution
        out["execution"] = {
            "horizon": ex.horizon,
            "execute_steps": ex.execute_steps,
            "strategy": ex.strategy,
            "max_staleness_s": ex.max_staleness_s,
        }
    if sig.pipeline_hash is not None:
        out["pipeline_hash"] = sig.pipeline_hash
    return out


def _require(data: dict, key: str, where: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise SignatureError(f"missing '{key}' in {where}")
    return data[key]


def _tensor_from_dict(name: str, data: dict) -> TensorSpec:
    where = f"observation '{name}'"
    dtype = _require(data, "dtype", where)
    joints = tuple(data["joints"]) if data.get("joints") is not None else None
    if "shape" in data:
        shape = tuple(int(n) for n in data["shape"])
    elif joints is not None:
        shape = (len(joints),)
    else:
        raise SignatureError(f"{where} needs 'shape' or 'joints'")
    return TensorSpec(dtype, shape, layout=data.get("layout"), units=data.get("units"), joints=joints)


def _action_from_dict(data: dict) -> ActionSpec:
    joints = tuple(_require(data, "joints", "action"))
    kinds = tuple(_require(data, "kinds", "action"))
    action_type = _require(data, "action_type", "action")
    if len(kinds) != len(joints):
        raise SignatureError(f"action kinds length {len(kinds)} != joints length {len(joints)}")
    for kind in kinds:
        if kind not in ACTION_KINDS:
            raise SignatureError(f"unknown action kind '{kind}', expected one of {ACTION_KINDS}")
    if action_type not in ACTION_TYPES:
        raise SignatureError(f"unknown action_type '{action_type}', expected one of {ACTION_TYPES}")
    return ActionSpec(_require(data, "dtype", "action"), _require(data, "units", "action"), action_type, joints, kinds)


def _execution_from_dict(data: dict) -> ExecutionSpec:
    strategy = _require(data, "strategy", "execution")
    if strategy not in STRATEGIES:
        raise SignatureError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")
    return ExecutionSpec(
        int(_require(data, "horizon", "execution")),
        int(_require(data, "execute_steps", "execution")),
        strategy,
        float(_require(data, "max_staleness_s", "execution")),
    )


def signature_from_dict(data: dict) -> PolicySignature:
    """Build a signature from a parsed YAML dict, raising SignatureError on bad input."""
    observations = _require(data, "observations", "signature")
    execution = data.get("execution")
    return PolicySignature(
        signature_version=int(_require(data, "signature_version", "signature")),
        robot=_require(data, "robot", "signature"),
        fps=int(_require(data, "fps", "signature")),
        observations={name: _tensor_from_dict(name, spec) for name, spec in observations.items()},
        action=_action_from_dict(_require(data, "action", "signature")),
        execution=_execution_from_dict(execution) if execution is not None else None,
        pipeline_hash=data.get("pipeline_hash"),
    )


def load_signature(path: Path) -> PolicySignature:
    """Load a policy_signature.yaml file."""
    return signature_from_dict(yaml.safe_load(Path(path).read_text()))


def dump_signature(sig: PolicySignature, path: Path) -> None:
    """Write a signature to a policy_signature.yaml file."""
    Path(path).write_text(yaml.safe_dump(signature_to_dict(sig), sort_keys=False))


def pipeline_hash(sig: PolicySignature, pipeline_config_text: str, component_versions: dict[str, str]) -> str:
    """Hash the I/O sections, pipeline configuration and component versions (execution excluded)."""
    payload = {**_io_dict(sig), "pipeline_config": pipeline_config_text, "component_versions": component_versions}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def io_sections_equal(a: PolicySignature, b: PolicySignature) -> bool:
    """Compare only the observations and action sections."""
    return _io_dict(a) == _io_dict(b)
