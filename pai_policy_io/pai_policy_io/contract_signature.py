"""Derive a policy signature from a Rosetta contract (phase-1 bridge)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rosetta.contract.schema import load_contract
from rosetta.contract.specs import iter_action_specs, iter_observation_specs

from pai_policy_io.signature import (
    SIGNATURE_VERSION,
    ActionSpec,
    PolicySignature,
    SignatureError,
    TensorSpec,
    pipeline_hash,
)

UNIT_OPERATORS = frozenset({"rad2deg", "deg2rad"})


def _check_si(spec) -> None:
    if any(op.name in UNIT_OPERATORS for op in spec.operators):
        raise SignatureError("contract applies a unit operator; the Policy I/O convention is SI")


def _observation_spec(spec) -> TensorSpec:
    if spec.is_image:
        height, width = spec.image_resize
        return TensorSpec("uint8", (height, width, 3), layout="HWC_RGB")
    _check_si(spec)
    joints = tuple(name.removeprefix("position.") for name in spec.names)
    return TensorSpec("float32", (len(joints),), units="si", joints=joints)


def signature_from_contract(contract_path: Path, robot: str) -> PolicySignature:
    """Build the policy signature declared by an SI Rosetta contract."""
    contract = load_contract(str(contract_path))
    observations = {spec.key: _observation_spec(spec) for spec in iter_observation_specs(contract)}
    actions = list(iter_action_specs(contract))
    if len(actions) != 1:
        raise SignatureError(f"expected exactly one action spec, found {len(actions)}")
    _check_si(actions[0])
    joints = tuple(actions[0].names)
    action = ActionSpec("float32", "si", "absolute_joint", joints, ("continuous",) * len(joints))
    return PolicySignature(SIGNATURE_VERSION, robot, int(contract.fps), observations, action)


def _dist_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def contract_component_versions() -> dict[str, str]:
    """Versions of the components that shape the contract-derived pipeline."""
    return {"rosetta": _dist_version("rosetta"), "pai_policy_io": _dist_version("pai_policy_io")}


def contract_pipeline_hash(contract_path: Path, robot: str) -> tuple[PolicySignature, str]:
    """Return the contract's signature and its pipeline hash."""
    sig = signature_from_contract(contract_path, robot)
    text = Path(contract_path).read_text()
    return sig, pipeline_hash(sig, text, contract_component_versions())
