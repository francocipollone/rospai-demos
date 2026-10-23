"""Tests for deriving a policy signature from a Rosetta contract."""

from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory
from rosetta.contract.schema import load_contract

from pai_policy_io.contract_signature import contract_pipeline_hash, signature_from_contract
from pai_policy_io.signature import SignatureError

CONTRACT = Path(get_package_share_directory("pai_data_collection")) / "config/rosetta/so_arm101_policy_io.yaml"
LEGACY = CONTRACT.with_name("so_arm101.yaml")
FPS = 50
JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper_joint",
)


def test_si_contract_loads():
    """The SI contract passes Rosetta validation."""
    load_contract(str(CONTRACT))


def test_signature_from_si_contract():
    """The SI contract maps to the expected signature."""
    sig = signature_from_contract(CONTRACT, robot="so_arm101")
    assert sig.fps == FPS and sig.execution is None and sig.pipeline_hash is None
    assert set(sig.observations) == {"observation.images.wrist", "observation.images.static", "observation.state"}
    assert sig.observations["observation.images.static"].shape == (480, 480, 3)
    assert sig.observations["observation.state"].joints == JOINTS
    assert sig.action.joints == JOINTS and sig.action.action_type == "absolute_joint"


def test_legacy_degree_contract_is_rejected():
    """A contract with unit operators is rejected."""
    with pytest.raises(SignatureError, match="unit operator"):
        signature_from_contract(LEGACY, robot="so_arm101")


def test_contract_pipeline_hash_changes_with_contract_text(tmp_path):
    """Editing the contract text changes the pipeline hash."""
    changed = tmp_path / "changed.yaml"
    changed.write_text(CONTRACT.read_text().replace("depth: 50", "depth: 51"))
    _, original = contract_pipeline_hash(CONTRACT, "so_arm101")
    _, modified = contract_pipeline_hash(changed, "so_arm101")
    assert original != modified
