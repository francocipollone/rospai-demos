"""Tests for the policy signature model and pipeline hash."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from pai_policy_io.signature import (
    ExecutionSpec,
    SignatureError,
    TensorSpec,
    dump_signature,
    io_sections_equal,
    load_signature,
    pipeline_hash,
    signature_from_dict,
)

FIXTURE = Path(__file__).parent / "fixtures" / "policy_signature_example.yaml"


def test_load_example_signature_round_trips(tmp_path):
    sig = load_signature(FIXTURE)
    assert sig.fps == 50 and sig.robot == "so_arm101"
    assert sig.observations["observation.images.wrist"] == TensorSpec("uint8", (480, 480, 3), layout="HWC_RGB")
    assert sig.observations["observation.state"].shape == (6,)
    assert sig.action.joints[0] == "shoulder_pan_joint" and sig.action.kinds == ("continuous",) * 6
    assert sig.execution == ExecutionSpec(100, 30, "blend", 0.2)
    dump_signature(sig, tmp_path / "s.yaml")
    assert load_signature(tmp_path / "s.yaml") == sig


def test_kinds_length_mismatch_raises():
    data = yaml.safe_load(FIXTURE.read_text())
    data["action"]["kinds"] = ["continuous"]
    with pytest.raises(SignatureError, match="kinds"):
        signature_from_dict(data)


def test_unknown_strategy_raises():
    data = yaml.safe_load(FIXTURE.read_text())
    data["execution"]["strategy"] = "magic"
    with pytest.raises(SignatureError, match="strategy"):
        signature_from_dict(data)


def test_pipeline_hash_is_deterministic_and_ignores_execution():
    sig = load_signature(FIXTURE)
    h1 = pipeline_hash(sig, "cfg", {"rosetta": "0.2.0"})
    assert h1.startswith("sha256:") and len(h1) == 7 + 64
    assert h1 == pipeline_hash(dataclasses.replace(sig, execution=None), "cfg", {"rosetta": "0.2.0"})
    assert h1 != pipeline_hash(sig, "cfg2", {"rosetta": "0.2.0"})
    assert h1 != pipeline_hash(sig, "cfg", {"rosetta": "0.2.1"})


def test_io_sections_equal_ignores_execution_and_hash():
    sig = load_signature(FIXTURE)
    other = dataclasses.replace(sig, execution=None, pipeline_hash="sha256:ffff")
    assert io_sections_equal(sig, other)
    changed = dataclasses.replace(sig, action=dataclasses.replace(sig.action, action_type="delta_joint"))
    assert not io_sections_equal(sig, changed)
