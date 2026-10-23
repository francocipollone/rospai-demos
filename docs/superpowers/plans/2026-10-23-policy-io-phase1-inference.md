# Policy I/O Phase 0+1: Tensor Messages, Inference Node, ACT Export, Rosetta Bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the SO-ARM101 with the existing pretrained ACT checkpoint through a new generic C++ ONNX inference node, with the Policy I/O Convention (`TensorList` topics, `policy_signature.yaml`, pipeline hash) enforced end to end, using Rosetta as a temporary bridge.

**Architecture:** Three new packages. `pai_tensor_msgs` defines `Tensor`/`TensorList`/`PipelineInfo`. `pai_policy_io` (Python) owns the signature + pipeline-hash code, a Rosetta `PolicyRunner` plugin that publishes Rosetta frames as `/policy/observation` and consumes `/policy/action_chunk`, and the ACT → ONNX exporter with normalisation and legacy `rad↔deg` baked in. `pai_policy_inference` (C++) is the robot-agnostic inference node: `TensorList` in, `TensorList` out, ONNX Runtime backend, latest-only scheduling, signature and hash checks at startup.

**Tech Stack:** ROS 2 Lyrical (RoboStack via Pixi), rclcpp + rclcpp_components, rosidl, yaml-cpp, ONNX Runtime C++ (`onnxruntime-cpp`, conda-forge), gtest; Python 3.14, rclpy, Rosetta 0.2.0 plugin API, LeRobot 0.6.0, torch.onnx, onnx, onnxruntime (Python), pytest.

**Spec:** [`docs/design/policy-io-and-inference.md`](../../design/policy-io-and-inference.md) — this plan implements migration phases **0** and **1** of §11. Phases 2 (action executor), 3 (pre-processing components), 4 (data pipeline) and 5 (acceleration) each get their own plan once this one ships; this plan must not pre-build them.

## Global Constraints

- Tensor names are the join key everywhere: `TensorList` entry name = signature key = ONNX graph input/output name (spec §5.2). No positional mapping anywhere.
- Units on topics and in the ONNX interface are **SI** (radians); legacy degree checkpoints bake `rad→deg` into inputs and `deg→rad` into outputs at export only (spec §5.3). No unit operator in the pipeline.
- Images: `uint8`, HWC, RGB. Vectors: 1-D `float32` in signature joint order. Action chunk: `float32 [horizon, action_dim]` named `action`, with `header.stamp` copied from the observation it came from (spec §5.6, §7.1).
- `fps = 50` for SO-ARM101; frame clock on the ROS clock (`use_sim_time` must work).
- Inference node has no robot knowledge: no joint names, units or controllers (spec §7.1).
- Pipeline hash = `sha256:` + hex SHA-256 over the signature `observations` + `action` sections, the pipeline configuration text, and component versions (spec §5.8). Mismatch is a startup error; override parameter logs at ERROR level.
- Exported model is a directory `model.onnx` + `policy_signature.yaml`; no statistics file travels with it (spec §7.2).
- The export must pass a numerical-equivalence test against the LeRobot checkpoint or fail (spec §7.2).
- Python: ruff config in `ruff.toml` (line length 120, docstrings required: `D` rules). C++: `-Wall -Wextra -Wpedantic`, C++17. CMake files pass `cmake-lint`. YAML/Markdown pass prettier. Run `pixi run lint` before every commit.
- Every new conda dependency goes in `pixi.toml` **and** the regenerated `pixi.lock` (CI runs `frozen: true`).
- Builds run via `pixi run build` (colcon, cmake-args in `colcon_defaults.yaml`). Tests via `pixi run colcon test --packages-select <pkg>` then `colcon test-result --verbose`.
- Python pytest tests in ament_cmake packages use `ament_add_pytest_test(... ENV "PYTEST_ADDOPTS=-p no:launch_testing -p no:launch_ros")` as `pai_teleop_ik/CMakeLists.txt` does.

## Review Focus

1. **`/policy/action_chunk` arrives with `horizon` shorter than `execution.horizon`, or `action_dim ≠ len(action.joints)`** — the bridge must reject the chunk and log, never index out of range or publish a truncated command (test in Task 4, `ChunkFollower.push` shape check).
2. **A chunk with NaN/Inf** — must be dropped whole; the last valid chunk keeps being followed (Task 4).
3. **Observation `TensorList` missing an input the ONNX graph needs, or carrying the wrong dtype/shape** — the inference node must raise a clear error naming the tensor and skip that observation, not call ORT with a dangling view (Task 7, `InferenceCore` tests).
4. **Sim time paused or clock reset** — bridge tick index `(now − t_obs)/dt` goes negative or huge; must clamp to `[0, horizon−1]` and never crash (Task 4, `action_at` tests with `now < t_obs`).
5. **`PipelineInfo` never arrives (bridge not running)** — the inference node must fail within `pipeline_info_timeout_s` with an actionable message instead of waiting forever or silently running unchecked (Task 8 verification step; core logic in Task 7 `HashGate`).

---

## File structure

```
pai_tensor_msgs/                      # Task 1
  CMakeLists.txt, package.xml
  msg/Tensor.msg, msg/TensorList.msg, msg/PipelineInfo.msg

pai_policy_io/                        # Tasks 2–5, 9 (ament_python)
  package.xml, setup.py, setup.cfg, resource/pai_policy_io
  pai_policy_io/__init__.py
  pai_policy_io/signature.py          # dataclasses, load/dump, pipeline_hash          (Task 2)
  pai_policy_io/contract_signature.py # signature_from_contract (Rosetta contract → signature) (Task 3)
  pai_policy_io/tensor_msgs.py        # numpy <-> Tensor/TensorList                     (Task 4)
  pai_policy_io/chunk_follower.py     # time-indexed chunk following, no ROS            (Task 4)
  pai_policy_io/rosetta_bridge.py     # TensorListBridgeRunner (rosetta.policy_runners) (Task 5)
  pai_policy_io/export/__init__.py
  pai_policy_io/export/lerobot_act.py # ACT -> ONNX exporter + equivalence check       (Task 9)
  pai_policy_io/export/cli.py         # `pai_export_act` console script                 (Task 9)
  test/test_signature.py, test_contract_signature.py, test_tensor_msgs.py,
  test/test_chunk_follower.py, test_rosetta_bridge.py, test_export_lerobot_act.py
  test/fixtures/policy_signature_example.yaml

pai_policy_inference/                 # Tasks 6–8 (ament_cmake, C++17)
  CMakeLists.txt, package.xml
  include/pai_policy_inference/signature.hpp
  include/pai_policy_inference/tensor_view.hpp     # TensorView/OwnedTensor + dtype helpers
  include/pai_policy_inference/inference_backend.hpp
  include/pai_policy_inference/ort_backend.hpp
  include/pai_policy_inference/inference_core.hpp  # bind by name, validate, run, LatestSlot, HashGate
  include/pai_policy_inference/inference_node.hpp
  src/signature.cpp, tensor_view.cpp, ort_backend.cpp, inference_core.cpp,
  src/inference_node.cpp, src/inference_node_main.cpp
  test/test_signature.cpp, test_tensor_view.cpp, test_inference_core.cpp
  test/fixtures/make_fixture.py, test/fixtures/add_one.onnx, test/fixtures/policy_signature.yaml

pai_data_collection/config/rosetta/so_arm101_policy_io.yaml      # SI contract (Task 3)
pai_data_collection/config/rosetta/policy_runner_tensorlist.yaml  # runner params (Task 5)
pai_bringup/launch/policy_io_inference.launch.py                  # Task 8
docs/demos/onnx-inference.md                                      # Task 10
pixi.toml, pixi.lock                                              # Task 1
```

---

### Task 1: Dependencies and `pai_tensor_msgs`

**Files:**
- Modify: `pixi.toml` (dependencies table), `pixi.lock` (regenerated)
- Create: `pai_tensor_msgs/package.xml`, `pai_tensor_msgs/CMakeLists.txt`, `pai_tensor_msgs/msg/Tensor.msg`, `pai_tensor_msgs/msg/TensorList.msg`, `pai_tensor_msgs/msg/PipelineInfo.msg`

**Interfaces:**
- Produces: `pai_tensor_msgs/msg/Tensor` `{string name; uint8 dtype; int64[] shape; uint8[] data}` with constants `DTYPE_UINT8=0 DTYPE_INT32=1 DTYPE_INT64=2 DTYPE_FLOAT32=3 DTYPE_FLOAT64=4 DTYPE_STRING=5`; `TensorList` `{std_msgs/Header header; Tensor[] tensors}`; `PipelineInfo` `{std_msgs/Header header; string signature_yaml; string pipeline_hash}`.
- Decision: `data` is row-major, C-contiguous, native-endian; no strides field (spec Q3 default: hosted here until NVIDIA's `ExperimentalTensor` is adopted; in Lyrical `uint8[]` is already `rosidl::Buffer<uint8_t>` in C++, which satisfies G8). `DTYPE_STRING` stores UTF-8 bytes with `shape=[n_bytes]` (for `task`).

- [ ] **Step 1: Add dependencies to `pixi.toml` `[dependencies]`**

```toml
# Policy I/O (pai_tensor_msgs, pai_policy_inference, pai_policy_io)
ros-lyrical-rosidl-default-generators = "*"
ros-lyrical-rosidl-default-runtime = "*"
ros-lyrical-rclcpp-components = "*"
ros-lyrical-ament-cmake-gtest = "*"
yaml-cpp = ">=0.8,<0.9"
onnxruntime-cpp = ">=1.20,<2"
onnxruntime = ">=1.20,<2"
onnx = "*"
```

- [ ] **Step 2: Regenerate the lock and verify the solve**

Run: `pixi install`
Expected: solve succeeds; `git diff --stat pixi.lock` shows additions; `pixi run python -c "import onnxruntime, onnx; print(onnxruntime.__version__)"` prints a version ≥ 1.20.

- [ ] **Step 3: Write the message files**

`msg/Tensor.msg`:
```
# One named, dense, row-major tensor. See docs/design/policy-io-and-inference.md §5.
uint8 DTYPE_UINT8=0
uint8 DTYPE_INT32=1
uint8 DTYPE_INT64=2
uint8 DTYPE_FLOAT32=3
uint8 DTYPE_FLOAT64=4
uint8 DTYPE_STRING=5

string name      # e.g. observation.images.wrist, observation.state, action
uint8 dtype      # one of DTYPE_*
int64[] shape    # e.g. [480, 480, 3]; [] for a scalar
uint8[] data     # C-contiguous, native endian; len == prod(shape) * itemsize
```
`msg/TensorList.msg`: `std_msgs/Header header` (stamp = frame tick time), `Tensor[] tensors`.
`msg/PipelineInfo.msg`: `std_msgs/Header header`, `string signature_yaml`, `string pipeline_hash`.

- [ ] **Step 4: Write `package.xml` (format 3, `rosidl_interface_packages` member) and `CMakeLists.txt`**

`rosidl_generate_interfaces(${PROJECT_NAME} "msg/Tensor.msg" "msg/TensorList.msg" "msg/PipelineInfo.msg" DEPENDENCIES std_msgs)`; `ament_export_dependencies(rosidl_default_runtime)`. Maintainer/license as in `pai_teleop_ik/package.xml`.

- [ ] **Step 5: Build and verify the interface is registered**

Run: `pixi run build --packages-select pai_tensor_msgs && pixi run ros2 interface show pai_tensor_msgs/msg/Tensor`
Expected: the six `DTYPE_*` constants and four fields print.

- [ ] **Step 6: Lint and commit**

```bash
pixi run lint
git add pixi.toml pixi.lock pai_tensor_msgs
git commit -m "feat(pai_tensor_msgs): add Tensor, TensorList and PipelineInfo messages"
```

---

### Task 2: `pai_policy_io` package and the policy signature

**Files:**
- Create: `pai_policy_io/package.xml`, `setup.py`, `setup.cfg`, `resource/pai_policy_io`, `pai_policy_io/__init__.py`, `pai_policy_io/signature.py`
- Test: `pai_policy_io/test/test_signature.py`, `pai_policy_io/test/fixtures/policy_signature_example.yaml` (the spec §5.7 example, verbatim, with `pipeline_hash: "sha256:0000"`)

**Interfaces:**
- Produces (`pai_policy_io/signature.py`):
  ```python
  SIGNATURE_VERSION = 1
  @dataclass(frozen=True) class TensorSpec: dtype: str; shape: tuple[int, ...]; layout: str | None = None; units: str | None = None; joints: tuple[str, ...] | None = None
  @dataclass(frozen=True) class ActionSpec: dtype: str; units: str; action_type: str; joints: tuple[str, ...]; kinds: tuple[str, ...]
  @dataclass(frozen=True) class ExecutionSpec: horizon: int; execute_steps: int; strategy: str; max_staleness_s: float
  @dataclass(frozen=True) class PolicySignature: signature_version: int; robot: str; fps: int; observations: dict[str, TensorSpec]; action: ActionSpec; execution: ExecutionSpec | None = None; pipeline_hash: str | None = None
  def signature_to_dict(sig: PolicySignature) -> dict          # YAML-ready, omits None sections
  def signature_from_dict(data: dict) -> PolicySignature       # raises SignatureError on bad input
  def load_signature(path: Path) -> PolicySignature
  def dump_signature(sig: PolicySignature, path: Path) -> None
  def pipeline_hash(sig: PolicySignature, pipeline_config_text: str, component_versions: dict[str, str]) -> str
  def io_sections_equal(a: PolicySignature, b: PolicySignature) -> bool   # observations + action only
  class SignatureError(ValueError)
  ```
- Rules pinned by tests: `TensorSpec.shape` for vector specs is `(len(joints),)` (derived when `joints` given and `shape` omitted); `ActionSpec.kinds` length must equal `joints` length; `kinds` ∈ {`continuous`,`binary`,`quaternion`,`rotation_6d`}; `action_type` ∈ {`absolute_joint`,`delta_joint`,`ee_pose`}; `strategy` ∈ {`replace`,`blend`,`temporal_ensemble`}.
- `pipeline_hash` = `"sha256:" + sha256(json.dumps({"observations": ..., "action": ..., "pipeline_config": text, "component_versions": versions}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()`.

- [ ] **Step 1: Write the failing tests**

```python
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
    data = yaml.safe_load(FIXTURE.read_text()); data["action"]["kinds"] = ["continuous"]
    with pytest.raises(SignatureError, match="kinds"):
        signature_from_dict(data)

def test_unknown_strategy_raises(): ...  # strategy: "magic" -> SignatureError match "strategy"

def test_pipeline_hash_is_deterministic_and_ignores_execution():
    sig = load_signature(FIXTURE)
    h1 = pipeline_hash(sig, "cfg", {"rosetta": "0.2.0"})
    assert h1.startswith("sha256:") and len(h1) == 7 + 64
    assert h1 == pipeline_hash(dataclasses.replace(sig, execution=None), "cfg", {"rosetta": "0.2.0"})
    assert h1 != pipeline_hash(sig, "cfg2", {"rosetta": "0.2.0"})
    assert h1 != pipeline_hash(sig, "cfg", {"rosetta": "0.2.1"})

def test_io_sections_equal_ignores_execution_and_hash(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_signature.py -q`
Expected: FAIL with `ModuleNotFoundError: pai_policy_io`.

- [ ] **Step 3: Create the ament_python package and implement `signature.py`**

`setup.py` with `packages=find_packages(exclude=["test"])`, `data_files` for `resource/` and `package.xml`, `install_requires=["setuptools", "numpy", "pyyaml"]`; `package.xml` with `<build_type>ament_python</build_type>`, `<exec_depend>` on `rclpy`, `pai_tensor_msgs`, `rosetta`, `python3-yaml`, `python3-numpy`; `<test_depend>python3-pytest</test_depend>`. Implement the dataclasses and functions above; YAML via `yaml.safe_load`/`yaml.safe_dump(sort_keys=False)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_signature.py -q`
Expected: 5 passed.

- [ ] **Step 5: Build with colcon, run via `colcon test`, lint, commit**

```bash
pixi run build --packages-select pai_policy_io
pixi run colcon test --packages-select pai_policy_io && pixi run colcon test-result --verbose
pixi run lint
git add pai_policy_io && git commit -m "feat(pai_policy_io): policy signature model, YAML I/O and pipeline hash"
```

---

### Task 3: SI contract and `signature_from_contract`

**Files:**
- Create: `pai_data_collection/config/rosetta/so_arm101_policy_io.yaml`, `pai_policy_io/pai_policy_io/contract_signature.py`
- Test: `pai_policy_io/test/test_contract_signature.py`

**Interfaces:**
- Consumes: `PolicySignature`, `TensorSpec`, `ActionSpec` from Task 2; Rosetta `load_contract(path) -> Contract`, `iter_observation_specs(contract)`, `iter_action_specs(contract)` (`ObservationStreamSpec.key/.names/.is_image/.image_resize/.dtype`, `ActionStreamSpec.names`).
- Produces:
  ```python
  def signature_from_contract(contract_path: Path, robot: str) -> PolicySignature   # execution=None, pipeline_hash=None
  def contract_component_versions() -> dict[str, str]   # {"rosetta": <importlib.metadata.version("rosetta")>, "pai_policy_io": <version("pai_policy_io")>}
  def contract_pipeline_hash(contract_path: Path, robot: str) -> tuple[PolicySignature, str]   # (sig, pipeline_hash(sig, contract_text, versions))
  ```
- Mapping rules: image spec → `TensorSpec("uint8", (h, w, 3), layout="HWC_RGB")` with `(h, w) = spec.image_resize`; non-image observation → `TensorSpec("float32", units="si", joints=<names with a leading "position." stripped>)`; action → `ActionSpec("float32", "si", "absolute_joint", joints=names, kinds=("continuous",)*n)`. A contract whose state or action declares `rad2deg`/`deg2rad` in `apply` raises `SignatureError("contract applies a unit operator; the Policy I/O convention is SI")`. More than one action spec raises `SignatureError`.
- The SI contract: copy of `so_arm101.yaml` with `robot_type: so_arm101`, `fps: 50`, same image specs, state spec without `apply`, action spec `select: [shoulder_pan_joint, shoulder_lift_joint, elbow_flex_joint, wrist_flex_joint, wrist_roll_joint, gripper_joint]` and `apply: [clamp: { min: -3.14159, max: 3.14159 }]`. Header comment: "Policy I/O (SI) contract for the phase-1 Rosetta bridge. No unit operators: tensors on /policy/* are radians."

- [ ] **Step 1: Write the failing tests**

```python
CONTRACT = Path(get_package_share_directory("pai_data_collection")) / "config/rosetta/so_arm101_policy_io.yaml"
LEGACY = CONTRACT.with_name("so_arm101.yaml")
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint", "wrist_flex_joint", "wrist_roll_joint", "gripper_joint")

def test_si_contract_loads(): load_contract(str(CONTRACT))

def test_signature_from_si_contract():
    sig = signature_from_contract(CONTRACT, robot="so_arm101")
    assert sig.fps == 50 and sig.execution is None and sig.pipeline_hash is None
    assert set(sig.observations) == {"observation.images.wrist", "observation.images.static", "observation.state"}
    assert sig.observations["observation.images.static"].shape == (480, 480, 3)
    assert sig.observations["observation.state"].joints == JOINTS
    assert sig.action.joints == JOINTS and sig.action.action_type == "absolute_joint"

def test_legacy_degree_contract_is_rejected():
    with pytest.raises(SignatureError, match="unit operator"):
        signature_from_contract(LEGACY, robot="so_arm101")

def test_contract_pipeline_hash_changes_with_contract_text(tmp_path): ...  # copy contract, change a QoS depth -> different hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_contract_signature.py -q`
Expected: FAIL (`FileNotFoundError` for the contract, `ImportError` for `contract_signature`).

- [ ] **Step 3: Write the SI contract and implement `contract_signature.py`**

Detect unit operators via `spec.operators` names (`op.name in {"rad2deg", "deg2rad"}`); verify the attribute name against `rosetta/contract/operators.py` before relying on it.

- [ ] **Step 4: Rebuild `pai_data_collection` (installs the new contract) and run the tests**

Run: `pixi run build --packages-select pai_data_collection pai_policy_io && cd pai_policy_io && pixi run python -m pytest test/test_contract_signature.py -q`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
pixi run lint
git add pai_data_collection/config/rosetta/so_arm101_policy_io.yaml pai_policy_io
git commit -m "feat(pai_policy_io): derive the policy signature from the SI Rosetta contract"
```

---

### Task 4: numpy ↔ `Tensor` conversion and the chunk follower

**Files:**
- Create: `pai_policy_io/pai_policy_io/tensor_msgs.py`, `pai_policy_io/pai_policy_io/chunk_follower.py`
- Test: `pai_policy_io/test/test_tensor_msgs.py`, `pai_policy_io/test/test_chunk_follower.py`

**Interfaces:**
- Consumes: `pai_tensor_msgs.msg.Tensor`, `TensorList` (Task 1).
- Produces (`tensor_msgs.py`):
  ```python
  NP_TO_DTYPE: dict[np.dtype, int]   # uint8→0, int32→1, int64→2, float32→3, float64→4
  DTYPE_TO_NP: dict[int, np.dtype]
  def array_to_tensor(name: str, array: np.ndarray) -> Tensor        # ascontiguousarray; raises TypeError on unsupported dtype
  def string_to_tensor(name: str, text: str) -> Tensor               # DTYPE_STRING, shape=[n_bytes]
  def tensor_to_array(tensor: Tensor) -> np.ndarray | str            # str for DTYPE_STRING; raises ValueError when len(data) != prod(shape)*itemsize
  def frame_to_tensorlist(frame: dict[str, np.ndarray | str], stamp: Time, casts: dict[str, np.dtype]) -> TensorList   # casts: key -> dtype to astype() before packing
  def tensorlist_to_frame(msg: TensorList) -> dict[str, np.ndarray | str]
  ```
- Produces (`chunk_follower.py`, no ROS):
  ```python
  class ChunkFollower:
      def __init__(self, action_dim: int, dt_ns: int): ...
      def push(self, t_obs_ns: int, chunk: np.ndarray) -> bool   # accepts float32 [H, action_dim], H>=1, finite; returns False (and keeps the previous chunk) otherwise
      def action_at(self, now_ns: int) -> np.ndarray | None       # None before any chunk; index clamp((now - t_obs) // dt_ns, 0, H-1)
      def remaining(self, now_ns: int) -> int                     # H - index, 0 when no chunk
      def reset(self) -> None
      @property
      def t_obs_ns(self) -> int | None
  ```
  `push` is "replace": a newer chunk (larger `t_obs_ns`) replaces the current one; an older one is ignored and returns `False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_array_round_trip_uint8_hwc():
    img = np.arange(2*3*3, dtype=np.uint8).reshape(2, 3, 3)
    t = array_to_tensor("observation.images.wrist", img)
    assert t.dtype == Tensor.DTYPE_UINT8 and list(t.shape) == [2, 3, 3] and len(t.data) == 18
    np.testing.assert_array_equal(tensor_to_array(t), img)

def test_float32_vector_round_trip(): ...        # dtype 3, 6 elems -> 24 bytes
def test_string_round_trip(): assert tensor_to_array(string_to_tensor("task", "place cubes on tray")) == "place cubes on tray"
def test_bad_length_raises(): ...                # data truncated -> ValueError
def test_frame_to_tensorlist_casts_and_stamps():
    msg = frame_to_tensorlist({"observation.state": np.zeros(6, np.float64)}, stamp, casts={"observation.state": np.float32})
    assert msg.header.stamp == stamp and msg.tensors[0].dtype == Tensor.DTYPE_FLOAT32

# chunk follower, dt = 20 ms
def test_action_at_indexes_by_time():
    f = ChunkFollower(6, 20_000_000); chunk = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 6))
    assert f.push(1_000_000_000, chunk)
    assert f.action_at(1_000_000_000)[0] == 0 and f.action_at(1_040_000_000)[0] == 2
    assert f.action_at(1_000_000_000 + 10 * 20_000_000)[0] == 4          # past the end: hold last
    assert f.action_at(900_000_000)[0] == 0 and f.remaining(1_040_000_000) == 3   # before t_obs: clamp to 0

def test_push_rejects_wrong_dim_nan_and_older():
    f = ChunkFollower(6, 20_000_000); good = np.zeros((3, 6), np.float32)
    assert f.push(2_000, good)
    assert not f.push(3_000, np.zeros((3, 5), np.float32))
    assert not f.push(3_000, np.full((3, 6), np.nan, np.float32))
    assert not f.push(1_000, np.ones((3, 6), np.float32))
    assert f.t_obs_ns == 2_000 and f.action_at(2_000)[0] == 0

def test_action_at_before_any_chunk_is_none(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_tensor_msgs.py test/test_chunk_follower.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `tensor_msgs.py` and `chunk_follower.py`**

`tensor_to_array` uses `np.frombuffer(bytes(tensor.data), dtype).reshape(shape).copy()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: same command. Expected: 9 passed.

- [ ] **Step 5: Lint and commit**

```bash
pixi run lint
git add pai_policy_io && git commit -m "feat(pai_policy_io): Tensor message conversion and time-indexed chunk follower"
```

---

### Task 5: Rosetta bridge `PolicyRunner` (`framework: tensorlist`)

**Files:**
- Create: `pai_policy_io/pai_policy_io/rosetta_bridge.py`, `pai_data_collection/config/rosetta/policy_runner_tensorlist.yaml`
- Modify: `pai_policy_io/setup.py` (entry point `rosetta.policy_runners: tensorlist = pai_policy_io.rosetta_bridge:TensorListBridgeRunner`)
- Test: `pai_policy_io/test/test_rosetta_bridge.py`

**Interfaces:**
- Consumes: `FrameIO` (`sample_frame()`, `warmed_up`, `publish_frame({"action": np.ndarray})`), `PolicyRunner` protocol (`setup(node, contract)`, `run(frames, *, task, stop_event) -> RunnerResult`, `feedback() -> RunnerFeedback`, `request_stop()`, `teardown()`), `NodeLike` (`declare_parameter/has_parameter/get_parameter/get_logger`, plus the real node's `create_publisher`, `create_subscription`, `create_timer`, `get_clock`, `destroy_*`). `contract_pipeline_hash` (Task 3), `frame_to_tensorlist`, `tensorlist_to_frame`, `ChunkFollower` (Task 4).
- Produces: `class TensorListBridgeRunner` with parameters (all read-only, declared with `has_parameter` guards):

  | Parameter | Default |
  | --- | --- |
  | `observation_topic` | `/policy/observation` |
  | `action_chunk_topic` | `/policy/action_chunk` |
  | `pipeline_info_topic` | `/policy/pipeline_info` |
  | `robot` | `so_arm101` |

  Behaviour: `setup()` builds `(signature, hash) = contract_pipeline_hash(contract_path, robot)`, creates publishers (observation: reliable, keep_last 1; pipeline info: reliable, transient_local, keep_last 1) and publishes one `PipelineInfo` (`signature_yaml = yaml.safe_dump(signature_to_dict(sig))`). `run()` resets a `ChunkFollower(len(sig.action.joints), 10**9 // fps)`, creates the chunk subscription (reliable, keep_last 1, own `MutuallyExclusiveCallbackGroup`) and a timer at `1/fps` (own `MutuallyExclusiveCallbackGroup`), then `stop_event.wait()`, destroys both, returns `RunnerResult(True, "Stopped")`. `_tick()`: skip until `frames.warmed_up`; `frame = frames.sample_frame()`; publish `frame_to_tensorlist(frame, now, casts={state keys: float32})` (drop keys not in `sig.observations`); `a = follower.action_at(now_ns)`; if `a is not None`: `frames.publish_frame({"action": a.astype(np.float64)})` and increment `published`. `_on_chunk(msg)`: `arrays = tensorlist_to_frame(msg)`; require `"action"` key; `follower.push(stamp_ns(msg.header), arrays["action"])`; log a throttled warning when `push` returns `False`. `feedback()` returns `RunnerFeedback(queue_depth=follower.remaining(now), published_actions=published)`. `request_stop()` is a no-op (the loop only waits on `stop_event`).
  - Decision: the timer-on-executor approach works because `policy_runner_node` spins a `MultiThreadedExecutor(num_threads=4)` (`rosetta/robots/ros2/nodes/node_utils.py`, `spin_lifecycle_node`), so `run()` may block while the timer and subscription callbacks keep firing.

- [ ] **Step 1: Write the failing tests** (no ROS graph: a `FakeNode` records declared parameters and created entities, exposes `fire_timer()` and `deliver(msg)`; a `FakeFrames` implements `FrameIO` with a fixed frame and a `published` list; `rclpy` message classes are importable without `rclpy.init`)

```python
def test_setup_publishes_pipeline_info_once():
    runner = TensorListBridgeRunner(); node = FakeNode(contract_path=CONTRACT); runner.setup(node, load_contract(str(CONTRACT)))
    info = node.published[runner.pipeline_info_topic]; assert len(info) == 1
    assert info[0].pipeline_hash == contract_pipeline_hash(CONTRACT, "so_arm101")[1]
    assert "observation.state" in info[0].signature_yaml

def test_tick_publishes_observation_in_signature_dtypes_and_follows_chunk():
    ... setup; start run() on a thread with stop_event; node.fire_timer() once
    obs = node.published["/policy/observation"][-1]
    assert {t.name for t in obs.tensors} == {"observation.images.wrist", "observation.images.static", "observation.state"}
    assert next(t for t in obs.tensors if t.name == "observation.state").dtype == Tensor.DTYPE_FLOAT32
    assert frames.published == []                                     # no chunk yet -> nothing published
    node.deliver(make_chunk(stamp=obs.header.stamp, horizon=4, value=0.5)); node.fire_timer()
    np.testing.assert_allclose(frames.published[-1]["action"], 0.5)
    assert runner.feedback().published_actions == 1
    stop_event.set(); thread.join(2); assert result.success

def test_tick_before_warmup_publishes_nothing(): ...
def test_chunk_with_wrong_dim_is_ignored(): ...   # feedback().queue_depth stays 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_rosetta_bridge.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `rosetta_bridge.py`, register the entry point, write `policy_runner_tensorlist.yaml`**

Params file (mirrors `policy_runner.yaml`'s header comment style):
```yaml
policy_runner:
  ros__parameters:
    framework: "tensorlist"
    observation_topic: "/policy/observation"
    action_chunk_topic: "/policy/action_chunk"
    pipeline_info_topic: "/policy/pipeline_info"
    robot: "so_arm101"
    default_max_duration_s: 0.0
```

- [ ] **Step 4: Run tests to verify they pass; verify the plugin resolves**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_rosetta_bridge.py -q` → 4 passed.
Run: `pixi run build --packages-select pai_policy_io pai_data_collection && pixi run python -c "from rosetta.policies import load_policy_runner; print(load_policy_runner('tensorlist'))"`
Expected: prints `<pai_policy_io.rosetta_bridge.TensorListBridgeRunner object ...>`.

- [ ] **Step 5: Lint and commit**

```bash
pixi run lint
git add pai_policy_io pai_data_collection/config/rosetta/policy_runner_tensorlist.yaml
git commit -m "feat(pai_policy_io): Rosetta PolicyRunner bridge publishing TensorList observations"
```

---

### Task 6: `pai_policy_inference` package: signature parsing and tensor views

**Files:**
- Create: `pai_policy_inference/package.xml`, `CMakeLists.txt`, `include/pai_policy_inference/signature.hpp`, `include/pai_policy_inference/tensor_view.hpp`, `src/signature.cpp`, `src/tensor_view.cpp`, `test/fixtures/policy_signature.yaml` (copy of the Task 2 fixture)
- Test: `test/test_signature.cpp`, `test/test_tensor_view.cpp`

**Interfaces:**
- Consumes: `pai_tensor_msgs::msg::Tensor` (Task 1).
- Produces (`namespace pai_policy_inference`):
  ```cpp
  // signature.hpp
  struct TensorSpec { std::string dtype; std::vector<int64_t> shape; };   // vectors: shape = {joints.size()}
  struct ActionSpec { std::string dtype; std::vector<std::string> joints; std::vector<std::string> kinds; };
  struct ExecutionSpec { int horizon; int execute_steps; std::string strategy; double max_staleness_s; };
  struct PolicySignature { int signature_version; std::string robot; int fps;
    std::map<std::string, TensorSpec> observations; ActionSpec action;
    std::optional<ExecutionSpec> execution; std::string pipeline_hash; };
  PolicySignature load_signature(const std::filesystem::path & yaml_path);   // throws std::runtime_error naming the missing/invalid key
  int64_t action_dim(const PolicySignature &);                               // joints.size()

  // tensor_view.hpp
  enum class DType : uint8_t { UINT8 = 0, INT32 = 1, INT64 = 2, FLOAT32 = 3, FLOAT64 = 4, STRING = 5 };
  DType dtype_from_string(const std::string &);     // "uint8" -> UINT8 ...; throws std::invalid_argument
  size_t itemsize(DType);                            // STRING -> 1
  int64_t element_count(const std::vector<int64_t> & shape);   // product, 1 for {}
  struct TensorView { std::string name; DType dtype; std::vector<int64_t> shape; const uint8_t * data; size_t nbytes; };
  struct OwnedTensor { std::string name; DType dtype; std::vector<int64_t> shape; std::vector<uint8_t> data; };
  TensorView view_of(const pai_tensor_msgs::msg::Tensor &);   // throws std::invalid_argument when data.size() != element_count*itemsize
  pai_tensor_msgs::msg::Tensor to_msg(const OwnedTensor &);
  ```
- `signature.cpp` reads only what the inference node needs (`observations.*.dtype/shape/joints`, `action.dtype/joints/kinds`, `execution`, `pipeline_hash`); `execution` is required (`load_signature` throws "execution section missing: not an exported model") — the node only ever loads exported models.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Signature, LoadsExample) {
  auto sig = load_signature(FIXTURE_DIR "/policy_signature.yaml");
  EXPECT_EQ(sig.fps, 50);
  EXPECT_EQ(sig.observations.at("observation.images.wrist").shape, (std::vector<int64_t>{480, 480, 3}));
  EXPECT_EQ(sig.observations.at("observation.state").shape, (std::vector<int64_t>{6}));
  EXPECT_EQ(action_dim(sig), 6);
  ASSERT_TRUE(sig.execution); EXPECT_EQ(sig.execution->horizon, 100); EXPECT_EQ(sig.execution->strategy, "blend");
  EXPECT_EQ(sig.pipeline_hash, "sha256:0000");
}
TEST(Signature, MissingExecutionThrows) { /* write fixture minus `execution` to a temp file */ EXPECT_THROW(load_signature(tmp), std::runtime_error); }

TEST(TensorView, ItemsizeAndCount) { EXPECT_EQ(itemsize(DType::FLOAT32), 4u); EXPECT_EQ(element_count({480, 480, 3}), 691200); EXPECT_EQ(element_count({}), 1); }
TEST(TensorView, ViewOfValidatesLength) { Tensor t; t.name = "x"; t.dtype = 3; t.shape = {6}; t.data.resize(24); EXPECT_EQ(view_of(t).nbytes, 24u);
  t.data.resize(23); EXPECT_THROW(view_of(t), std::invalid_argument); }
TEST(TensorView, ToMsgRoundTrip) { OwnedTensor o{"action", DType::FLOAT32, {2, 3}, std::vector<uint8_t>(24, 7)}; auto m = to_msg(o);
  EXPECT_EQ(m.name, "action"); EXPECT_EQ(m.dtype, 3); EXPECT_EQ(m.shape, (std::vector<int64_t>{2, 3})); EXPECT_EQ(m.data.size(), 24u); }
```

- [ ] **Step 2: Create the package and run the tests to verify they fail**

`package.xml`: `ament_cmake`, depends `rclcpp`, `rclcpp_components`, `pai_tensor_msgs`, `std_msgs`, `yaml-cpp`, `onnxruntime` (vendor; declare as `<depend>` only if a rosdep key exists, otherwise comment), `<test_depend>ament_cmake_gtest</test_depend>`.
`CMakeLists.txt`: C++17; `find_package(yaml-cpp REQUIRED)`; library `pai_policy_inference_core` (signature, tensor_view, ort_backend, inference_core); `ament_add_gtest` per test file with `target_compile_definitions(... FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures")`.
Run: `pixi run build --packages-select pai_policy_inference`
Expected: compile FAIL (headers missing).

- [ ] **Step 3: Implement `signature.cpp` and `tensor_view.cpp`**

- [ ] **Step 4: Build and run the tests**

Run: `pixi run build --packages-select pai_policy_inference && pixi run colcon test --packages-select pai_policy_inference && pixi run colcon test-result --verbose`
Expected: `test_signature`, `test_tensor_view` pass, 0 failures.

- [ ] **Step 5: Lint and commit**

```bash
pixi run lint
git add pai_policy_inference && git commit -m "feat(pai_policy_inference): signature loader and tensor views"
```

---

### Task 7: ONNX Runtime backend, `InferenceCore`, `LatestSlot`, `HashGate`

**Files:**
- Create: `include/pai_policy_inference/inference_backend.hpp`, `ort_backend.hpp`, `inference_core.hpp`, `src/ort_backend.cpp`, `src/inference_core.cpp`, `test/fixtures/make_fixture.py`, `test/fixtures/add_one.onnx`
- Modify: `pai_policy_inference/CMakeLists.txt` (link `onnxruntime`, add tests)
- Test: `test/test_inference_core.cpp`

**Interfaces:**
- Consumes: Task 6 types.
- Produces:
  ```cpp
  // inference_backend.hpp
  struct IoSpec { std::string name; DType dtype; std::vector<int64_t> shape; };   // shape entries -1 = dynamic
  class InferenceBackend { public: virtual ~InferenceBackend() = default;
    virtual std::vector<IoSpec> inputs() const = 0; virtual std::vector<IoSpec> outputs() const = 0;
    virtual std::vector<OwnedTensor> run(const std::vector<TensorView> & inputs) = 0; };   // inputs in inputs() order

  // ort_backend.hpp
  struct OrtOptions { std::string provider = "cpu"; int intra_op_threads = 0; };   // provider: "cpu" | "cuda"
  class OrtBackend : public InferenceBackend { public: OrtBackend(const std::filesystem::path & model, const OrtOptions &); ... };
  // Constructor throws std::runtime_error("execution provider 'cuda' is not available in this ONNX Runtime build") when asked for an absent provider.
  // run() wraps inputs with Ort::Value::CreateTensor(memory_info, const_cast<...>(data), nbytes, shape...) — no copy of image bytes.

  // inference_core.hpp
  class InferenceCore { public: InferenceCore(std::unique_ptr<InferenceBackend>, PolicySignature);
    // Startup validation (constructor throws std::runtime_error):
    //   every backend input name must be a signature observation with matching dtype and shape;
    //   backend must have exactly one output named "action", FLOAT32, shape {horizon, action_dim} (dynamic dims accepted).
    pai_tensor_msgs::msg::TensorList infer(const pai_tensor_msgs::msg::TensorList & obs);
    // Binds by name; throws std::invalid_argument("observation missing tensor 'x'" / "tensor 'x': expected float32[6], got uint8[6]").
    // Output: header = obs.header; one tensor "action"; verifies the backend output shape == {horizon, action_dim}.
  };
  template <typename T> class LatestSlot { public: void put(T); std::optional<T> take_blocking(std::chrono::milliseconds timeout); size_t dropped() const; };
  // put() replaces an untaken value and increments dropped(); take_blocking waits on a condvar.
  class HashGate { public: HashGate(std::string expected, bool enforce);
    // check(received) -> std::string message; returns "" when equal; when enforce && !equal throws std::runtime_error with both hashes;
    // when !enforce && !equal returns the loud message for the node to log at ERROR.
    std::string check(const std::string & received) const; };
  ```
- Fixture `add_one.onnx` (generated by `make_fixture.py` with `onnx.helper`, opset 17, committed, < 2 KB): inputs `observation.state` float32 `[6]`, `observation.images.wrist` uint8 `[4, 4, 3]`; output `action` float32 `[2, 6]` = `Expand(Add(state, 1), [2, 6])`; the image input feeds `Cast→ReduceSum→Mul 0` added to the output so the graph consumes it. Fixture signature (`test/fixtures/policy_signature.yaml` updated here): `observation.images.wrist` shape `[4, 4, 3]`, `observation.state` 6 joints, `execution.horizon: 2`.

- [ ] **Step 1: Generate the fixture**

Run: `pixi run python pai_policy_inference/test/fixtures/make_fixture.py` then `pixi run python -c "import onnx; onnx.checker.check_model('pai_policy_inference/test/fixtures/add_one.onnx')"`
Expected: file written, checker silent.

- [ ] **Step 2: Write the failing tests**

```cpp
static TensorList make_obs(float state_value) { /* stamp {12, 34}; state float32[6]=state_value; image uint8[4,4,3]=0 */ }

TEST(OrtBackend, ReportsIo) { OrtBackend b(FIXTURE "/add_one.onnx", {}); ASSERT_EQ(b.inputs().size(), 2u); EXPECT_EQ(b.outputs()[0].name, "action"); }
TEST(OrtBackend, UnavailableProviderThrows) { EXPECT_THROW(OrtBackend(FIXTURE "/add_one.onnx", {"nope", 0}), std::runtime_error); }
TEST(InferenceCore, RunsAndCopiesStamp) { InferenceCore core(std::make_unique<OrtBackend>(FIXTURE "/add_one.onnx", OrtOptions{}), load_signature(FIXTURE "/policy_signature.yaml"));
  auto out = core.infer(make_obs(1.5f)); EXPECT_EQ(out.header.stamp.sec, 12); EXPECT_EQ(out.header.stamp.nanosec, 34u);
  ASSERT_EQ(out.tensors.size(), 1u); EXPECT_EQ(out.tensors[0].name, "action"); EXPECT_EQ(out.tensors[0].shape, (std::vector<int64_t>{2, 6}));
  const float * a = reinterpret_cast<const float *>(out.tensors[0].data.data()); EXPECT_FLOAT_EQ(a[0], 2.5f); EXPECT_FLOAT_EQ(a[11], 2.5f); }
TEST(InferenceCore, MissingInputThrows) { /* obs without the image */ EXPECT_THROW(core.infer(obs), std::invalid_argument); }
TEST(InferenceCore, WrongDtypeThrows) { /* state as float64 */ EXPECT_THROW(core.infer(obs), std::invalid_argument); }
TEST(InferenceCore, SignatureMismatchAtConstruction) { /* signature with 5 joints */ EXPECT_THROW(InferenceCore(...), std::runtime_error); }
TEST(LatestSlot, KeepsOnlyNewest) { LatestSlot<int> s; s.put(1); s.put(2); EXPECT_EQ(s.take_blocking(10ms), 2); EXPECT_EQ(s.dropped(), 1u); EXPECT_FALSE(s.take_blocking(10ms)); }
TEST(HashGate, EnforcesAndOverrides) { EXPECT_EQ(HashGate("sha256:a", true).check("sha256:a"), ""); EXPECT_THROW(HashGate("sha256:a", true).check("sha256:b"), std::runtime_error);
  EXPECT_NE(HashGate("sha256:a", false).check("sha256:b"), ""); }
```

- [ ] **Step 3: Build to verify the tests fail**

Run: `pixi run build --packages-select pai_policy_inference`
Expected: compile FAIL (missing headers).

- [ ] **Step 4: Implement `ort_backend.cpp` and `inference_core.cpp`; link ORT in CMake**

CMake: `find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)` and `find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime REQUIRED)` (conda prefix is on `CMAKE_PREFIX_PATH`); link the core library against it and `yaml-cpp::yaml-cpp`. Map `DType` ↔ `ONNXTensorElementDataType` in `ort_backend.cpp`. Provider "cuda" → `OrtSessionOptionsAppendExecutionProvider_CUDA` guarded by `Ort::GetAvailableProviders()`.

- [ ] **Step 5: Build and run the tests**

Run: `pixi run build --packages-select pai_policy_inference && pixi run colcon test --packages-select pai_policy_inference && pixi run colcon test-result --verbose`
Expected: all gtest binaries pass.

- [ ] **Step 6: Lint and commit**

```bash
pixi run lint
git add pai_policy_inference && git commit -m "feat(pai_policy_inference): ONNX Runtime backend and robot-agnostic inference core"
```

---

### Task 8: Inference node (component + executable) and launch file

**Files:**
- Create: `include/pai_policy_inference/inference_node.hpp`, `src/inference_node.cpp`, `src/inference_node_main.cpp`, `pai_bringup/launch/policy_io_inference.launch.py`
- Modify: `pai_policy_inference/CMakeLists.txt` (component library + `rclcpp_components_register_node`, executable `inference_node`), `pai_bringup/package.xml` (`<exec_depend>pai_policy_inference</exec_depend>`, `pai_policy_io`, `rosetta`)

**Interfaces:**
- Consumes: `InferenceCore`, `OrtBackend`, `LatestSlot`, `HashGate`, `load_signature` (Tasks 6–7).
- Produces: `class InferenceNode : public rclcpp::Node` (component `pai_policy_inference::InferenceNode`, executable `inference_node`). Parameters:

  | Parameter | Default | Meaning |
  | --- | --- | --- |
  | `model_dir` | `""` (required) | Directory with `model.onnx` and `policy_signature.yaml` |
  | `backend` | `onnxruntime` | Only value accepted in this plan; others throw at startup |
  | `provider` | `cpu` | `cpu` or `cuda` |
  | `intra_op_threads` | `0` | ORT default |
  | `observation_topic` | `/policy/observation` | |
  | `action_chunk_topic` | `/policy/action_chunk` | |
  | `pipeline_info_topic` | `/policy/pipeline_info` | transient_local subscription |
  | `enforce_pipeline_hash` | `true` | `false` logs the mismatch at ERROR and continues |
  | `pipeline_info_timeout_s` | `10.0` | Fail startup if no `PipelineInfo` arrives in time |

  Startup order in the constructor: load signature → build `OrtBackend` → `InferenceCore` → subscribe to `PipelineInfo` (reliable, transient_local, keep_last 1) and wait up to the timeout (a `rclcpp::WaitSet` on that subscription, so no executor is needed) → `HashGate(sig.pipeline_hash, enforce).check(received)` → log `RCLCPP_INFO` "model <dir>: horizon=H action_dim=D hash OK" → observation subscription (reliable, keep_last 1, pushes into `LatestSlot<TensorList::ConstSharedPtr>`) → publisher (reliable, keep_last 1) → worker `std::thread` loop: `take_blocking(100ms)` → `core.infer` → publish; `std::invalid_argument` from `infer` is logged with `RCLCPP_WARN_THROTTLE` (1 s) and the observation is skipped; `LatestSlot::dropped()` is logged at DEBUG every 100 drops. Destructor stops the worker.
  Startup errors throw `std::runtime_error`; `inference_node_main.cpp` catches, logs at FATAL, returns 1.
- Launch file `policy_io_inference.launch.py` arguments: `model_dir` (required), `use_sim_time` (`false`), `provider` (`cpu`), `contract_path` (default `$(pai_data_collection share)/config/rosetta/so_arm101_policy_io.yaml`), `enforce_pipeline_hash` (`true`). Includes Rosetta's `policy_runner_launch.py` with `params_file:=.../policy_runner_tensorlist.yaml`, `contract_path`, `use_sim_time`, and starts `inference_node` with the parameters above.

- [ ] **Step 1: Implement the node, main, CMake registration and launch file**

- [ ] **Step 2: Build; verify startup refusal without a pipeline**

Run (Zenoh router running in another terminal):
`pixi run ros2 run pai_policy_inference inference_node --ros-args -p model_dir:=$PWD/pai_policy_inference/test/fixtures -p pipeline_info_timeout_s:=2.0`
Expected: exits with code 1 within ~2 s; FATAL log contains `no PipelineInfo received on /policy/pipeline_info within 2.0 s`.

- [ ] **Step 3: Verify hash enforcement and the override**

Run, with the node from Step 2 restarted with `pipeline_info_timeout_s:=30.0`, in a second terminal:
`pixi run ros2 topic pub --qos-durability transient_local --qos-reliability reliable -1 /policy/pipeline_info pai_tensor_msgs/msg/PipelineInfo "{pipeline_hash: 'sha256:ffff'}"`
Expected: node exits 1 with `pipeline hash mismatch: model sha256:0000, pipeline sha256:ffff`. Repeat with `-p enforce_pipeline_hash:=false`: node logs the same text at ERROR and stays up.

- [ ] **Step 4: Verify inference on the fixture through topics**

With the node up (hash `sha256:0000` published), run `pixi run ros2 topic echo /policy/action_chunk` and publish one observation from Python:
```bash
pixi run python - <<'EOF'
import rclpy, numpy as np
from rclpy.node import Node
from pai_policy_io.tensor_msgs import frame_to_tensorlist
from pai_tensor_msgs.msg import TensorList
rclpy.init(); n = Node("obs_pub"); p = n.create_publisher(TensorList, "/policy/observation", 1)
msg = frame_to_tensorlist({"observation.state": np.full(6, 1.5, np.float32), "observation.images.wrist": np.zeros((4, 4, 3), np.uint8)}, n.get_clock().now().to_msg(), casts={})
import time; time.sleep(1.0); p.publish(msg); time.sleep(0.5); rclpy.shutdown()
EOF
```
Expected: one `TensorList` on `/policy/action_chunk` with `tensors[0].name: action`, `shape: [2, 6]`, 48 bytes of data, and the same `header.stamp` as the published observation.

- [ ] **Step 5: Lint and commit**

```bash
pixi run lint
git add pai_policy_inference pai_bringup
git commit -m "feat(pai_policy_inference): inference node with hash gate and latest-only scheduling; launch file"
```

---

### Task 9: ACT → ONNX exporter with baked normalisation, legacy units and equivalence check

**Files:**
- Create: `pai_policy_io/pai_policy_io/export/__init__.py`, `pai_policy_io/pai_policy_io/export/lerobot_act.py`, `pai_policy_io/pai_policy_io/export/cli.py`
- Modify: `pai_policy_io/setup.py` (console script `pai_export_act = pai_policy_io.export.cli:main`)
- Test: `pai_policy_io/test/test_export_lerobot_act.py`

**Interfaces:**
- Consumes: `PolicySignature`, `ExecutionSpec`, `dump_signature`, `dataclasses.replace` (Task 2); `contract_pipeline_hash` (Task 3). LeRobot 0.6.0: `ACTPolicy.from_pretrained(path)`, `policy.config.chunk_size`, `policy.config.image_features` (dict key → feature), `policy.config.normalization_mapping` (`FeatureType` → `NormalizationMode`), `make_pre_post_processors(policy.config, pretrained_path=path)` → `(preprocessor, postprocessor)`; the `NormalizerProcessorStep` inside `preprocessor.steps` exposes `state_dict()` with keys `"<feature>.<stat>"` (`mean`, `std`, `min`, `max`); `policy.predict_action_chunk(batch) -> [1, chunk, dim]`; `lerobot.utils.constants.OBS_STATE`, `OBS_IMAGES`, `ACTION`.
- Produces (`lerobot_act.py`):
  ```python
  @dataclass(frozen=True)
  class ExportOptions:
      legacy_units: str | None = None      # "deg": state rad->deg before the model, action deg->rad after
      execute_steps: int = 30
      strategy: str = "blend"
      max_staleness_s: float = 0.2
      opset: int = 17
      check_frames: int = 8               # equivalence frames; 0 disables (export still refused unless --unsafe-skip-check)
      atol: float = 1e-3; rtol: float = 1e-3
      dataset: str | None = None          # LeRobot repo id or local root for check frames; None -> random frames within stats

  class ActOnnxWrapper(torch.nn.Module):
      def __init__(self, policy: ACTPolicy, stats: dict[str, dict[str, torch.Tensor]], image_keys: list[str], state_key: str, legacy_units: str | None): ...
      def forward(self, *inputs: torch.Tensor) -> torch.Tensor
      # inputs in signature order: images uint8 [H,W,3] each, then state float32 [D]. Returns action float32 [horizon, D].
      # images: float()/255 -> permute(2,0,1) -> unsqueeze(0) -> normalise per VISUAL mode (stats reshaped [C,1,1])
      # state: unsqueeze(0) -> (x*180/pi if legacy) -> normalise per STATE mode -> model -> unnormalise ACTION -> (x*pi/180 if legacy) -> squeeze(0)
      # MEAN_STD: (x-mean)/(std+1e-8), inverse x*std+mean. MIN_MAX: 2*(x-min)/where(max-min==0,1e-8,max-min)-1, inverse (x+1)/2*denom+min. IDENTITY passes through. Other modes raise NotImplementedError.

  def extract_stats(preprocessor) -> dict[str, dict[str, torch.Tensor]]      # from the NormalizerProcessorStep.state_dict()
  def build_wrapper(pretrained: str, signature: PolicySignature, opts: ExportOptions) -> tuple[ActOnnxWrapper, ACTPolicy, preprocessor, postprocessor]
  def export_onnx(wrapper: ActOnnxWrapper, signature: PolicySignature, out_path: Path, opset: int) -> None
      # torch.onnx.export(wrapper, example_inputs, out_path, input_names=[image keys..., "observation.state"], output_names=["action"], opset_version=opset, dynamo=False, do_constant_folding=True); then onnx.checker.check_model
  def reference_chunk(policy, preprocessor, postprocessor, frame: dict[str, np.ndarray], legacy_units) -> np.ndarray   # LeRobot path exactly as lerobot_rosetta/policy_server.py does it, output converted to SI
  def check_equivalence(onnx_path: Path, policy, preprocessor, postprocessor, frames: list[dict[str, np.ndarray]], opts: ExportOptions) -> float   # returns max abs error; raises ExportEquivalenceError beyond tolerance
  def export_act(pretrained: str, signature: PolicySignature, pipeline_hash: str, out_dir: Path, opts: ExportOptions) -> Path
      # writes out_dir/model.onnx and out_dir/policy_signature.yaml with execution=ExecutionSpec(policy.config.chunk_size, opts.execute_steps, opts.strategy, opts.max_staleness_s) and pipeline_hash; refuses (raises) if signature.action.joints length != policy action dim or image shapes != policy image feature shapes (C,H,W → H,W,C)
  class ExportEquivalenceError(RuntimeError)
  ```
- CLI `pai_export_act --pretrained <hf id|path> --contract <yaml> --robot so_arm101 --out <dir> [--legacy-units deg] [--dataset <repo id|path>] [--check-frames 8] [--execute-steps 30] [--strategy blend] [--max-staleness-s 0.2] [--provider cpu]`. Prints the max abs error and the pipeline hash on success; exits 2 on `ExportEquivalenceError`.
- Dataset check frames: `LeRobotDataset(repo_id, root=...)` items (`observation.images.*` float CHW in [0,1] → `(x*255).round().to(uint8).permute(1,2,0)`; `observation.state` degrees → radians when `legacy_units == "deg"`). Frame indices: `np.linspace(0, len(ds)-1, check_frames).astype(int)`.

- [ ] **Step 1: Write the failing tests** (unit tests use a tiny randomly-initialised ACT: `ACTConfig(input_features={"observation.state": PolicyFeature(STATE,(6,)), "observation.images.wrist": PolicyFeature(VISUAL,(3,8,8))}, output_features={"action": PolicyFeature(ACTION,(6,))}, chunk_size=4, n_action_steps=4, dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1, vision_backbone="resnet18", pretrained_backbone_weights=None, use_vae=False)`, with synthetic stats; mark the module `pytest.importorskip("torch")`, `pytest.importorskip("lerobot")`)

```python
def test_wrapper_matches_lerobot_reference_on_random_frames(tiny_policy, tmp_path):
    wrapper, policy, pre, post = build_wrapper_from_policy(tiny_policy, stats, signature, ExportOptions(legacy_units=None))
    frames = [random_frame() for _ in range(3)]
    export_onnx(wrapper, signature, tmp_path / "model.onnx", opset=17)
    err = check_equivalence(tmp_path / "model.onnx", policy, pre, post, frames, ExportOptions())
    assert err < 1e-4

def test_legacy_deg_wrapper_takes_radians_and_returns_radians(tiny_policy):
    # with stats in degrees: feeding state_rad to the legacy wrapper == feeding deg to the plain wrapper, output /(180/pi)
    ...
def test_export_act_writes_artifact_with_execution_and_hash(tiny_policy, tmp_path):
    out = export_act_from_policy(...)   # same body as export_act but takes an in-memory policy (keep export_act a thin loader around it)
    sig = load_signature(out / "policy_signature.yaml")
    assert sig.execution == ExecutionSpec(4, 2, "blend", 0.2) and sig.pipeline_hash == "sha256:abc"
    onnx.checker.check_model(str(out / "model.onnx"))
    names = [i.name for i in onnx.load(out / "model.onnx").graph.input]
    assert names == ["observation.images.wrist", "observation.state"]
def test_export_refuses_action_dim_mismatch(tiny_policy, tmp_path): ...   # signature with 5 joints -> SignatureError
def test_equivalence_failure_raises(tmp_path): ...   # monkeypatch ORT output +1 -> ExportEquivalenceError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pai_policy_io && pixi run python -m pytest test/test_export_lerobot_act.py -q`
Expected: FAIL with `ImportError` (or skipped if torch is absent: then run `pixi run install-ml-deps` first; the exporter tests require torch).

- [ ] **Step 3: Implement `lerobot_act.py`, `cli.py`, register the console script**

- [ ] **Step 4: Run tests to verify they pass**

Run: same command. Expected: 5 passed.

- [ ] **Step 5: Export the real pretrained checkpoint and verify equivalence on recorded frames**

Run:
```bash
pixi run pai_export_act --pretrained francocipollone/rospai_act_sim_arm101_place_cubes_on_tray \
  --contract $(pixi run ros2 pkg prefix pai_data_collection)/share/pai_data_collection/config/rosetta/so_arm101_policy_io.yaml \
  --robot so_arm101 --legacy-units deg \
  --dataset francocipollone/rospai_sim_arm101_place_cubes_on_tray --check-frames 8 \
  --out models/act_so101
```
Expected: prints `max abs error: <value> rad (atol 1e-3)` with value < 1e-3, the pipeline hash, and `models/act_so101/{model.onnx,policy_signature.yaml}` exist; `policy_signature.yaml` has `execution.horizon: 100` (the checkpoint's `chunk_size`; record the actual value in the commit message if different).

- [ ] **Step 6: Add `models/` to `.gitignore`, lint and commit**

```bash
pixi run lint
git add .gitignore pai_policy_io && git commit -m "feat(pai_policy_io): ACT to ONNX exporter with baked normalisation and legacy unit wrapping"
```

---

### Task 10: End-to-end run, docs and pixi task

**Files:**
- Create: `docs/demos/onnx-inference.md`
- Modify: `pixi.toml` (`[tasks]`: `so-arm-onnx-inference`), `docs/README.md` and `README.md` (one row each pointing at the new guide), `README.md` packages table (three new packages)

**Interfaces:**
- Consumes: everything above. The pixi task: `so-arm-onnx-inference = { cmd = ["ros2", "launch", "pai_bringup", "policy_io_inference.launch.py", "model_dir:=${MODEL_DIR:-models/act_so101}", "use_sim_time:=true"] }` (verify pixi expands `${VAR:-default}` in `cmd` arrays; if not, drop the default and document `MODEL_DIR`).

- [ ] **Step 1: Run the pretrained checkpoint through the new path in Gazebo**

Terminals: `pixi run zenoh-router`; `pixi run so-arm-gz`; `pixi run so-arm-onnx-inference` (after Task 9 Step 5 produced `models/act_so101`); then
`pixi run ros2 action send_goal /run_policy rosetta_interfaces/action/RunPolicy "{prompt: 'place cubes on tray'}" --feedback`.
Expected: inference node logs `hash OK`; `ros2 topic hz /policy/observation` ≈ 50 Hz; `ros2 topic hz /policy/action_chunk` > 1 Hz on CPU; feedback `published_actions` increases; the arm performs the task comparably to the Rosetta + LeRobot path (`docs/demos/pretrained-demo.md`). Record observed `/policy/action_chunk` rate and CPU/GPU used in the doc.

- [ ] **Step 2: Verify the hash gate end to end**

Edit a QoS depth in a copy of `so_arm101_policy_io.yaml`, launch with `contract_path:=<copy>`.
Expected: inference node exits with `pipeline hash mismatch`. Relaunch with `enforce_pipeline_hash:=false`: it runs and the ERROR line is visible.

- [ ] **Step 3: Write `docs/demos/onnx-inference.md`**

Sections: what changed vs `pretrained-demo.md` (one paragraph + the §4 mermaid diagram from the design doc trimmed to phase 1: Rosetta bridge → `/policy/observation` → inference node → `/policy/action_chunk` → bridge → controller); Export (the Task 9 Step 5 command, flag table: `--legacy-units`, `--dataset`, `--check-frames`); Run (Step 1 commands); Parameters table for `inference_node`; Troubleshooting (hash mismatch, no PipelineInfo, provider unavailable, equivalence failure). Link the design doc and note phases 2–5 are upcoming.

- [ ] **Step 4: Update READMEs and pixi task; lint; commit**

```bash
pixi run lint
git add docs README.md pixi.toml
git commit -m "docs: ONNX inference guide and so-arm-onnx-inference task"
```

---

## Self-review notes

- **Spec coverage (phase 0+1):** TensorList package (T1), signature + hash (T2–T3), bridge publishing Rosetta frames as TensorList and consuming chunks (T4–T5), generic ORT inference node with name binding, latest-only scheduling, stamp copy, signature + hash startup checks (T6–T8), ACT exporter with baked normalisation, legacy `deg↔rad`, naming, equivalence test, artifact layout (T9), exit criterion "pretrained checkpoint drives the arm through the new node" (T10). Upstreaming `rmw_zenoh_cpp ≥ 0.10.5` to RoboStack (spec phase 0) is external work and not a task here. Not in scope by design: executor strategies/kinds/limits (phase 2), C++ pre-processing + resize convention + oracle tests (phase 3), recording/converter/replay host (phase 4), OpenVINO/TensorRT (phase 5; the `backend` parameter rejects them until then).
- **Type consistency:** `TensorView`/`OwnedTensor`/`DType` are defined in T6 and used verbatim in T7–T8; `ChunkFollower.push/action_at/remaining` from T4 are the only chunk API the bridge uses in T5; `contract_pipeline_hash` (T3) feeds both the bridge (T5) and the CLI (T9).
- **Proportion:** bodies appear only for the fixture graph, the normalisation formulas (which the ONNX graph must reproduce exactly) and the message definitions.
