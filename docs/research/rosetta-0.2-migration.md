# Rosetta 0.2.0 Migration

Bumping this repository from Rosetta `fb3860c` (the last commit of the 0.1.0 line) to the **0.2.0** package set, which restructured Rosetta into a framework-agnostic core plus pluggable robot and learning-framework adapters.

Every section of the 0.2.0 changelog marked **Breaking** touches this repository. Contracts written against 0.1.0 do not load, two of the four node/CLI entry points we document were renamed, a fourth package joined the set, and the new LeRobot adapter requires **LeRobot 0.6.x** — which is the real cost of the bump.

> **Status: plan of record.** The upstream analysis in §1–§3 is sourced from the pinned commits below and verified against this workspace's `.pixi` environment. §4 is the execution plan; §5 records what was actually done.

---

## Contents

- [1. What changed upstream](#1-what-changed-upstream)
- [2. Contract schema v2](#2-contract-schema-v2)
- [3. Compatibility assessment for this repository](#3-compatibility-assessment-for-this-repository)
- [4. Execution plan](#4-execution-plan)
- [5. Migration record](#5-migration-record)
- [6. Appendix: pinned commits and references](#6-appendix-pinned-commits-and-references)

---

## 1. What changed upstream

### 1.1 The restructure, and why it happened

Rosetta 0.2.0 split one package into three layers:

| Layer             | Modules                                                  | May import                                                                   |
| ----------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Core              | `rosetta.contract`, `rosetta.frames`, `rosetta.policies` | Nothing robot- or ML-specific. No `rclpy`, no `*_msgs`, no `torch`/`lerobot` |
| Robot adapter     | `rosetta.robots.ros2.*`                                  | ROS 2                                                                        |
| Framework adapter | **`lerobot_rosetta`** (separate package)                 | LeRobot                                                                      |

Two consequences drive most of the migration work:

**The contract declares its robot interface.** `robot_interface: ros2` is a new required top-level field. `ros2` is the only value today, but the schema no longer presumes it: alignment timelines are _open names that the interface attests to_ (`receive` for every ROS 2 channel, `header` additionally for message types carrying a `std_msgs/Header`) rather than a fixed enum. Naming a timeline a channel does not provide is a load-time error.

**Learning frameworks resolve by entry point, not by import.** `rosetta.dataset_writers` and `rosetta.policy_runners` are the groups; `lerobot_rosetta` registers under the name `lerobot` in both. `rosetta_port --framework` and `policy_runner_node`'s `framework` parameter select one by name. Codecs and operators became open registries too (`rosetta.codecs`, `rosetta.operators`), so supporting a new message type is a third-party entry point rather than a fork.

The practical payoff for us is narrow but real: `rosetta` core is now importable and testable without ROS 2 or torch on the path, and the LeRobot pin lives in one package we could swap.

### 1.2 The package set grew to five

| Package                        | Role                                                                                 | Status here                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `rosetta`                      | Core library, ROS 2 nodes, bag conversion                                            | pinned — bump. **Build type changed `ament_cmake` → `ament_python`** |
| `rosetta_interfaces`           | Actions and services                                                                 | pinned — bump                                                        |
| `lerobot_robot_rosetta`        | LeRobot `Robot` plugin (`--robot.type=rosetta`)                                      | pinned — bump                                                        |
| **`lerobot_rosetta`**          | **LeRobot framework adapter: dataset writer, policy runner, gRPC inference servers** | **NEW — must be added**                                              |
| `lerobot_teleoperator_rosetta` | LeRobot `Teleoperator` plugin, experimental                                          | skip (HIL only)                                                      |

`lerobot_rosetta` is not optional. It holds everything that used to live inside `rosetta` on the LeRobot side, so without it `rosetta_port` has no dataset writer and `policy_runner_node` has no policy runner — both fail at resolution time with no backend registered.

Naming, since it is easy to confuse: `lerobot_rosetta` is _Rosetta loading LeRobot_ (Rosetta resolves it by entry point). `lerobot_robot_rosetta` is _LeRobot loading Rosetta_ (LeRobot auto-discovers it by the `lerobot_robot_*` name prefix). They point in opposite directions and both are needed.

### 1.3 Renamed entry points

| 0.1.0 — what this repo documents today | 0.2.0                         |
| -------------------------------------- | ----------------------------- |
| `python -m rosetta.port_bags`          | `rosetta_port` console script |
| `rosetta_client_launch.py`             | `policy_runner_launch.py`     |
| `rosetta_client_node`                  | `policy_runner_node`          |
| `rosetta_hil_launch.py`                | `hil_launch.py`               |

`episode_recorder_launch.py` and `episode_keyboard_node` keep their names. `rosetta_port` keeps `--raw-dir`, `--contract`, `--repo-id`, `--root`, and adds `--framework`, `--streaming-encoding`, `--num-shards`/`--shard-index`.

### 1.4 Behavioural changes that affect our demos

**Record raw, decode late.** Episodes are now recorded as raw rosbag2 messages with the exact contract text embedded in the bag's `metadata.yaml`. All decoding, resizing, alignment, and unit conversion moved to port or inference time. The payoff: **re-porting with a revised contract no longer requires re-recording.** Conversion reads the contract you pass it and treats the embedded copy as provenance, warning on mismatch.

**`record_all: true` is the default.** The recorder captures every topic on the graph, like `ros2 bag record -a`, with `exclude_topics`/`include_topics` regex lists for the exceptions. Contract topics become a _manifest_ — required to be present, never a filter. Only `/rosout` and `/parameter_events` are dropped automatically. Our sim bags will get materially larger, since the whole `ros_gz_bridge` graph now lands in them.

**One `image_transport` stream per camera.** Contract-declared streams win as-is; otherwise the recorder picks the best advertised stream by preference `compressed > zstd > theora > compressedDepth > raw`. The reason is load: `image_transport` republishers encode only while subscribed, so recording all of them makes one camera node encode every frame several ways at once. Our sim path (`ros_gz_bridge`, MuJoCo) publishes raw only, so declaring raw stays correct there.

**Warmup, then zero-fill.** No frames are emitted until every observation stream has produced at least one sample. After warmup, a stream with no sample at a tick zero-fills at its static dim, so every frame has the declared shape. The offline porter and the live bridge share this code path, gated by an upstream `test_bag_live_parity.py`, so a gap looks identical in training data and at inference.

**The contract travels.** Ported datasets carry a `meta/rosetta_contract.yaml` sidecar. Leave `policy_runner_node`'s `contract_path` empty and it resolves the contract from the checkpoint chain: `pretrained_name_or_path` → `train_config.json` → training dataset → sidecar. A non-empty `contract_path` is used as given and never compared against the checkpoint's own.

**Sim-clock pacing replaced `sim_time_multiplier`.** With `use_sim_time` true, the LeRobot adapter paces `get_observation`/`send_action` on the ROS sim clock at the contract `fps`. Directly relevant to our MuJoCo and Gazebo recording tasks, which currently pass no `use_sim_time` to the recorder.

**Action results were redesigned** (in `rosetta_interfaces`). `bool success` is gone from all three action results; the terminal `GoalStatus` is the mechanics and a new `string termination_reason` names the cause, with legal values declared as constants in each `.action`. Cancelling an untimed recording — the only way to end one — is now `CANCELED` rather than a reported failure. `max_duration_s` moved onto the goal. `ManageEpisode` gained `outcome` (did the robot do the task) as a signal independent of how the episode ended. All three nodes also gained `~/start_*` and `~/cancel_*` services for clients that can call services but not actions.

**All nodes are lifecycle nodes**, with `configure` creating subscriptions and disabled publishers, `activate` enabling publishers and opening goal acceptance, and `deactivate`/`cleanup` publishing the declared safety action. Goals are accepted only while active, one at a time.

---

## 2. Contract schema v2

Sections changed from **lists of entries carrying a `key:` field** to **mappings keyed by frame key**. Within an entry:

| 0.1.0                                                      | 0.2.0                                                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `topic:`, `type:`, `qos:` at entry level                   | all inside `channel: {topic, type, qos, dtype, safety, decoder, encoder}`                                   |
| `selector: {names: [...]}`                                 | `select: [...]` — a bare list                                                                               |
| `align: {strategy, stamp, tol_ms}`                         | `align: {strategy, timeline}`; `stamp` → `timeline`, `tol_ms` → `tolerance_ms` (only meaningful for `asof`) |
| `unit_conversion: rad2deg`                                 | `apply: [rad2deg]`                                                                                          |
| `image: {resize: [h, w]}`                                  | `apply: [resize: [h, w]]`                                                                                   |
| `from_tensor: {clamp: [lo, hi]}`                           | `apply: [clamp: {min: lo, max: hi}]`                                                                        |
| `safety_behavior: hold`                                    | `channel.safety: hold`                                                                                      |
| `publish: {topic, type, layout, qos, strategy}` on actions | gone — an action uses the same `channel`, read right-to-left                                                |
| `max_duration_s:` in the contract                          | ROS parameter `default_max_duration_s`, default no limit                                                    |
| `recording: {storage: mcap}`                               | launch argument `storage_id`                                                                                |
| —                                                          | `robot_interface: ros2`, **required**                                                                       |

### 2.1 `apply` is directional, and the order matters

On the record path, operators run front-to-back through their forward direction. On the serve path they run **back-to-front through their inverse**. Operators carry invertibility tiers:

| Operator  | Form                        | Tier                                               |
| --------- | --------------------------- | -------------------------------------------------- |
| `rad2deg` | `rad2deg`                   | `BIJECTIVE` — round-trip verified at contract load |
| `clamp`   | `clamp: {min: lo, max: hi}` | `BIDIRECTIONAL`                                    |
| `resize`  | `resize: [h, w]`            | `FORWARD_ONLY` — image observations only           |

**An action's `apply` accepts only `BIDIRECTIONAL` or `BIJECTIVE` operators**, refused at contract load rather than silently corrupting actions at deployment.

This makes operator order load-bearing for our contract. The old `from_tensor.clamp` meant "clamp the outgoing command, in radians". To reproduce that, `clamp` must come **before** `rad2deg`:

```yaml
apply: [clamp: { min: -3.14159, max: 3.14159 }, rad2deg]
```

Serve runs that backwards through the inverses — `deg2rad`, then `clamp` — so the clamp lands in radians immediately before publish. Writing `[rad2deg, clamp]` instead would clamp _degrees_ to ±3.14 and destroy every command.

### 2.2 Load-time validation is now total

`load_contract` returning means the contract is fully valid in this environment, enforced in three layers: shape parsing (unknown keys, missing `align`, duplicate YAML keys, non-importable codec paths), interface attestation (does this channel provide the timeline you named, does the message type exist), and registry resolution (decodability, encoder registration, image geometry, operator invertibility, feature-name uniqueness). Nothing downstream re-validates. A single `load_contract` call is therefore a complete contract test.

### 2.3 The LeRobot key-count limit

The porter writes one dataset feature per contract key, but LeRobot's live path collapses all numeric observations into one hardcoded `observation.state` and all actions into one hardcoded `action`. A contract with more keys than that ports and trains fine and then fails to deploy, so `lerobot_robot_rosetta` refuses it at `connect()` and at `policy_runner_node`'s configure transition: **at most one numeric observation key and at most one action key**, images exempt in any number. `load_contract` does not enforce this, since the contract layer stays backend-neutral.

Our contract has one numeric observation key, two image keys, and one action key, so it complies.

---

## 3. Compatibility assessment for this repository

### 3.1 The blocker: LeRobot 0.6.x is mandatory

The new `lerobot_rosetta` imports APIs that **do not exist** in the LeRobot 0.5.1 currently installed in `.pixi/envs/default`:

| Import                                                          | In 0.5.1?        | Needed for                        |
| --------------------------------------------------------------- | ---------------- | --------------------------------- |
| `lerobot.configs.RGBEncoderConfig`                              | **missing**      | `dataset_writer` → `rosetta_port` |
| `lerobot.policies.get_policy_class`, `make_pre_post_processors` | **not exported** | `policy_server` → local inference |
| `lerobot.rewards.factory`                                       | **missing**      | `classifier_server` (HIL only)    |
| `lerobot.async_inference.*`                                     | present          | policy runner                     |
| `lerobot.utils.device_utils.is_torch_device_available`          | present          | device resolution                 |

So both porting and deployment break on 0.5.1. Upstream `rosetta_ws` pins **`lerobot v0.6.0`**; PyPI also has 0.6.1. Both ship as `py3-none-any` wheels with `requires_python >=3.12`, so this workspace's Python 3.14 is not excluded a priori — but the transitive pins are a fresh solve, and `pixi.toml` carries a stack of workarounds that all need revalidation against 0.6.0's metadata:

- `numpy >=2.3,<3` dependency-override (no py314 build below 2.3; LeRobot declared `<2.3`)
- `protobuf >=6.31.1,<7` override (`gz-msgs` pins protobuf into the conda solve)
- `scipy >=1.11,<2` (LeRobot 0.5.x `pi0_fast` imports the removed `scipy.fftpack.idct`)
- `packaging >=24.2,<26` (LeRobot packaging constraint)
- the `draccus` git pin for PEP-604 unions on Python 3.14

**This is the single largest risk in the bump, and it is a pixi-solve risk rather than a Rosetta risk.**

Extras also change. Rosetta's installation guide asks for `lerobot[dataset,async]`, and 0.6.0 moved `datasets`/`pandas`/`pyarrow`/`torchcodec` out of core into the `dataset` extra, so `rosetta_port`'s dataset writer needs it. `feetech` stays: `docs/hardware/calibration.md` and `pai_data_collection/README.md` both document `lerobot-calibrate`/`lerobot-replay --robot.type=so101_follower`, which need the servo SDK. Final set: `["dataset", "feetech", "async"]`.

### 3.2 What is already fine

- **ROS 2 distro.** `rosetta` 0.2.0 is pure `ament_python` and uses only stock `rosbag2_py` API (`SequentialReader`, `StorageOptions`, `ConverterOptions`, `get_metadata`, `read_next`). Nothing Jazzy-specific, so Lyrical Luth should work despite upstream targeting Jazzy.
- **ROS dependencies.** Every `exec_depend` in the new `package.xml` is already in the environment: `tf2_msgs`, `rosgraph_msgs`, `control_msgs`, `nav_msgs`, `trajectory_msgs`, `std_srvs`, `rosidl_runtime_py`, `rcl_interfaces`, `lifecycle_msgs`, `rosbag2_storage_mcap`, `diagnostic_msgs`. No pixi additions needed on the ROS side.
- **OpenCV.** `cv2` is present, and is now a hard requirement — compressed-image decoding has no fallback decoder.
- **`ament_python` with `--symlink-install`.** `colcon_defaults.yaml` sets `symlink-install: true`, which for an `ament_python` package runs `setup.py develop`. The environment has setuptools **79.0.1**, i.e. below the 80.0 cutoff where `develop` was removed, so this works — but it is a latent trap worth a comment, since a future setuptools bump breaks the build with a confusing error.
- **Message types.** `sensor_msgs/msg/Image`, `sensor_msgs/msg/JointState`, and `std_msgs/msg/Float64MultiArray` are all still built-in codecs.

### 3.3 Two traps specific to our contract

**`Float64MultiArray` provides no `header` timeline.** Our action channel is `/forward_position_controller/commands`, a `std_msgs/msg/Float64MultiArray`, which carries no `std_msgs/Header`. It therefore provides only the `receive` timeline, and `align: {timeline: header}` on it is a load-time error. It must be `receive`. This replaces the old `publish.strategy: {mode: nearest, tolerance_ms: 100}`, which has no direct equivalent.

**`select` names on a MultiArray are arbitrary labels.** The `Float64MultiArray` decoder ignores `spec.names` entirely; `select` exists to declare the vector width and to name the dataset features. Keeping the `*.pos` names for `lerobot-replay`/`so101_follower` interop is still valid and costs nothing. (This does _not_ hold for `JointState`, where `select` paths are real field lookups — `position.shoulder_pan_joint` resolves to `msg.position[msg.name.index("shoulder_pan_joint")]`.)

---

## 4. Execution plan

Steps 1 and 2 land together: the intermediate state does not build, so splitting the repos bump from the LeRobot bump buys nothing.

### Step 1 — `pai.repos`

Bump three pins to `origin/main` as of 2026-08-01 and add `lerobot-rosetta`. See [§6](#6-appendix-pinned-commits-and-references) for the SHAs. Upstream's own `rosetta_ws/repos/src.repos` tracks `main` for all five packages, so these four HEADs are a coherent, mutually tested set.

### Step 2 — `pixi.toml` dependencies

- `lerobot` `==0.5.1` → `==0.6.0`, extras `["dataset", "async"]`.
- Re-solve, then revalidate every workaround listed in [§3.1](#31-the-blocker-lerobot-06x-is-mandatory); some may be obsolete and some may need new bounds.
- Nothing to add on the ROS side.

**Gate: `pixi install` and `pixi run build` must both pass before touching the contract.** If 0.6.0 will not solve on Python 3.14, the bump stalls here and the fallback (0.6.1, or dropping to Python 3.13) is a separate decision.

### Step 3 — Rewrite the contract

`pai_data_collection/config/rosetta/so_arm101.yaml`, to schema v2, applying the two traps from [§3.3](#33-two-traps-specific-to-our-contract) and the operator ordering from [§2.1](#21-apply-is-directional-and-the-order-matters). Dropped from the file: `max_duration_s`, `recording.storage`, `tol_ms` on `hold` entries, and `layout: flat`.

### Step 4 — `pixi.toml` tasks

- `rosetta-record-mujoco`: add `use_sim_time:=true` so sim-clock pacing engages.
- Add a task wrapping the `rosetta_port` console script, replacing the documented `python -m rosetta.port_bags`.
- Consider a deploy task for `policy_runner_launch.py`.
- Consider trimming `record_all` for the sim demos via `exclude_topics`, or accept larger bags.

### Step 5 — Documentation

| File                                | Work                                                                                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `docs/demos/end-to-end-pipeline.md` | Heaviest: the launch/CLI renames, the new contract, `record_all`, the action-result changes, contract-from-checkpoint deployment |
| `pai_data_collection/README.md`     | Same renames plus the new contract snippet                                                                                       |
| `README.md`                         | Add `lerobot_rosetta` to the external-packages table                                                                             |

### Step 6 — Verification

1. `pixi run build` clean.
2. `load_contract` on the migrated contract — per [§2.2](#22-load-time-validation-is-now-total), this alone proves codecs, timelines, operator invertibility, and image geometry.
3. Record a short MuJoCo episode; confirm the bag and the embedded contract in `metadata.yaml`.
4. `rosetta_port` that bag; confirm the `meta/rosetta_contract.yaml` sidecar.
5. Deploy an existing checkpoint through `policy_runner_launch.py`.

Step 5 is the one that may not pass. Datasets ported under the old schema have no contract sidecar, and their feature statistics came from a different pipeline. Existing checkpoints should still load with an explicit `contract_path`, but the datasets may be worth re-porting — which the new record-raw design makes cheap, though only for bags recorded after the bump.

---

## 5. Migration record

| Step                 | Status      | Notes                                                                                                                                              |
| -------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. `pai.repos`       | **done**    | Three pins bumped, `lerobot-rosetta` added                                                                                                         |
| 2. `pixi.toml` deps  | **done**    | `lerobot==0.6.0` with `["dataset", "feetech", "async"]`. Solve succeeded on Python 3.14; all five previously-missing adapter APIs verified present |
| 3. Contract v2       | **done**    | Rewritten and validated through `load_contract`; operator direction verified numerically                                                           |
| 4. `pixi.toml` tasks | **done**    | `use_sim_time:=true` on `rosetta-record-mujoco`; new `rosetta-port` task                                                                           |
| 5. Documentation     | **done**    | Pipeline guide, pre-trained demo, data-collection README, both README package tables                                                               |
| 6. Verification      | **partial** | Build, contract load, entry-point discovery, and CLI all pass. Runtime steps (record → port → deploy) need a robot session                         |

### What the bump turned up

**Every previously-missing LeRobot API resolved on 0.6.0.** `pixi install` solved without touching any of the existing workarounds, and all of `RGBEncoderConfig`, `get_policy_class`, `make_pre_post_processors`, `rewards.factory`, and the async-inference surface import cleanly on Python 3.14. Every one of the existing overrides is still load-bearing: 0.6.0 still declares `numpy<2.3.0` (no py314 build below 2.3) and now pins `draccus==0.10.0` exactly, so the fork override is what makes the solve work at all.

**Operator ordering verified numerically, not just read off the docs.** With `apply: [clamp, rad2deg]`, the serve path maps 90° to 1.5708 rad and clamps an out-of-distribution 400° to 3.14159 rad — the clamp landing in radians, last. The record path maps π/2 rad to 90°, and the in-range round trip is the identity.

**`rosetta_port` is not on `PATH`.** Upstream's docs show a bare `rosetta_port`, but rosetta's `setup.cfg` sets `install_scripts=$base/lib/rosetta`, which is where `ros2 run` looks and not a `PATH` entry. It must be invoked as `ros2 run rosetta rosetta_port`. (`lerobot_rosetta` has no such directive, so _its_ `rosetta_policy_server` does land on `PATH` — the two packages differ.)

**`--framework` confirms the adapter wiring end to end.** `ros2 run rosetta rosetta_port --help` reports `Installed: lerobot`, which means the `rosetta.dataset_writers` entry point resolved through the newly added `lerobot_rosetta` package. That is the single best smoke test that the restructure is correctly installed.

**Pre-existing bug found: `policy_device:=xpu` never worked.** `policy_device` is not a launch argument in `policy_runner_launch.py` — and it was not one at our old pin either. `ros2 launch` silently drops undeclared arguments (verified: the node starts normally with `policy_device:=cuda` passed and ignored), so the device has always come from rosetta's `params/policy_runner.yaml` default of `cuda`. Every Intel XPU user following [the pre-trained demo](../demos/pretrained-demo.md) has been getting the `cpu` fallback. This is unrelated to the bump — `Enable xpu (#73)` changed `install_ml_deps.sh`, `pai.repos`, and the docs, but there was no launch argument for the docs to be describing.

Fixed by shipping [`pai_data_collection/config/rosetta/policy_runner.yaml`](../../pai_data_collection/config/rosetta/policy_runner.yaml) and passing it as `params_file:=`, which is the only mechanism that actually sets the device. 0.2.0 at least logs a warning on fallback now, where the old version did not.

### Still open

- **`scripts/install_ml_deps.sh` installs a torch below LeRobot 0.6.0's floor.** 0.6.0 declares `torch<2.12.0,>=2.7`, but the standard-NVIDIA branch of the script installs `torch==2.5.1+cu124` (the RTX 5090 branch installs 2.9.1, and the XPU/CPU branches 2.10.0, all fine). Picking the replacement CUDA wheel is a hardware-compatibility call that wants testing on an actual non-Blackwell NVIDIA box, so it is left as-is rather than guessed at. The `dataset` extra also wants `torchcodec>=0.3`, which only the RTX 5090 branch installs today.
- **Runtime verification.** Steps 3 to 5 of [§4 Step 6](#step-6--verification) need a robot session: record a MuJoCo episode, port it, and deploy a checkpoint.
- **Re-porting existing datasets.** Datasets ported under the old schema carry no contract sidecar. Existing checkpoints should still deploy with an explicit `contract_path`, but re-porting from bags is worth doing for anything trained further.

---

## 6. Appendix: pinned commits and references

### Target pins

| Package                 | Commit                                     | Date       |
| ----------------------- | ------------------------------------------ | ---------- |
| `rosetta`               | `dfd07e40e1a20825e3a3f37cc897663fb3759285` | 2026-08-01 |
| `rosetta_interfaces`    | `17d06b639960b8a4de0d1970b4c42d306951d567` | 2026-08-01 |
| `lerobot_robot_rosetta` | `cd9b4c140ee5b6b02f3e57f76dfeb1a737646d86` | 2026-08-01 |
| `lerobot_rosetta`       | `1ea49d620fb344360e4982b9d59031c09efd4564` | 2026-08-01 |

Superseding `rosetta` `fb3860c`, `rosetta_interfaces` `90e9791`, `lerobot_robot_rosetta` `e4ed91e`.

### References

- [Rosetta documentation](https://iblnkn.github.io/rosetta/) — the Sphinx site new in 0.2.0
- [Contract reference](https://iblnkn.github.io/rosetta/reference/contract.html) and [node reference](https://iblnkn.github.io/rosetta/reference/nodes.html)
- [Design explanation](https://iblnkn.github.io/rosetta/explanation/design.html) — "record everything, decide later", and the package division
- [`iblnkn/rosetta`](https://github.com/iblnkn/rosetta) `CHANGELOG.rst` — the authoritative breaking-change list
- [`iblnkn/lerobot-rosetta`](https://github.com/iblnkn/lerobot-rosetta) — the new LeRobot framework adapter
- [`iblnkn/rosetta_ws`](https://github.com/iblnkn/rosetta_ws) — upstream's own pixi workspace; `repos/src.repos` and `repos/libs.repos` are the reference version set
- This repo: [end-to-end pipeline](../demos/end-to-end-pipeline.md), [`so_arm101.yaml` contract](../../pai_data_collection/config/rosetta/so_arm101.yaml), [`pai.repos`](../../pai.repos)
