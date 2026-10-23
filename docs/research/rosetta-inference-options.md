# Inference on top of Rosetta: options

**Scope.** This document covers the ways we can run policy inference for the SO-ARM101 demos while keeping Rosetta for the robot ↔ policy translation. It explains the parts of Rosetta that matter for the decision, describes three ways to build on it, and recommends an order.

**Versions.** Rosetta 0.2.0 (`dfd07e4`), `lerobot-rosetta` 0.2.0, LeRobot 0.6.0, ROS 2 Lyrical. Nothing here has been built or benchmarked yet. Statements about how Rosetta behaves come from reading its source; latency figures are estimates.

## TL;DR

- Rosetta's **contract and `TopicBridge`** handle all the robot-side processing: topics to arrays, units, resizing, timing, safety. That holds whichever inference engine we use, so swapping the engine does not cost us recording, datasets, or the contract.
- There are three ways to plug in our own inference. In increasing cost: **(1) a custom `PolicyRunner` plugin**, **(2) our own node that uses `TopicBridge` as a library**, **(3) a C++ inference node**.
- Options 1 and 2 keep train/serve parity by construction. Option 3 is the only one that takes Python out of the 50 Hz loop, but in practice it means re-implementing part of Rosetta in C++.
- **Recommendation:** measure the current path first. Then try an in-process Option 1 runner, which removes today's gRPC hop and keeps Rosetta intact. Only go to C++ if the numbers demand it, and ask Rosetta upstream for help before rewriting it ourselves.

## 1. Rosetta in brief

### What it is

Rosetta is a ROS 2 ↔ robot-learning bridge. One YAML file per robot, the **contract**, declares which topics are observations and actions, how to convert them, and at what rate. Rosetta uses that file both **offline** (recording bags and converting them to a training dataset) and **online** (feeding a live robot to a policy). Because both directions run the same code, the policy sees the same thing at deployment that it saw in training.

### The packages

| Package                 | Role                                                                                               | Depends on LeRobot? |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ------------------- |
| `rosetta`               | The core: contract, frames, ROS 2 nodes, the dataset converter (`rosetta_port`), plugin registries | No                  |
| `rosetta_interfaces`    | ROS 2 actions and services (`RunPolicy`, `RecordEpisode`, …)                                       | No                  |
| `lerobot_rosetta`       | Plugs LeRobot into Rosetta: dataset writer, policy runner, policy server                           | Yes                 |
| `lerobot_robot_rosetta` | Plugs a Rosetta robot into LeRobot's tools (`--robot.type=rosetta`)                                | Yes                 |

The core never imports LeRobot. LeRobot is attached only through plugins.

### The layers

| Layer              | Pieces                                                            | Tied to an ML framework?                              |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------------------- |
| Offline            | `episode_recorder_node` (bags), `rosetta_port` (bags → dataset)   | No. The dataset format is a plugin (`DatasetWriter`). |
| **The core layer** | **Contract + `TopicBridge`**: topics ↔ frames, in both directions | **No. This is the part worth keeping.**               |
| Online             | `policy_runner_node` + a `PolicyRunner` plugin                    | Only the plugin                                       |

### Frames and "dataset space"

Rosetta's unit of exchange is the **frame**: a dict mapping each contract key to a NumPy array. For SO-ARM101 at 50 Hz:

- `observation.images.wrist`, `observation.images.static`: 480×480×3 `uint8` RGB
- `observation.state`: 6 joint positions in **degrees**
- `action`: 6 joint targets in **degrees**

These values are in **dataset space**: exactly what was written into the training dataset. A policy reads observation frames and returns action frames, and never touches a ROS message.

### What Rosetta does, and what it doesn't

| Rosetta handles (robot side)                               | You handle (model side)                            |
| ---------------------------------------------------------- | -------------------------------------------------- |
| Decoding images to RGB arrays, picking joints by name      | Converting images to float tensors in [0, 1]       |
| `resize`, radians ↔ degrees, clamping, from the contract   | Normalising with the training dataset's statistics |
| Resampling every stream onto the 50 Hz frame clock         | Running the model (device, batching)               |
| Waiting until every stream has data; zero-filling dropouts | Unnormalising the outputs                          |
| Rejecting NaN/Inf commands; publishing all streams or none | Queueing and blending action chunks                |
| Safety watchdog: hold the last command if actions stop     | Pacing the control loop                            |

Today LeRobot does the right-hand column inside its policy server. **Any alternative engine has to do it too.** Normalising with the wrong statistics produces confident but meaningless actions, with no error to point at the cause.

### Extension points

| Extension point | What you provide                                            | How it's selected                                     |
| --------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| `PolicyRunner`  | Online inference: control loop and model                    | `framework` parameter of `policy_runner_node`         |
| `DatasetWriter` | Offline dataset format                                      | `rosetta_port --framework <name>`                     |
| Operators       | New `apply:` transforms in the contract                     | Named in the contract                                 |
| Codecs          | Support for new ROS message types                           | The channel's message type, or a path in the contract |
| `FrameIO`       | A non-ROS 2 transport (a Python interface with no registry) | Whoever hosts the runner                              |

All of these except `FrameIO` are standard Python entry points: installing a package that registers one is enough. Unknown or broken plugins fail loudly when they load.

### Today's deployment path

`policy_runner_node` loads the `lerobot` runner. The runner starts a LeRobot **policy server as a separate Python process** and talks to it over **gRPC**. Every observation therefore crosses two process boundaries before reaching the model. Our working assumption, which still needs measuring, is that this hop costs more than ROS transport or Rosetta's own processing.

## 2. Three ways to build custom inference

### Option 1: a custom `PolicyRunner` plugin

We write a Python class with five methods: `setup`, `run`, `feedback`, `request_stop`, `teardown`. We register it under the `rosetta.policy_runners` entry point and set `framework: <our name>` in the params file. `run()` is our control loop: sample a frame, run the model, publish an action frame.

- **We keep:** the whole node, including contract loading, `TopicBridge`, the `run_policy` action, start/cancel services, feedback, and safety handling on cancel or deactivate.
- **We write:** the model-side column above, plus the control loop. The engine can be anything: ONNX Runtime, TensorRT, torch in-process, or a client to a remote GPU server.
- **Effort:** low, a few hundred lines of Python. It is additive, and LeRobot stays selectable alongside it.
- **Latency:** an **in-process** runner removes today's gRPC hop and subprocess. Frames still pass through Python.
- **Testing:** `run()` only depends on the `FrameIO` interface, so it can be unit-tested with a fake bridge and no ROS.

### Option 2: our own node, with `TopicBridge` as a library

We write our own ROS 2 lifecycle node. It loads the contract (`load_contract`), creates a `TopicBridge`, and calls `sample_frame()` / `publish_frame()` from its own loop.

- **When:** the "configure, then run against a goal" model doesn't fit. Examples are a policy that runs whenever the node is active, or a supervisor that switches between several policies.
- **We keep:** all of Rosetta's robot-side processing and parity.
- **We lose, and must handle ourselves:** the action server, services and feedback; sending the safety command on deactivate; running inference on its own thread so it doesn't block subscriptions or the watchdog.
- **Effort:** low to medium.

### Option 3: a C++ inference node

A C++ node subscribes to the cameras and `/joint_states`, runs the model (for example ONNX Runtime), and publishes commands. Rosetta still records and converts data for training, but **does not run at deployment**.

- **Gain:** the only option that removes Python from the 50 Hz path.
- **Cost:** the contract becomes a specification we re-implement in C++: resizing, unit conversions, joint selection, resampling, warm-up, NaN guard, all-or-nothing publish, and the safety hold. In practice this is **rewriting Rosetta's online half**, plus the model-side work.
- **Main risk: silent train/serve skew.** Small mismatches still produce plausible-looking actions, just worse ones. A concrete trap: Rosetta's resize uses NumPy rounding (half to even), which neither `cv::resize` nor `std::round` reproduces.
- **Prerequisites:** an ONNX export of our ACT policy (LeRobot has none yet; it is planned for 0.7.0), a parity test that replays recorded bags through both paths, and this repo's first C++ package.
- **Effort:** high.

### Comparison

|                                  | Option 1: `PolicyRunner`              | Option 2: own node                    | Option 3: C++ node        |
| -------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------- |
| Contract + robot-side processing | Rosetta                               | Rosetta                               | **We re-implement**       |
| Safety watchdog                  | Rosetta                               | Rosetta, plus our deactivate handling | **We re-implement**       |
| Action server / services         | Rosetta                               | We write or omit                      | We write or omit          |
| Model-side pre/post              | We write                              | We write                              | We write                  |
| Train/serve parity               | Guaranteed                            | Guaranteed                            | **Must be tested**        |
| Python in the 50 Hz path         | Yes                                   | Yes                                   | No                        |
| Effort                           | Low                                   | Low–medium                            | High                      |
| Best for                         | Swapping the engine, remote inference | Custom node lifecycles                | Hard latency requirements |

## 3. Avoiding a C++ rewrite: what Rosetta could provide

If Option 3 becomes necessary, a private rewrite is the worst way to get it. Four things Rosetta could provide upstream:

- **A. Conformance tests (low cost).** Shared test vectors: bags, the frames they should produce, and action → message pairs. Any reimplementation could then prove it matches. Worth asking for in every scenario.
- **B. Contract operators exported to ONNX (medium cost).** `resize`, `rad2deg` and `clamp` all have exact ONNX equivalents. Rosetta could generate pre- and post-processing subgraphs from the contract and fuse them with the policy model. The C++ node would shrink to the non-numeric parts (decoding, joint selection, timing, safety), which are much harder to get subtly wrong.
- **C. An official C++ contract runtime (high cost).** Written once upstream and validated against A. It could ship as plugins for the C++ policy pipeline work happening in the `ros2_control` community.
- **D. A standalone frame-publisher node (low–medium cost).** Rosetta would publish assembled frames as messages for any C++ consumer. Python stays in the observation path, so it doesn't help latency on its own.

The sensible ask is **A + B**. Treat C as the long-term target, if the C++ pipeline ecosystem matures.

## 4. Decoupling from LeRobot

Because the seams are independent, **datasets and inference can be decoupled separately**. We could keep training on LeRobot datasets while serving with our own runner.

- **Inference:** replace with a custom `PolicyRunner` (Option 1). The cheapest step.
- **Dataset format:** replace with a custom `DatasetWriter`, only if training moves off LeRobot.
- **HIL reward classifier:** currently LeRobot-provided. Needs its own runner if we use human-in-the-loop training.
- **Conventions that stay in Rosetta's core:** depth images are rejected (a LeRobot limitation); image streams are identified by the `observation.images.*` key prefix; and recovering the contract automatically from a checkpoint only works for LeRobot checkpoints, so other engines must always set `contract_path`.
- **Payoff:** most of the dependency workarounds in our `pixi.toml` exist because of LeRobot, and they would go away with it.

## 5. Recommendation and next steps

1. **Measure the current path.** Time `TopicBridge`, the gRPC hop and inference separately. Every latency claim so far is an estimate.
2. **Build an in-process Option 1 runner.** It is cheap, keeps parity by construction, removes the gRPC hop, and is the first step toward LeRobot independence regardless.
3. **Prototype an ACT → ONNX export**, with a numerical equivalence test against the torch checkpoint. Option 3 needs it, and it is useful on its own (TensorRT, edge deployment).
4. **Only if measurements still justify C++:** ask Rosetta upstream for conformance tests and ONNX operator export (A + B) before writing a C++ node.

_The full analysis, with code examples for each option and source references, is in `docs/research/policy-execution-and-buffer-backends.md` (§6–§7)._
