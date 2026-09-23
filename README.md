# Abstraction_experiment — Minimal teaching system for Unitree G1 vision-guided grasping (ROS edition)

A **trimmed-down catkin project** extracted from the `g1_intellect` repository. It
keeps the complete runtime chain "vision detection → arm SDK control (with voxel
A\* obstacle avoidance) → Revo2 dexterous-hand grasp", plus the two other arm
manipulation flows from the original: **elevator-button press** and
**berry picking**. Architecture, package names, service names and topic names
are identical to the original repository; navigation / SLAM / voice / follower
and the main_process orchestration layer have been cut. **Voxel A\* obstacle
avoidance is fully retained and enabled by default** (`~use_astar_for_grasp`
defaults to true, same as the original repository).

Three arm services are the demo entry points, all on the single arm node:

| Service | What it does | Extra weights needed |
| --- | --- | --- |
| `/g1_arm_simple/grasp_bottle` | Detect a bottle/can, plan (A\*), grasp, lift | none beyond the base set |
| `/g1_arm_simple/press_elevator_button` | Press a button at a given xyz with the index finger | `elevator_button/*` (fetched by manifest) + `button_cls.pt` (fetched) |
| `/g1_arm_simple/pick_berry` | Find the nearest ripe berry, pinch-grasp it, retract | `grasp_berry/*` + `unified_*_yolo11s*.pt` (fetched by manifest) |

Target platform: robot-side Linux, ROS1 Noetic + Python 3.10 (conda environment
`g1_intellect_env`).

Model weights are hosted in two public Hugging Face repos (see Section 5 for
how to fetch them):

- Arm models (IK MLPs + voxel A\* data, ~28 GB): https://huggingface.co/Mattewkang/Abstraction_experiment-arm-models
- Camera models (YOLO detectors + classifiers, ~58 MB): https://huggingface.co/Mattewkang/Abstraction_experiment-camera-models

## 1. System architecture (same as the original repository)

```
realsense2_camera (external apt package, rs_camera.launch align_depth:=true)
    -> /camera/color/image_raw
       /camera/aligned_depth_to_color/image_raw      <- shared by detection + voxel obstacles
       /camera/color/camera_info
            |
            v
g1_camera / object_detector_node.py           (YOLO + button classifier + depth back-projection,
    -> /perception/get_object_position         continuous 5 Hz inference with a fresh-frame gate)
                                              (label+index -> base_link xyz; label may be a fine
                                               button class: up/down/1/B1..B6/open/close/...)
g1_camera / berry_detector_node.py            (berry YOLO + ripeness classifier + colour-aware depth,
    -> /perception/get_berries                 on demand; returns ALL berries in torso_link)
            |
            v
g1_arm_abs / arm_node_abs.py (node name g1_arm_simple, arm control via unitree_sdk2py DDS)
    -> /g1_arm_simple/grasp_bottle
       get xyz from camera -> MLP IK for pregrasp/grasp joints ->
       voxel A* (plan home->pregrasp over head-camera depth obstacle voxels) ->
       hand pre_grasp -> Cartesian approach to grasp -> hand grasp_5f -> vertical lift ->
       (optional retract / GoHome)
    -> /g1_arm_simple/press_elevator_button   (arm + xyz given by the caller)
       hand point_gesture -> elevator-button MLP IK for pre-press/press joints ->
       voxel A* current->pre_press -> direct 5-waypoint leg pre_press->press ->
       settle -> retract -> home -> hand release
    -> /g1_arm_simple/pick_berry              (bare Trigger, one berry per call)
       /perception/get_berries -> ripeness filter (~pick_berry_ripeness) -> nearest berry ->
       arm from y sign -> berry MLP IK -> voxel A* -> hand pre_grasp_berry -> approach ->
       hand grasp_berry -> retract -> home
    -> /g1_arm_simple/{acquire,release}_arm_control (SDK takeover / hand-back, see Section 4)
    -> /g1_arm_simple/voxel_obstacles         (obstacle point cloud avoided by A*, RViz visualization)
            |
            v
g1_hands / revo2_tactile_node.py x2           (bc_stark_sdk Modbus serial)
    -> /g1_hand_{left,right}/{pre_grasp, grasp_5f, release, point_gesture,
                              grasp_berry, pre_grasp_berry}
```

Notes:

- The demo entry point is simply the arm node's own
  `rosservice call /g1_arm_simple/grasp_bottle`; the original repository's
  main_process orchestration layer is not needed.
- `arm_node_abs.launch` keeps the original **TF stub** (base_link→torso_link
  identity + base_link→camera_link extrinsic static_transform_publisher,
  `publish_tf_stubs` defaults to true) and optional RViz (`rviz` defaults to
  true, preloaded with the voxel obstacle display config). The detection node
  parses the extrinsics from yaml for projection itself and does not query TF;
  the TF stub mainly serves RViz's Fixed Frame.
- The three arm flows share one node, one DDS link and one voxel A\* planner.
  Each motion service takes the arms over from the robot's own controller
  automatically (`acquire_arm_control` is implicit); nothing in this project
  hands them back on its own, so call `release_arm_control` (or `GoHome`, which
  releases the hand but keeps SDK control) when you are done. See Section 4.

## 2. Project layout

```
Abstraction_experiment/      Same top-level layout as g1_intellect (packages live at the repo root)
  README.md                  This file
  requirements.txt           Python dependencies (trimmed from the original top-level requirements.txt)
  start_demo.sh              One-shot demo launcher on PC1 (PC2 camera/hands brought up remotely via SSH)
  catkin_ws/                 In-repo workspace; src/ holds symlinks to the packages above
  docs/
    REMOTE_BRINGUP.md        PC1/PC2 split, credentials, chrony time sync, safety notes
  external/
    unitree_sdk2_python/     Unitree DDS SDK (verbatim copy of the original external/)
  g1_arm_abs/                Arm control: grasp / press / berry pipelines + voxel A*
  g1_camera/                 Object detector + button classifier + berry detector, remote camera / time-sync scripts
  g1_hands/                  Revo2 dexterous-hand driver (trimmed) + remote dual-hand scripts
```

The original g1_intellect repository also has g1_beacon / g1_follower /
g1_gimbal / g1_locomotion / g1_navigation / g1_perception / g1_slam /
g1_voice_intent / main_process / pkg_g1_follower and other packages at the top
level. This project removes them wholesale per the trim list in Section 6, so
they do not appear here. The remote camera and time-sync scripts originally
belonged to g1_perception and have been merged into g1_camera.

## 3. Installation (robot-side Linux)

```bash
# 0) Prerequisites: ROS Noetic + realsense2_camera driver
sudo apt install ros-noetic-realsense2-camera

# 1) conda environment (Python 3.10)
conda activate g1_intellect_env
pip install -r requirements.txt
pip install -e external/unitree_sdk2_python

# 2) The workspace is already inside the repo: catkin_ws/src already contains
#    symlinks to ../../<pkg>, no manual ln -s needed. (The symlink layout also
#    lets arm_controller's fallback SDK path probe walk up to external/; if you
#    copy the packages instead, it relies entirely on the pip install -e SDK.)

# 2.5) This project was assembled via Windows/OneDrive, so script exec bits may
#      be lost. roslaunch runs devel-space scripts directly and needs exec bits:
chmod +x /path/to/Abstraction_experiment/*/scripts/*.py /path/to/Abstraction_experiment/*/scripts/*.sh

# 3) Build (inside the conda env so that empy==3.3.4 takes effect)
cd /path/to/Abstraction_experiment/catkin_ws
source /opt/ros/noetic/setup.bash
PATH=$CONDA_PREFIX/bin:$PATH catkin_make
source devel/setup.bash

# 4) Download model weights from the public Hugging Face repos (no account or token needed).
#    Note: g1_arm_abs includes the voxel A* bitmaps, about 28 GB of download / disk, see Section 5.
rosrun g1_arm_abs fetch_weights.py     # grasp_bottle x4 + elevator_button x6 + grasp_berry x4 + abstraction_map x12
rosrun g1_camera  fetch_weights.py     # object detector + button classifier + berry detector + ripeness classifier
```

## 4. Running the demo

Two deployments are supported:

- **Single machine**: camera and hands are both attached to this machine —
  just run `roslaunch g1_arm_abs grasp_demo.launch`.
- **PC1/PC2 split** (the robot's actual wiring: the D435i and both hands are
  attached to the G1's PC2) — run `./start_demo.sh` on PC1. It brings up the
  camera and hands on PC2 over SSH, and keeps only roscore / detection / arm
  locally. See [docs/REMOTE_BRINGUP.md](docs/REMOTE_BRINGUP.md) for credentials
  and chrony time sync.

```bash
# PC1/PC2 split: one shot (fill in ~/.g1_pc2_env first)
export G1_NETWORK_INTERFACE=ens33
./start_demo.sh

# Single machine: one command brings up the whole system (RealSense + detection + hands + arm + RViz)
export G1_NETWORK_INTERFACE=eth0        # Unitree DDS network interface
roslaunch g1_arm_abs grasp_demo.launch
# Headless over SSH:
#   roslaunch g1_arm_abs grasp_demo.launch use_rviz:=false

# Place the bottle inside the right-arm workspace (x<=0.39 m, z>=-0.02 m, torso_link frame), then:
rosservice call /g1_arm_simple/grasp_bottle

# Press an elevator button. The caller supplies arm + xyz of the button face
# (torso_link); get it from the detector first, e.g. the "up" call button:
rosservice call /perception/get_object_position "{label: 'up', index: 0}"
rosservice call /g1_arm_simple/press_elevator_button \
  "{arm: right, x: 0.35, y: -0.10, z: 0.20, pre_press_dx: 0, wait: true, timeout: 0}"
# pre_press_dx: 0 -> node default ~press_elevator_button_default_pre_press_dx (0.05 m)

# Pick one berry (berry detector is up by default, bringup_berry_detector:=true):
rosservice call /g1_arm_simple/pick_berry

# Wrap up: release the hand + both arms back to home, then hand the arms back to the robot
rosservice call /g1_arm_simple/GoHome
rosservice call /g1_arm_simple/release_arm_control
```

`grasp_demo.launch` args: `bringup_camera`, `bringup_hands`, `bringup_arm`,
`bringup_berry_detector` (all default true), `use_rviz`, `publish_tf_stubs`,
`grasp_bottle_label`, `network_interface`, `simulation_mode`.

Common debugging entry points:

| Service | Purpose |
| --- | --- |
| `/g1_arm_simple/grasp_bottle` (`std_srvs/Trigger`) | Get bottle coordinates from the camera + full grasp flow (main demo entry) |
| `/g1_arm_simple/press_elevator_button` (`PressElevatorButton`) | Point gesture + A\* to pre-press + direct press leg + settle + retract + home + release. Caller gives arm and xyz |
| `/g1_arm_simple/pick_berry` (`std_srvs/Trigger`) | Query `/perception/get_berries`, filter by `~pick_berry_ripeness` (default `black`), nearest berry, arm from y sign, pinch grasp, retract, home |
| `/g1_arm_simple/acquire_arm_control` / `release_arm_control` | Take the arms over from / hand them back to the robot's own controller (weight ramp 0↔1). Motion services acquire implicitly; release refuses unless both arms are at home |
| `/perception/get_berries` (`GetBerries`) | Test the berry detector alone (`ripeness: ""` = all, or `black` / `red` / `white`) |
| `/g1_arm_simple/move_arm_joints` | Drive one arm to 5 joint angles |
| `/g1_arm_simple/infer_grasp_from_point` | NN inference of pregrasp/grasp joints only, no motion |
| `/g1_arm_simple/plan_grasp_from_point` | Planning only (A\* pre-path + lift/retract/home joints), no motion |
| `/g1_arm_simple/execute_grasp_from_point` | Full grasp execution for a given xyz |
| `/g1_arm_simple/execute_astar_path_from_point` | Grasp execution for a given xyz, forcing A\* |
| `/g1_arm_simple/plan_to_pose` | Plan to an arbitrary 5-DoF end-effector pose (optional A\*) |
| `/g1_arm_simple/GoHome` / `GoHomeArm` | Both arms / one arm back to home + release |
| `/perception/get_object_position` | Test vision alone (label: "bottle", index: 0) |
| `/g1_hand_right/pre_grasp` etc. | Test a hand alone |

Dry-run before a real motion: set `~grasp_bottle_dry_run:=true` or
`~pick_berry_dry_run:=true` on the arm node (or `rosparam set
/g1_arm_simple/<param> true` before launch) to print only the target and IK
result without moving the arm; `arm_node_abs.launch simulation_mode:=true`
skips DDS entirely.

### Press-button and berry parameters

- **Press**: `~press_elevator_button_workspace_max_x` (0.45 m, farther than the
  grasp gate so the fingertip can reach a forward-mounted panel; keep ≤ the
  voxel grid edge 0.6 m) / `_min_z` (-0.02 m); per-arm offsets
  `~press_elevator_button_{r,l}_{x,y,z}_offset` (x defaults to -0.005 = press
  0.5 cm deeper than the detected face); `~press_elevator_button_default_pre_press_dx`
  (0.05), `_default_retract_dx` (0.15), `_post_settle_sec` (1.5), `_hand_settle_sec` (1.0).
- **Berry**: `~pick_berry_ripeness` (`black`), `~pick_berry_arm_y_split` (0.0),
  `~pick_berry_use_astar` (true), `~pick_berry_workspace_max_x` (0.42) /
  `_min_z` (-0.2, lower than the grasp floor because berries hang low),
  per-arm `~pick_berry_{r,l}_{x,y,z}_offset`, `~pick_berry_dry_run` (false).
  Berry detector tuning lives in `g1_camera/config/berry_params.yaml`
  (ripeness colour gates, temporal depth median, residual `x_offset`).
- **Control hand-back**: `~release_arm_control_after_init_home` (true) hands the
  arms back to the robot after the startup home move; `~control_switch_delay` /
  `~control_switch_settle` (0.5 s each) shape every takeover / hand-back ramp.

### Viewing A\* obstacles in RViz

`grasp_demo.launch` (or `arm_node_abs.launch`) defaults to `rviz:=true` and
opens RViz with the `config/voxel_obstacles.rviz` config, showing the topic
`/g1_arm_simple/voxel_obstacles` (frame `torso_link`) — the depth obstacle
voxels A\* is currently avoiding. To view manually: set RViz's Fixed Frame to
`torso_link` (provided by publish_tf_stubs), Add → PointCloud2 → pick that topic.

### Notes on publish_tf_stubs

In the original repository the base_link→…→camera TF chain is published by
g1_slam (URDF + transforms.launch), and `publish_tf_stubs:=true` would then
cause a TF double-parent conflict. **This project does not include g1_slam, so
keep the default true**; set `publish_tf_stubs:=false` only if you run another
stack publishing these TFs under the same roscore.

## 5. Model weights

All weights live in two **public** Hugging Face model repos, one per package:

| Package | Hugging Face repo | Contents (manifest `config/model_weights.yaml`) |
| --- | --- | --- |
| g1_arm_abs | [Mattewkang/Abstraction_experiment-arm-models](https://huggingface.co/Mattewkang/Abstraction_experiment-arm-models) | 4 grasp_bottle + 6 elevator_button + 4 grasp_berry + 12 abstraction_map (voxel A\*) files, 26 total, ~28 GB |
| g1_camera | [Mattewkang/Abstraction_experiment-camera-models](https://huggingface.co/Mattewkang/Abstraction_experiment-camera-models) | `find_elevator_bottle.pt`, `button_cls.pt`, `unified_detector_yolo11s.pt`, `unified_classifier_yolo11s_cls.pt`, ~58 MB |

### How to fetch

The recommended way is the per-package script, which reads the manifest,
downloads every entry from the pinned repo commit (`hf_revision`), verifies its
sha256, and skips files already present with the right hash:

```bash
conda activate g1_intellect_env          # provides huggingface_hub
source catkin_ws/devel/setup.bash
rosrun g1_arm_abs fetch_weights.py       # -> g1_arm_abs/models/<nested path>, ~28 GB
rosrun g1_camera  fetch_weights.py       # -> g1_camera/models/<basename>
```

The repos are public, so no Hugging Face account, login or `HF_TOKEN` is
needed. Re-running is safe and cheap for the small files; the arm script does
hash the two 14 GB bitmaps on every run, which takes about a minute.

Alternatives, if you prefer not to use the scripts:

```bash
# Hugging Face CLI (pip install huggingface_hub), same layout as the scripts expect:
hf download Mattewkang/Abstraction_experiment-arm-models    --local-dir g1_arm_abs/models
hf download Mattewkang/Abstraction_experiment-camera-models --local-dir g1_camera/models

# git + git-lfs:
git lfs install
git clone https://huggingface.co/Mattewkang/Abstraction_experiment-arm-models    g1_arm_abs/models
git clone https://huggingface.co/Mattewkang/Abstraction_experiment-camera-models g1_camera/models
```

The manifests pin a specific repo commit; `hf download` / `git clone` fetch
the latest `main`, which is the same content unless the repos are updated
later. `verify_pinned_weights()` at arm-node startup compares sha256 against
the manifest regardless of how the files got there.

To publish updated weights: push the new file to the repo, then bump
`hf_revision` and the file's `sha256` in the package manifest.

### Berry weights

| Consumer | File | Notes |
| --- | --- | --- |
| arm `pick_berry` MLP IK | `g1_arm_abs/models/grasp_berry/{left,right}/best_model.pth` + `x_scaler.pkl` | Manifest-pinned, so `verify_pinned_weights()` refuses to start without them like every other entry |
| `berry_detector_node` | `g1_camera/models/unified_detector_yolo11s.pt` | Detector (0:button 1:bottle 2:berry). The node FATAL-exits at startup if missing; other nodes are unaffected |
| `berry_detector_node` | `g1_camera/models/unified_classifier_yolo11s_cls.pt` | Ripeness classifier (black / red / white). Missing = warning, labels stay `berry` and the ripeness filter matches nothing |

Set `bringup_berry_detector:=false` on `grasp_demo.launch` if you do not need
the berry flow on a given machine.

### Size warning (voxel A\* bitmaps)

- The two files `abstraction_map/voxel_config_bitmap_{left,right}.uint8` total
  **about 28 GB** (per the manifest comment; the first `fetch_weights.py` run
  needs corresponding download time and disk space, check free space first).
- The manifest pins the **decompressed raw `.uint8` bitmaps**; the files
  downloaded by `fetch_weights.py` are usable directly and there is **no need**
  to run `decompress_once.py`.
  `src/g1_arm_abs/voxel_planner_deploy/decompress_once.py` is only for a one-time
  decompression when what you have is a `.uint8.gz` archive (e.g. copied from
  another machine).
- At node startup `verify_pinned_weights()` checks all 26 manifest files: if any
  is missing (including the two bitmaps, the elevator_button and grasp_berry
  files) it `logfatal`s and exits, telling you to run
  `rosrun g1_arm_abs fetch_weights.py` first. The two bitmaps are only
  checked for existence by default (`verify_sha256: false`; a full sha256 would
  add ~30 s startup delay). To force full verification use
  `~force_full_weight_sha256:=true`.
- The first plan JIT-compiles the numba kernels and memmaps the bitmaps, so a
  slow first A\* plan on cold start is normal.

### Detector and button-classifier weights

`g1_camera/config/params.yaml` defaults to `models/find_elevator_bottle.pt`
(detector) and `models/button_cls.pt` (button classifier), both downloaded by
`fetch_weights.py` from the manifest. The classifier runs on every `button`
detection and renames it to one of `up, down, 1, B1..B6, open, close,
emergency_yellow, phone` when its probability is ≥ `button_classifier_conf`
(0.15), so `get_object_position` can be asked for `label: "B2"` directly; the
press flow relies on this. Set `button_classifier: ""` to disable it.

The original repository's `params.yaml` had drifted to
`unified_classifier_yolo11s_cls.pt` for this key; this project points it back at
the manifest-pinned `button_cls.pt` so a fresh deployment works out of the box.

**The measured classes are `{0: button, 1: drink_can}`; there is no `bottle`.**
(The original repository's `g1_camera/CLAUDE.md` lists `button`, `bottle`, which
does not match the actual weights; corrected on 2026-09-07 by checking
`YOLO(...).names` on this machine.) The arm node's own default
`~grasp_bottle_label` in `params.py` is still `bottle`, so using it directly is
guaranteed to fail:

```
success: False
message: "get_object_position failed: No valid detection found for label 'bottle'"
```

Therefore `arm_node_abs.launch` / `grasp_demo.launch` override the default
`grasp_bottle_label` to `drink_can` to match the weights actually downloaded, so
a fresh deployment works out of the box.

The robot in the original repository actually uses
`unified_detector_yolo11s.pt` (classes 0:button 1:bottle 2:berry), but that
weight has no entry in the original manifest (pre-existing drift, not introduced
by this trim), so `fetch_weights.py` cannot download it. To match the robot's
current state exactly, copy that file manually from
`g1_intellect/g1_camera/models/`, change `model:` in `params.yaml` back to
`models/unified_detector_yolo11s.pt`, **and at the same time change
`grasp_bottle_label` back to `bottle`**:

```bash
roslaunch g1_arm_abs grasp_demo.launch grasp_bottle_label:=bottle
```

Note: the manifest only determines the sha256 verification scope; **if the model
file itself is missing, the detection node exits with FATAL at YOLO load** (dies
at startup, after which `grasp_bottle` fails because
`/perception/get_object_position` never appears). Make sure the file `model:`
points to actually exists before launching.

## 6. Differences from the original repository

### Packages removed wholesale

`main_process`, `g1_navigation`, `g1_slam`, `g1_voice_intent`,
`g1_locomotion`, `g1_gimbal`, `g1_perception`, `g1_beacon`,
`pkg_g1_follower`, `catkin_ws`, `docs`.

### g1_arm_abs (complete: grasp, press, berry, voxel A\*, control hand-back)

The package is now a verbatim copy of the original `g1_arm_abs` except for two
files, so `press_elevator_button`, `pick_berry` and
`acquire_arm_control` / `release_arm_control` behave exactly as on the robot:

| File | Change relative to the original |
| --- | --- |
| `src/g1_arm_abs/arm_controller.py` | Only `_ensure_local_unitree_sdk_path` is modified: realpath + walk up directory levels probing for `external/unitree_sdk2_python`, to fit this layout; everything else unchanged |
| `launch/arm_node_abs.launch` | Adds the `grasp_bottle_label` arg (default `drink_can`, see Section 5); everything else identical, including the press workspace args and the control-switch args |
| `launch/grasp_demo.launch` | **New**: one command brings up RealSense + object detector + berry detector + hands + arm |

Not carried over (unrelated to the three flows): `scripts/dump_voxel_obstacles.py`
(standalone debugging tool), `msg/` (three msgs unregistered and unused in the
original), `CLAUDE.md` / `README.md` (package docs; this file covers them).

<details>
<summary>Historical note: the earlier trimmed edition of this package</summary>

Copied as-is: `setup.py`, `.gitignore`, `scripts/fetch_weights.py`,
`srv/{MoveArmJoints,InferGraspFromPoint,ExecuteGraspFromPoint,ExecuteAstarPathFromPoint,PlanGraspFromPoint,PlanToPose,GoHomeArm}.srv`,
`config/voxel_obstacles.rviz`,
`src/g1_arm_abs/{__init__,joint_index,utils,arm_controller_proc,solve_grasp_pose,path_ops,joint_validation,pose_compensation,nn_backend,grasp_plan_math,hand_services,motion_exec,homing,grasp_pipeline,pose_planning,async_dispatch,camera_lookup,weight_verify,astar_planning,voxel_setup,voxel_planner_adapter}.py`,
`src/g1_arm_abs/voxel_planner_deploy/` (voxel_phase3_query.py,
voxel_phase4_astar.py, voxel_numba_kernels.py, decompress_once.py, README.md).

Modified (each file header carries a "source + trimmed-edition changes" note;
changes are **limited to deleting press/berry**):

| File | Change |
| --- | --- |
| `scripts/arm_node_abs.py` | Based on the original, deletes only: press (PressElevatorButton service/handler/ctx, point_gesture client and accessors, fire_delayed_point_gesture) and berry (pick_berry service/handler, berry NN loading, berry IK/pose/workspace/A\* variants, berry retract, GetBerries client) plus the corresponding imports. Voxel A\* (_init_voxel_planner, the real _plan_astar_segments / _plan_astar_path_between_joints, ~execute_astar_path_from_point) is identical to the original |
| `src/g1_arm_abs/params.py` | Deletes only the berry model path, point_gesture / berry gesture service names, the press parameter block and the berry pick parameter block; the voxel parameter block and `~use_astar_for_grasp` (default true) / `~use_astar_for_go_home` (default false) etc. are identical to the original |
| `src/g1_arm_abs/arm_controller.py` | Only `_ensure_local_unitree_sdk_path` is modified: realpath + walk up directory levels probing for `external/unitree_sdk2_python`, to fit the new layout; everything else unchanged |
| `config/model_weights.yaml` | Only the 6 elevator_button entries are removed; 4 grasp_bottle + 12 abstraction_map = 16 entries kept as-is (including `verify_sha256: false` for the bitmaps) |
| `CMakeLists.txt` / `package.xml` | Only PressElevatorButton.srv removed; tf2_ros / rviz dependencies kept |
| `launch/arm_node_abs.launch` | Only the two press workspace parameters removed; the voxel parameter section, TF stub and RViz section are identical to the original |
| `launch/grasp_demo.launch` | **New**: one command brings up RealSense + detection + hands + arm, passing through publish_tf_stubs / use_rviz |

Removed at the time: `berry_lookup.py`, `elevator_button_ik.py`,
`press_planning.py`, `press_pipeline.py`, `srv/PressElevatorButton.srv`,
`test/` (berry IK error tool). All of these are back.

</details>

### g1_camera

Copied as-is from the original: `srv/{GetObjectPosition,GetBerries}.srv`,
`scripts/{fetch_weights,berry_detector_node,button_classifier}.py`,
`launch/{object_detector,berry_detector}.launch`, `config/berry_params.yaml`,
`test/visualize_berry.py`, `.gitignore`.

Modified relative to the original:

| File | Change |
| --- | --- |
| `scripts/object_detector_node.py` | Keeps this project's streaming frame cache + continuous inference (`~inference_rate_hz`, `~max_frame_age_sec`, `~max_rgb_depth_skew_sec`), which the original does not have, **and** re-adds the original's button-classifier path (load at init, batched crop classification after YOLO, `button_class` / `button_class_score` in the JSON records and the annotated image) |
| `config/params.yaml` | Detector default `find_elevator_bottle.pt`; classifier default `button_cls.pt` (original: `unified_classifier_yolo11s_cls.pt`, unpinned); fine button classes listed in `class_radius_m`; streaming-cache keys; **extrinsics preserved verbatim** |
| `config/model_weights.yaml` | Detector + button classifier + berry detector + ripeness classifier entries, from the original's berry-pinning branch |
| `CMakeLists.txt` / `package.xml` | `object_marker_node.py`, `object_visualization.rviz` and `visualization_msgs` stay out (RViz marker visualisation, unrelated to the three flows); remote-bringup shell scripts installed via `install(PROGRAMS)` |

Removed: `object_marker_node.py`, `config/object_visualization.rviz`,
`launch/{object_markers,object_visualization}.launch`.

### g1_hands

Copied as-is: `scripts/revo2_tactile_node.py` (including the berry gesture
services), `src/g1_hands/{__init__,grasp_controller,revo2_utils}.py`,
`msg/TactileStatus.msg`, `setup.py`,
`launch/revo2_{left,right,dual}_hand_node.launch`.

Modified: `CMakeLists.txt` / `package.xml` (install entries for the two
auto_release srvs and two scripts removed).

Removed: `auto_release_monitor_node.py`, `record_tactile.py`,
`srv/{ArmAutoRelease,DisarmAutoRelease}.srv`, `launch/auto_release_monitor.launch`.

## 7. Safety notes

- **Voxel A\* is on by default (same as the original repository)**: the
  home→pregrasp segment of the grasp and forced-A\* services is planned with
  obstacle avoidance over the head-camera depth obstacle voxels. But note that
  (1) the pregrasp→grasp Cartesian approach segment and the lift/retract
  segments do **not** check obstacles, (2) obstacles outside the camera's field
  of view or behind the robot are not in the voxel map, (3) `~voxel_shortcut`
  defaults to false, do not enable it (original repository comment: the
  shortcut's straight-line check misses diagonal crossings and once caused a
  collision). Keep basic clearance around the target during demos.
- **Press leg is not obstacle-checked**: `press_elevator_button` plans only the
  current→pre-press segment with A\*; the pre-press→press leg is five direct
  waypoints, and retract is direct. Keep the panel area clear and start with a
  dry probe of `get_object_position` to confirm the target xyz is sane
  (the press workspace gate is 0.45 m in x, wider than the grasp gate).
- **Berry pick is one berry per call and trusts the detector**: it picks the
  nearest berry of `~pick_berry_ripeness` and chooses the arm from the berry's
  y sign. With the ripeness classifier missing every berry is labelled `berry`
  and the filter matches nothing, so the call fails safely without motion.
- **Arm control hand-back**: the arms start under the robot's own controller and
  each motion service takes them over automatically. Nothing hands them back
  after a task in this project (the original's main_process did), so call
  `release_arm_control` when done; it refuses unless both arms are at home, so
  run `GoHome` first.
- **Missing A\* bitmaps refuse to start**: `verify_pinned_weights()` checks all
  26 manifest files and FATALs on any missing one (telling you to run
  fetch_weights.py). It never silently degrades to running without avoidance.
- **Velocity clamp**: do not raise `~velocity_limit` (default 20, per-cycle clamp
  in the 250 Hz control loop).
- **Weight ramp**: on exit, `stop(release_control=True)` ramps the motion-mode
  weight from 1 to 0 before releasing. Do not kill -9 the arm node process; use
  Ctrl-C so it goes through the shutdown flow, otherwise the arm loses torque
  abruptly.
- **Workspace gate**: grasp targets are gated by `~workspace_max_x` (0.39 m) /
  `~workspace_min_z` (-0.02 m) and rejected outright when out of range. Do not
  relax these — the NN training distribution only covers this range.
- **NN joint clamp**: `~clip_inferred_joints_to_limits` defaults to true and
  clamps MLP outputs back to the URDF joint limits; keep it on.
- The DDS link may be unstable on first power-up; the arm node retries via
  `~connect_retries` (3 attempts). Both arms go home on startup
  (`~init_to_home_on_start`, direct, no A\*), so make sure the area around both
  arms is clear before launching.
- Swapping the two hand serial ports (default left `/dev/ttyUSB0`, right
  `/dev/ttyUSB1`) swaps the left and right hands. If one hand is absent, disable
  it with `roslaunch ... grasp_demo.launch bringup_hands:=false` or g1_hands'
  `bringup_left_hand:=false`.
