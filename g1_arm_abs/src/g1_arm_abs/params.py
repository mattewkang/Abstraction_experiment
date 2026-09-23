# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/params.py

Centralised ROS parameter loading for the g1_arm_simple node, extracted from
``arm_node_abs.__init__``.

``load_params`` performs every ``rospy.get_param`` read the node needs and
returns a namespace whose attribute names match the node's ``self._*`` fields
exactly (the node applies them via ``self.__dict__.update(vars(p))``). The
param-derived ``_device`` and ``_pose_comp_cfg`` are built here too, since they
depend only on params. Live ROS objects (controller, NN models, service
proxies, publishers, the voxel adapter) are NOT built here — the node owns
those.

ROS-light: reads params and constructs the torch device; advertises nothing.
The reads carry no side effects, so consolidating them here (ahead of the
node's object construction) preserves behaviour.
"""

import os
from types import SimpleNamespace

import rospy

from g1_arm_abs.nn_backend import make_device
from g1_arm_abs.pose_compensation import PoseCompensationConfig

__all__ = ["load_params"]


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def load_params(pkg_root: str) -> SimpleNamespace:
    """Read all node parameters and return them as a namespace of ``_*`` fields.

    ``pkg_root`` is the package root (derived from the script path) used to
    resolve default model paths.
    """
    p = SimpleNamespace()

    # -------------------------------------------------
    # 基本参数
    # -------------------------------------------------
    p._network_interface = rospy.get_param(
        "~network_interface",
        os.environ.get("G1_NETWORK_INTERFACE", "eth0")
    )
    p._simulation_mode = _to_bool(rospy.get_param("~simulation_mode", False))
    p._motion_mode = _to_bool(rospy.get_param("~motion_mode", True))
    p._control_frequency = float(rospy.get_param("~control_frequency", 250.0))
    p._velocity_limit = float(rospy.get_param("~velocity_limit", 20.0))
    # Run the DDS ArmController (250 Hz publish loop) in a dedicated child
    # process so the node's vision/inference/service threads cannot stall it
    # through the GIL. Ignored in simulation mode (in-process controller).
    p._controller_subprocess = _to_bool(
        rospy.get_param("~controller_subprocess", True)
    )

    # SDK takeover / hand-back. The arm starts under the robot's own
    # controller; every motion service takes over (arm_sdk weight 0->1)
    # and ~release_arm_control / the orchestrator hands back (1->0). Both
    # ramps last ~control_switch_delay seconds and are followed by a
    # ~control_switch_settle hold before the next command may start.
    p._control_switch_delay = float(rospy.get_param("~control_switch_delay", 0.5))
    p._control_switch_settle = float(rospy.get_param("~control_switch_settle", 0.5))
    p._release_arm_control_after_init_home = _to_bool(
        rospy.get_param("~release_arm_control_after_init_home", True)
    )
    # ~release_arm_control refuses to hand back unless BOTH arms measure
    # within this per-joint tolerance (rad, max abs error) of their
    # ~{arm}_home_joints_deg. Looser than ~position_tolerance because the
    # arm may have settled slightly since the home move finished.
    p._release_arm_control_require_home = _to_bool(
        rospy.get_param("~release_arm_control_require_home", True)
    )
    p._release_arm_control_home_tolerance = float(
        rospy.get_param("~release_arm_control_home_tolerance", 0.15)
    )

    p._connect_retries = int(rospy.get_param("~connect_retries", 3))
    p._connect_retry_interval = float(
        rospy.get_param("~connect_retry_interval", 2.0)
    )

    p._default_timeout = float(rospy.get_param("~default_timeout", 5.0))
    p._position_tolerance = float(rospy.get_param("~position_tolerance", 0.07))

    # -------------------------------------------------
    # NN / 推理参数
    # -------------------------------------------------
    grasp_bottle_dir = os.path.join(pkg_root, "models", "grasp_bottle")
    _model_file = "best_model.pth"

    p._use_cuda = _to_bool(rospy.get_param("~use_cuda", False))
    p._device = make_device(p._use_cuda)

    p._left_model_path = rospy.get_param(
        "~left_model_path",
        os.path.join(grasp_bottle_dir, "left", _model_file)
    )
    p._left_scaler_path = rospy.get_param(
        "~left_scaler_path",
        os.path.join(grasp_bottle_dir, "left", "x_scaler.pkl")
    )
    p._right_model_path = rospy.get_param(
        "~right_model_path",
        os.path.join(grasp_bottle_dir, "right", _model_file)
    )
    p._right_scaler_path = rospy.get_param(
        "~right_scaler_path",
        os.path.join(grasp_bottle_dir, "right", "x_scaler.pkl")
    )

    p._left_hidden_dims  = rospy.get_param("~left_hidden_dims",  [256, 512, 256])
    p._right_hidden_dims = rospy.get_param("~right_hidden_dims", [256, 512, 256])
    p._left_dropout  = float(rospy.get_param("~left_dropout",  0.0))
    p._right_dropout = float(rospy.get_param("~right_dropout", 0.0))

    # ---------------------------------------------------------------------
    # Berry-pick grasp MLP (separate weight tree from grasp_bottle). Used
    # ONLY by ~pick_berry; the bottle grasp and elevator-button press paths
    # keep their own models untouched. Same MLPRegressor 5->5 architecture as
    # grasp_bottle, so it reuses the same hidden-dim / dropout params.
    # ---------------------------------------------------------------------
    grasp_berry_dir = os.path.join(pkg_root, "models", "grasp_berry")
    p._left_berry_model_path = rospy.get_param(
        "~left_berry_model_path",
        os.path.join(grasp_berry_dir, "left", _model_file)
    )
    p._left_berry_scaler_path = rospy.get_param(
        "~left_berry_scaler_path",
        os.path.join(grasp_berry_dir, "left", "x_scaler.pkl")
    )
    p._right_berry_model_path = rospy.get_param(
        "~right_berry_model_path",
        os.path.join(grasp_berry_dir, "right", _model_file)
    )
    p._right_berry_scaler_path = rospy.get_param(
        "~right_berry_scaler_path",
        os.path.join(grasp_berry_dir, "right", "x_scaler.pkl")
    )

    p._pre_dist = float(rospy.get_param("~pre_dist", 0.12))
    p._delta_rz_deg = float(rospy.get_param("~delta_rz_deg", 40.0))

    p._use_pose_compensation = _to_bool(rospy.get_param("~use_pose_compensation", True))
    p._tilt_deg = float(rospy.get_param("~tilt_deg", 0.0))

    p._r_x_offset = float(rospy.get_param("~r_x_offset", 0.03))
    p._l_x_offset = float(rospy.get_param("~l_x_offset", 0.035))
    p._r_y_offset = float(rospy.get_param("~r_y_offset", 0.015))
    p._l_y_offset = float(rospy.get_param("~l_y_offset", -0.043))
    p._r_z_offset = float(rospy.get_param("~r_z_offset", 0.01))
    p._l_z_offset = float(rospy.get_param("~l_z_offset", 0.01))
    p._pose_comp_cfg = PoseCompensationConfig(
        use_compensation=p._use_pose_compensation,
        tilt_deg=p._tilt_deg,
        r_x_offset=p._r_x_offset,
        r_y_offset=p._r_y_offset,
        r_z_offset=p._r_z_offset,
        l_x_offset=p._l_x_offset,
        l_y_offset=p._l_y_offset,
        l_z_offset=p._l_z_offset,
    )

    p._lift_dz = float(rospy.get_param("~lift_dz", 0.10))

    p._retract_after_grasp = _to_bool(rospy.get_param("~retract_after_grasp", True))
    p._retract_x_target = float(rospy.get_param("~retract_x_target", 0.15))

    p._post_grasp_home_roll_offset_deg = {
        "left":  float(rospy.get_param("~post_grasp_home_left_roll_offset_deg",   7.0)),
        "right": float(rospy.get_param("~post_grasp_home_right_roll_offset_deg", -7.0)),
    }
    p._workspace_max_x = float(rospy.get_param("~workspace_max_x", 0.39))
    p._workspace_min_z = float(rospy.get_param("~workspace_min_z", -0.02))
    p._left_home_joints_deg = rospy.get_param(
        "~left_home_joints_deg",
        [20.1, 13.0, 0.0, 49.8, 0.0],
    )
    p._right_home_joints_deg = rospy.get_param(
        "~right_home_joints_deg",
        [20.1, -13.0, 0.0, 49.8, 0.0],
    )
    p._init_to_home_on_start = _to_bool(rospy.get_param("~init_to_home_on_start", True))

    # -------------------------------------------------
    # A* planner params
    # -------------------------------------------------
    p._use_astar_for_grasp = _to_bool(rospy.get_param("~use_astar_for_grasp", True))
    p._use_astar_for_go_home = _to_bool(rospy.get_param("~use_astar_for_go_home", False))
    p._astar_start_from_current = _to_bool(rospy.get_param("~astar_start_from_current", True))
    p._astar_pregrasp_fine_step_deg = float(
        rospy.get_param("~astar_pregrasp_fine_step_deg", 1.0)
    )
    p._astar_merge_joint_thresh_rad = float(
        rospy.get_param("~astar_merge_joint_thresh_rad", 0.03)
    )

    # -------------------------------------------------
    # A* planner: voxel-based obstacle-aware backend
    # -------------------------------------------------
    p._voxel_output_dir = rospy.get_param("~voxel_output_dir", "")
    p._voxel_depth_topic = rospy.get_param(
        "~voxel_depth_topic", "/camera/aligned_depth_to_color/image_raw"
    )
    p._voxel_camera_info_topic = rospy.get_param(
        "~voxel_camera_info_topic", "/camera/color/camera_info"
    )
    p._voxel_depth_scale = float(rospy.get_param("~voxel_depth_scale", 0.001))
    p._voxel_min_depth = float(rospy.get_param("~voxel_min_depth", 0.15))
    p._voxel_max_depth = float(rospy.get_param("~voxel_max_depth", 2.5))
    p._voxel_subsample_stride = int(
        rospy.get_param("~voxel_subsample_stride", 4)
    )
    p._voxel_camera_translation = list(rospy.get_param(
        "~voxel_camera_translation", [0.0576235, 0.01753, 0.42987]
    ))
    p._voxel_camera_rotation_ypr = list(rospy.get_param(
        "~voxel_camera_rotation_ypr", [0.0, 0.8307767239493009, 0.0]
    ))
    p._voxel_obstacle_frame_offset = list(rospy.get_param(
        "~voxel_obstacle_frame_offset", [0.0, 0.0, 0.0]
    ))
    p._voxel_defer_joint = int(rospy.get_param("~voxel_defer_joint", 2))
    p._voxel_defer_cost_mult = float(rospy.get_param("~voxel_defer_cost_mult", 2.0))
    p._voxel_h_weight = float(rospy.get_param("~voxel_h_weight", 1.0))
    p._voxel_shortcut = _to_bool(rospy.get_param("~voxel_shortcut", False))
    p._voxel_warmup_on_start = _to_bool(
        rospy.get_param("~voxel_warmup_on_start", False)
    )
    p._voxel_publish_obstacles = _to_bool(
        rospy.get_param("~voxel_publish_obstacles", True)
    )
    p._voxel_obstacles_topic = rospy.get_param(
        "~voxel_obstacles_topic", "/g1_arm_simple/voxel_obstacles"
    )
    p._voxel_obstacles_frame = rospy.get_param(
        "~voxel_obstacles_frame", "torso_link"
    )
    p._voxel_obstacles_first_only = _to_bool(
        rospy.get_param("~voxel_obstacles_first_only", False)
    )
    p._voxel_obstacles_live_rate = float(
        rospy.get_param("~voxel_obstacles_live_rate", 5.0)
    )
    p._voxel_obstacles_live_arm = str(
        rospy.get_param("~voxel_obstacles_live_arm", "right")
    ).lower().strip()
    p._voxel_obstacle_inflation_cells = int(
        rospy.get_param("~voxel_obstacle_inflation_cells", 0)
    )
    p._voxel_obstacle_history_frames = int(
        rospy.get_param("~voxel_obstacle_history_frames", 3)
    )
    p._voxel_obstacle_history_min_obs = int(
        rospy.get_param("~voxel_obstacle_history_min_obs", 1)
    )
    p._voxel_obstacle_depth_median_filter = _to_bool(
        rospy.get_param("~voxel_obstacle_depth_median_filter", True)
    )
    p._voxel_fresh_depth_timeout_sec = float(
        rospy.get_param("~voxel_fresh_depth_timeout_sec", 1.0)
    )
    p._debug_print_both_paths = _to_bool(
        rospy.get_param("~debug_print_both_paths", False)
    )

    # -------------------------------------------------
    # Hand services (per-arm names + legacy single-bus fallback)
    # -------------------------------------------------
    legacy_pre_grasp = str(rospy.get_param("~pre_grasp_service_name", "") or "").strip()
    legacy_grasp_5f  = str(rospy.get_param("~grasp_5f_service_name", "") or "").strip()
    legacy_release   = str(rospy.get_param("~release_service_name", "") or "").strip()

    p._pre_grasp_service_name_right = rospy.get_param(
        "~pre_grasp_service_name_right",
        legacy_pre_grasp or "/g1_hand_right/pre_grasp",
    )
    p._pre_grasp_service_name_left = rospy.get_param(
        "~pre_grasp_service_name_left",
        legacy_pre_grasp or "/g1_hand_left/pre_grasp",
    )
    p._grasp_5f_service_name_right = rospy.get_param(
        "~grasp_5f_service_name_right",
        legacy_grasp_5f or "/g1_hand_right/grasp_5f",
    )
    p._grasp_5f_service_name_left = rospy.get_param(
        "~grasp_5f_service_name_left",
        legacy_grasp_5f or "/g1_hand_left/grasp_5f",
    )
    p._release_service_name_right = rospy.get_param(
        "~release_service_name_right",
        legacy_release or "/g1_hand_right/release",
    )
    p._release_service_name_left = rospy.get_param(
        "~release_service_name_left",
        legacy_release or "/g1_hand_left/release",
    )

    p._point_gesture_service_name_right = rospy.get_param(
        "~point_gesture_service_name_right",
        "/g1_hand_right/point_gesture",
    )
    p._point_gesture_service_name_left = rospy.get_param(
        "~point_gesture_service_name_left",
        "/g1_hand_left/point_gesture",
    )

    # Berry-grasp hand service (per-arm). Distinct from grasp_5f because a
    # berry needs a small-object finger shape, not the bottle 5-finger grip.
    # g1_hands does not implement this yet; these are the names the picker
    # will call once it does.
    p._grasp_berry_service_name_right = rospy.get_param(
        "~grasp_berry_service_name_right",
        "/g1_hand_right/grasp_berry",
    )
    p._grasp_berry_service_name_left = rospy.get_param(
        "~grasp_berry_service_name_left",
        "/g1_hand_left/grasp_berry",
    )
    # Berry approach (pre-grasp) hand service, per-arm. Distinct from the
    # bottle pre_grasp because the berry approach pose differs.
    p._pre_grasp_berry_service_name_right = rospy.get_param(
        "~pre_grasp_berry_service_name_right",
        "/g1_hand_right/pre_grasp_berry",
    )
    p._pre_grasp_berry_service_name_left = rospy.get_param(
        "~pre_grasp_berry_service_name_left",
        "/g1_hand_left/pre_grasp_berry",
    )

    # Elevator-button press params
    p._press_elevator_default_pre_press_dx = float(
        rospy.get_param("~press_elevator_button_default_pre_press_dx", 0.05)
    )
    p._press_elevator_post_settle_sec = float(
        rospy.get_param("~press_elevator_button_post_settle_sec", 1.5)
    )
    p._press_elevator_default_retract_dx = float(
        rospy.get_param("~press_elevator_button_default_retract_dx", 0.15)
    )
    p._press_elevator_hand_settle_sec = float(
        rospy.get_param("~press_elevator_button_hand_settle_sec", 1.0)
    )
    # x offset is SUBTRACTED from the press target like y/z. Since +x is
    # forward into the button, a NEGATIVE value presses that many metres deeper
    # than the detected face. Default -0.005 = press 0.5 cm further forward on
    # both arms.
    p._press_elevator_r_x_offset = float(
        rospy.get_param("~press_elevator_button_r_x_offset", -0.005)
    )
    p._press_elevator_l_x_offset = float(
        rospy.get_param("~press_elevator_button_l_x_offset", -0.005)
    )
    p._press_elevator_r_y_offset = float(
        rospy.get_param("~press_elevator_button_r_y_offset", -0.008)
    )
    p._press_elevator_l_y_offset = float(
        rospy.get_param("~press_elevator_button_l_y_offset", -0.008)
    )
    p._press_elevator_r_z_offset = float(
        rospy.get_param("~press_elevator_button_r_z_offset", 0.003)
    )
    p._press_elevator_l_z_offset = float(
        rospy.get_param("~press_elevator_button_l_z_offset", 0.003)
    )
    p._press_workspace_max_x = float(
        rospy.get_param("~press_elevator_button_workspace_max_x",
                        p._workspace_max_x)
    )
    p._press_workspace_min_z = float(
        rospy.get_param("~press_elevator_button_workspace_min_z",
                        p._workspace_min_z)
    )

    # Legacy single-name aliases (right-arm defaults).
    p._pre_grasp_service_name = p._pre_grasp_service_name_right
    p._grasp_5f_service_name  = p._grasp_5f_service_name_right
    p._release_service_name   = p._release_service_name_right
    p._hand_service_timeout = float(rospy.get_param("~hand_service_timeout", 3.0))
    p._release_non_blocking_on_home = _to_bool(
        rospy.get_param("~release_non_blocking_on_home", True)
    )
    p._hand_settle_time = float(rospy.get_param("~hand_settle_time", 0.0))
    p._grasp_5f_retries = int(rospy.get_param("~grasp_5f_retries", 3))
    p._grasp_retry_interval = float(rospy.get_param("~grasp_retry_interval", 0.4))
    p._post_grasp_hold_time = float(rospy.get_param("~post_grasp_hold_time", 1.0))
    p._lift_after_grasp = _to_bool(rospy.get_param("~lift_after_grasp", True))
    p._clip_inferred_joints_to_limits = _to_bool(
        rospy.get_param("~clip_inferred_joints_to_limits", True)
    )
    p._enable_pregrasp_breakpoint = _to_bool(
        rospy.get_param("~enable_pregrasp_breakpoint", False)
    )
    p._pregrasp_breakpoint_seconds = float(
        rospy.get_param("~pregrasp_breakpoint_seconds", 0.0)
    )
    p._pre_to_target_waypoint_count = int(
        rospy.get_param("~pre_to_target_waypoint_count", 3)
    )

    # Object-position (camera bottle lookup) + grasp_bottle params
    p._object_position_service_name = rospy.get_param(
        "~object_position_service_name",
        "/perception/get_object_position"
    )
    p._object_position_timeout = float(rospy.get_param("~object_position_timeout", 2.0))
    p._grasp_bottle_arm = rospy.get_param("~grasp_bottle_arm", "right").strip().lower()
    p._grasp_bottle_label = rospy.get_param("~grasp_bottle_label", "bottle")
    p._grasp_bottle_index = int(rospy.get_param("~grasp_bottle_index", 0))
    p._grasp_bottle_wait = _to_bool(rospy.get_param("~grasp_bottle_wait", True))
    p._grasp_bottle_timeout = float(
        rospy.get_param("~grasp_bottle_timeout", p._default_timeout)
    )
    p._grasp_bottle_dry_run = _to_bool(rospy.get_param("~grasp_bottle_dry_run", False))

    # Berry pick (camera GetBerries lookup) params
    p._berry_position_service_name = rospy.get_param(
        "~berry_position_service_name",
        "/perception/get_berries"
    )
    p._berry_position_timeout = float(rospy.get_param("~berry_position_timeout", 2.0))
    # Target ripeness to pick ("" = any of red / black / white).
    p._pick_berry_ripeness = str(rospy.get_param("~pick_berry_ripeness", "black")).strip()
    # Arm selection by the target berry's y (base_link, +y = left): pick the
    # left arm when y >= split, else the right arm.
    p._pick_berry_arm_y_split = float(rospy.get_param("~pick_berry_arm_y_split", 0.0))
    p._pick_berry_wait = _to_bool(rospy.get_param("~pick_berry_wait", True))
    p._pick_berry_timeout = float(
        rospy.get_param("~pick_berry_timeout", p._default_timeout)
    )
    p._pick_berry_dry_run = _to_bool(rospy.get_param("~pick_berry_dry_run", False))
    # Whether the berry pick uses the voxel A* pre-path (home -> pregrasp).
    # Default true: route the home->pregrasp leg through the obstacle-aware
    # planner. Set false to skip A* and drive straight to the pregrasp joints via
    # a single MoveArmJoints target.
    p._pick_berry_use_astar = _to_bool(rospy.get_param("~pick_berry_use_astar", True))
    # Goal-reachability tolerance (cells, ~4 deg/cell) for the berry A* pre-path.
    # Relaxed vs the strict grasp default (1) because the pregrasp is only a
    # staging point below the berry -- the exact grasp is reached afterward by
    # the Cartesian leg -- so a loose A* landing is fine and avoids the
    # "BerryPregrasp not reachable" abort. Default 5 (~20 deg on the loosest
    # axis): the A* grid cannot always land within a few cells of the IK
    # pregrasp (e.g. shoulder_roll off by 5 cells), and the Cartesian leg
    # corrects the residual before the grasp.
    p._pick_berry_astar_max_cell_diff = int(
        rospy.get_param("~pick_berry_astar_max_cell_diff", 5)
    )
    # Berry-specific workspace floor: berries hang lower than bottles, so the
    # berry pick gates its target against this min_z (default -0.2) instead of
    # the grasp ~workspace_min_z, without relaxing the bottle/press gates.
    p._pick_berry_workspace_min_z = float(
        rospy.get_param("~pick_berry_workspace_min_z", -0.2)
    )
    # Berry-specific forward reach limit: berries can sit farther out than the
    # grasp ~workspace_max_x, so the berry pick gates against this max_x (default
    # 0.42) instead, without relaxing the bottle/press gates.
    p._pick_berry_workspace_max_x = float(
        rospy.get_param("~pick_berry_workspace_max_x", 0.42)
    )
    # Berry pick has its OWN pose compensation (per-arm offsets SUBTRACTED from
    # the target + a forward tilt), independent of the bottle grasp
    # (~r_/~l_{x,y,z}_offset) and the press offsets. Default identity (all zero,
    # tilt 0): the grasp_berry MLP already places the tip on the detected berry,
    # so no bottle-style pullback is wanted. Tune per arm if a residual bias
    # shows up (measure it with test/check_berry_ik_error.py). Positive offset ->
    # target pulled back toward the body / down; negative -> further out / up.
    p._pick_berry_use_pose_compensation = _to_bool(
        rospy.get_param("~pick_berry_use_pose_compensation", True)
    )
    p._pick_berry_tilt_deg = float(rospy.get_param("~pick_berry_tilt_deg", 0.0))
    p._pick_berry_r_x_offset = float(rospy.get_param("~pick_berry_r_x_offset", 0.0))
    p._pick_berry_r_y_offset = float(rospy.get_param("~pick_berry_r_y_offset", -0.005))
    p._pick_berry_r_z_offset = float(rospy.get_param("~pick_berry_r_z_offset", 0.0))
    p._pick_berry_l_x_offset = float(rospy.get_param("~pick_berry_l_x_offset", -0.005))
    p._pick_berry_l_y_offset = float(rospy.get_param("~pick_berry_l_y_offset", -0.035))
    p._pick_berry_l_z_offset = float(rospy.get_param("~pick_berry_l_z_offset", -0.01))
    p._berry_pose_comp_cfg = PoseCompensationConfig(
        use_compensation=p._pick_berry_use_pose_compensation,
        tilt_deg=p._pick_berry_tilt_deg,
        r_x_offset=p._pick_berry_r_x_offset,
        r_y_offset=p._pick_berry_r_y_offset,
        r_z_offset=p._pick_berry_r_z_offset,
        l_x_offset=p._pick_berry_l_x_offset,
        l_y_offset=p._pick_berry_l_y_offset,
        l_z_offset=p._pick_berry_l_z_offset,
    )
    # Berry pregrasp is a straight-DOWN approach (its own staging, not the
    # bottle's fanned pre_dist/delta_rz): the pregrasp sits directly below the
    # berry at the same x,y by this distance (m), then the arm rises into it.
    p._pick_berry_pregrasp_dz = float(
        rospy.get_param("~pick_berry_pregrasp_dz", 0.05)
    )
    # Settle pause (s) after the arm is commanded to the grasp pose and before
    # the berry grasp hand is triggered. A fixed sleep instead of verifying the
    # arm reached (avoids the stale-state wait issue on repeated picks).
    p._pick_berry_grasp_settle_sec = float(
        rospy.get_param("~pick_berry_grasp_settle_sec", 0.7)
    )
    # Berry retract pull-back x target (m). The single retract waypoint slides
    # back along the wrist yaw to this x (y computed from rz, z = grasp z), via
    # make_retract_pose -- same geometry as the bottle grasp -- then the arm
    # returns to its home joints.
    p._pick_berry_retract_x_target = float(
        rospy.get_param("~pick_berry_retract_x_target", 0.15)
    )
    # After the berry grasp, pull the hand straight back (no lift) to
    # ~retract_x_target at the grasp height before finishing.
    p._pick_berry_retract = _to_bool(rospy.get_param("~pick_berry_retract", True))

    return p
