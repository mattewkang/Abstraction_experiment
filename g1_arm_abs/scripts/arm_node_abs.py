#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
g1_arm_abs/scripts/arm_node_abs.py

功能：
1. move_arm_joints
   输入 5 个关节角，控制左/右臂运动到目标关节角

2. infer_grasp_from_point
   输入目标点 x,y,z，通过 NN 推理出：
   - pregrasp_pose
   - grasp_pose
   - pregrasp_joint_angles
   - grasp_joint_angles

3. execute_grasp_from_point
   输入目标点 x,y,z，
   自动执行：
   - move to pregrasp_joint_angles
   - call /g1_hand/pre_grasp
   - move to grasp_joint_angles
   - call /g1_hand/grasp_5f
   - move to lift pose [0.3, 0, 0.15]

4. GoHome
   双臂都回到 home 位姿，home 位姿的关节角可配置，默认为： [20.1, 11.2, 0.0, 49.8, 0.0], [20.1, -11.2, 0.0, 49.8, 0.0]

5. grasp_bottle
   调用 g1_camera 的 GetObjectPosition.srv 获取 bottle 坐标，
   然后执行与 execute_grasp_from_point 相同的抓取流程

6. execute_astar_path_from_point
   输入目标点 x,y,z，先由 NN 推理 pregrasp/grasp joint，
   再用 voxel A* 规划并按 waypoint 执行
"""

import logging
import math
import os
import time
from dataclasses import replace

import numpy as np
import rospy

from g1_camera.srv import GetObjectPosition, GetBerries
from std_srvs.srv import Trigger, TriggerResponse

from g1_arm_abs import ArmController, ARM_DOF
from g1_arm_abs.arm_controller_proc import ArmControllerProxy
from g1_arm_abs.utils import (
    rad_list_to_deg,
)
from g1_arm_abs.path_ops import (
    flatten_path,
    interpolate_path_deg,
    downsample_waypoints_keep_last,
)
from g1_arm_abs.joint_validation import (
    WorkspaceBounds,
    validate_joint_request,
    check_target_workspace,
    check_joint_limits,
    clip_joints_to_limits,
)
from g1_arm_abs.pose_compensation import (
    PoseCompensationConfig,
    compensate_pose,
)
from g1_arm_abs.nn_backend import (
    make_device,
    load_ai_tools,
    solve_ai_ik,
)
from g1_arm_abs.grasp_plan_math import (
    build_horizontal_pre_to_target_waypoints,
    infer_grasp_joints,
    make_retract_pose,
)
from g1_arm_abs.hand_services import (
    call_trigger_service,
    call_trigger_service_with_retry,
    release_both_hands,
    release_single_hand,
)
from g1_arm_abs.motion_exec import (
    wait_until_reached,
    wait_until_dual_reached,
    execute_joint_target,
    execute_waypoint_path,
    execute_dual_waypoint_paths,
)
from g1_arm_abs.astar_planning import (
    plan_astar_segments,
    plan_astar_path_between_joints,
    debug_log_voxel_vs_no_obstacles,
)
from g1_arm_abs.homing import (
    HomingContext,
    go_home_dual,
    go_home_arm,
)
from g1_arm_abs.press_planning import (
    PressContext,
    plan_press_waypoints,
)
from g1_arm_abs.press_pipeline import (
    PressExecContext,
    execute_press,
)
from g1_arm_abs.grasp_pipeline import (
    GraspContext,
    plan_grasp,
    execute_grasp,
)
from g1_arm_abs.async_dispatch import (
    fire_release_background,
    fire_delayed_point_gesture,
)
from g1_arm_abs.camera_lookup import get_object_position
from g1_arm_abs.berry_lookup import get_berries, nearest_berry
from g1_arm_abs.params import load_params
from g1_arm_abs.pose_planning import (
    PoseContext,
    plan_to_pose,
)
# Obstacle-aware voxel A* backend. build_voxel_adapter imports
# VoxelPlannerAdapter eagerly so syntax errors in the adapter surface at
# launch time; the heavy VoxelObstacleAStar loads on first use.
from g1_arm_abs.voxel_setup import build_voxel_adapter
from g1_arm_abs.solve_grasp_pose import (
    compute_grasp_and_pregrasp as _shared_compute_grasp_and_pregrasp,
)
from g1_arm_abs.weight_verify import verify_pinned_weights
from g1_arm_abs.srv import (
    MoveArmJoints,
    MoveArmJointsRequest,
    MoveArmJointsResponse,
    InferGraspFromPoint,
    InferGraspFromPointResponse,
    ExecuteGraspFromPoint,
    ExecuteGraspFromPointResponse,
    ExecuteAstarPathFromPoint,
    ExecuteAstarPathFromPointResponse,
    PlanGraspFromPoint,
    PlanGraspFromPointRequest,
    PlanGraspFromPointResponse,
    PlanToPose,
    PlanToPoseRequest,
    PlanToPoseResponse,
    GoHomeArm,
    GoHomeArmRequest,
    GoHomeArmResponse,
    PressElevatorButton,
    PressElevatorButtonResponse,
)

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _ensure_ros_home_writable():
    """Use a writable ROS_HOME when ~/.ros is on a read-only mount."""
    ros_home = os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))
    log_dir = os.path.join(ros_home, "log")

    try:
        os.makedirs(log_dir, exist_ok=True)
        if os.access(ros_home, os.W_OK) and os.access(log_dir, os.W_OK):
            return
    except OSError:
        pass

    fallback_ros_home = "/tmp/ros_home"
    os.makedirs(os.path.join(fallback_ros_home, "log"), exist_ok=True)
    os.environ["ROS_HOME"] = fallback_ros_home


class ArmNodeAbs:
    def __init__(self):
        _ensure_ros_home_writable()
        rospy.init_node("g1_arm_simple")

        debug = _to_bool(rospy.get_param("~debug", False))
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level)
        self._logger = logging.getLogger("g1_arm_simple")

        verify_pinned_weights(
            force_sha256_all=_to_bool(
                rospy.get_param("~force_full_weight_sha256", False)
            )
        )

        # -------------------------------------------------
        # ROS parameters (read centrally in g1_arm_abs.params.load_params)
        # -------------------------------------------------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(script_dir)
        self.__dict__.update(vars(load_params(pkg_root)))
        self._voxel_adapter = None  # set by _init_voxel_planner()

        self._logger.info(f"inference backend: PyTorch | device: {self._device}")
        self._logger.info(
            f"pose compensation: tilt_deg={self._tilt_deg} | "
            f"right(x={self._r_x_offset:+.4f}, y={self._r_y_offset:+.4f}, z={self._r_z_offset:+.4f}) | "
            f"left (x={self._l_x_offset:+.4f}, y={self._l_y_offset:+.4f}, z={self._l_z_offset:+.4f})"
        )
        self._logger.info(f"lift: pure-vertical dz={self._lift_dz:+.4f} m")
        self._logger.info(f"left model path: {self._left_model_path}")
        self._logger.info(f"left scaler path: {self._left_scaler_path}")
        self._logger.info(f"right model path: {self._right_model_path}")
        self._logger.info(f"right scaler path: {self._right_scaler_path}")

        # -------------------------------------------------
        # 底层 controller：继续沿用你已经跑通的版本
        # -------------------------------------------------
        # ~controller_subprocess=true isolates the 250 Hz DDS publish loop in
        # a child process (GIL isolation from vision/inference/service
        # threads); simulation mode always uses the in-process controller.
        if self._controller_subprocess and not self._simulation_mode:
            controller_cls = ArmControllerProxy
            self._logger.info("Using subprocess ArmController (GIL-isolated)")
        else:
            controller_cls = ArmController
        self._controller = controller_cls(
            motion_mode=self._motion_mode,
            simulation_mode=self._simulation_mode,
            network_interface=self._network_interface,
            control_frequency=self._control_frequency,
            velocity_limit=self._velocity_limit,
            control_switch_delay=self._control_switch_delay,
            control_switch_settle=self._control_switch_settle,
        )

        # -------------------------------------------------
        # 加载 NN
        # -------------------------------------------------
        self._left_model,  self._left_scaler  = self._load_ai_tools(
            self._left_model_path,  self._left_scaler_path,
            self._left_hidden_dims,  self._left_dropout,
        )
        self._right_model, self._right_scaler = self._load_ai_tools(
            self._right_model_path, self._right_scaler_path,
            self._right_hidden_dims, self._right_dropout,
        )

        # -------------------------------------------------
        # Berry-pick NN (separate weight tree; used only by ~pick_berry).
        # Loaded with raise_on_missing=False so a missing berry weight cannot
        # abort startup for the bottle-grasp / elevator-press paths; the berry
        # IK call raises a clear error at pick time if the model is absent.
        # -------------------------------------------------
        self._logger.info(f"left berry model path:  {self._left_berry_model_path}")
        self._logger.info(f"right berry model path: {self._right_berry_model_path}")
        self._left_berry_model,  self._left_berry_scaler  = self._load_berry_ai_tools(
            self._left_berry_model_path,  self._left_berry_scaler_path,
            self._left_hidden_dims,  self._left_dropout,
        )
        self._right_berry_model, self._right_berry_scaler = self._load_berry_ai_tools(
            self._right_berry_model_path, self._right_berry_scaler_path,
            self._right_hidden_dims, self._right_dropout,
        )

        # -------------------------------------------------
        # Hand services
        # -------------------------------------------------

        # Per-arm clients (right + left). Callers pick with
        # self._pre_grasp_client_for(arm) etc.
        self._pre_grasp_clients = {
            "right": rospy.ServiceProxy(self._pre_grasp_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._pre_grasp_service_name_left,  Trigger),
        }
        self._grasp_5f_clients = {
            "right": rospy.ServiceProxy(self._grasp_5f_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._grasp_5f_service_name_left,  Trigger),
        }
        self._release_clients = {
            "right": rospy.ServiceProxy(self._release_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._release_service_name_left,  Trigger),
        }
        self._point_gesture_clients = {
            "right": rospy.ServiceProxy(self._point_gesture_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._point_gesture_service_name_left,  Trigger),
        }
        self._grasp_berry_clients = {
            "right": rospy.ServiceProxy(self._grasp_berry_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._grasp_berry_service_name_left,  Trigger),
        }
        self._pre_grasp_berry_clients = {
            "right": rospy.ServiceProxy(self._pre_grasp_berry_service_name_right, Trigger),
            "left":  rospy.ServiceProxy(self._pre_grasp_berry_service_name_left,  Trigger),
        }
        # Right-side aliases for the legacy single-client attributes; many
        # downstream call sites still use these to address the right hand
        # implicitly (previous single-hand behaviour).
        self._pre_grasp_client = self._pre_grasp_clients["right"]
        self._grasp_5f_client  = self._grasp_5f_clients["right"]
        self._release_client   = self._release_clients["right"]
        self._object_position_client = rospy.ServiceProxy(
            self._object_position_service_name,
            GetObjectPosition
        )
        self._berry_position_client = rospy.ServiceProxy(
            self._berry_position_service_name,
            GetBerries
        )
        if self._use_astar_for_grasp:
            self._init_voxel_planner()

        # -------------------------------------------------
        # Services
        # -------------------------------------------------
        self._move_srv = rospy.Service(
            "~move_arm_joints",
            MoveArmJoints,
            self._handle_move_arm_joints,
        )

        self._infer_srv = rospy.Service(
            "~infer_grasp_from_point",
            InferGraspFromPoint,
            self._handle_infer_grasp_from_point,
        )

        self._execute_srv = rospy.Service(
            "~execute_grasp_from_point",
            ExecuteGraspFromPoint,
            self._handle_execute_grasp_from_point,
        )
        self._execute_astar_srv = rospy.Service(
            "~execute_astar_path_from_point",
            ExecuteAstarPathFromPoint,
            self._handle_execute_astar_path_from_point,
        )
        self._plan_grasp_srv = rospy.Service(
            "~plan_grasp_from_point",
            PlanGraspFromPoint,
            self._handle_plan_grasp_from_point,
        )
        self._plan_to_pose_srv = rospy.Service(
            "~plan_to_pose",
            PlanToPose,
            self._handle_plan_to_pose,
        )

        self._go_home_srv = rospy.Service(
            "~GoHome",
            Trigger,
            self._handle_go_home,
        )
        self._go_home_arm_srv = rospy.Service(
            "~GoHomeArm",
            GoHomeArm,
            self._handle_go_home_arm,
        )
        self._grasp_bottle_srv = rospy.Service(
            "~grasp_bottle",
            Trigger,
            self._handle_grasp_bottle,
        )
        self._pick_berry_srv = rospy.Service(
            "~pick_berry",
            Trigger,
            self._handle_pick_berry,
        )
        self._press_elevator_button_srv = rospy.Service(
            "~press_elevator_button",
            PressElevatorButton,
            self._handle_press_elevator_button,
        )
        self._acquire_arm_control_srv = rospy.Service(
            "~acquire_arm_control",
            Trigger,
            self._handle_acquire_arm_control,
        )
        self._release_arm_control_srv = rospy.Service(
            "~release_arm_control",
            Trigger,
            self._handle_release_arm_control,
        )
        self._move_arm_service_name = rospy.resolve_name("~move_arm_joints")
        self._move_arm_client = rospy.ServiceProxy(
            self._move_arm_service_name,
            MoveArmJoints
        )

        self._logger.info("ArmNodeAbs initialized")

    def _init_voxel_planner(self):
        """Construct the obstacle-aware voxel planner adapter.

        VoxelObstacleAStar itself is loaded lazily on first plan(), so the
        adapter mostly just wires up the ROS subscriptions. Delegated to
        voxel_setup.build_voxel_adapter; pkg_root is derived from THIS
        script's path so the bitmap / decompress_once.py paths resolve.
        """
        pkg_root = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(pkg_root)  # scripts/ -> package root
        self._voxel_adapter = build_voxel_adapter(
            pkg_root,
            self._logger,
            voxel_output_dir=self._voxel_output_dir,
            depth_topic=self._voxel_depth_topic,
            camera_info_topic=self._voxel_camera_info_topic,
            depth_scale=self._voxel_depth_scale,
            min_depth=self._voxel_min_depth,
            max_depth=self._voxel_max_depth,
            subsample_stride=self._voxel_subsample_stride,
            camera_translation=self._voxel_camera_translation,
            camera_rotation_ypr=self._voxel_camera_rotation_ypr,
            obstacle_frame_offset=self._voxel_obstacle_frame_offset,
            defer_joint=self._voxel_defer_joint,
            defer_cost_mult=self._voxel_defer_cost_mult,
            h_weight=self._voxel_h_weight,
            shortcut=self._voxel_shortcut,
            publish_obstacles=self._voxel_publish_obstacles,
            obstacles_topic=self._voxel_obstacles_topic,
            obstacles_frame=self._voxel_obstacles_frame,
            publish_first_only=self._voxel_obstacles_first_only,
            inflation_cells=self._voxel_obstacle_inflation_cells,
            live_publish_rate=self._voxel_obstacles_live_rate,
            live_publish_arm=self._voxel_obstacles_live_arm,
            history_frames=self._voxel_obstacle_history_frames,
            history_min_obs=self._voxel_obstacle_history_min_obs,
            depth_median_filter=self._voxel_obstacle_depth_median_filter,
            fresh_depth_timeout_sec=self._voxel_fresh_depth_timeout_sec,
            warmup_on_start=self._voxel_warmup_on_start,
        )

    def _flatten_path(self, path_rad):
        return flatten_path(path_rad)

    def _get_home_joints_rad(self, arm):
        home_deg = self._left_home_joints_deg if arm == "left" else self._right_home_joints_deg
        home = np.radians(np.array(home_deg, dtype=float).reshape(-1))
        if len(home) != ARM_DOF:
            raise ValueError(
                f"~{arm}_home_joints_deg must have {ARM_DOF} values, got {len(home)}"
            )
        return home

    def _get_start_joint_for_astar(self, arm):
        if self._astar_start_from_current:
            q_now = self._get_current_arm_q(arm)
            if q_now is not None and len(q_now) == ARM_DOF and np.all(np.isfinite(q_now)):
                return np.array(q_now, dtype=float)
        return np.zeros(ARM_DOF, dtype=float)

    def _plan_astar_segments(self, arm, q_start_rad, q_pregrasp_rad, q_target_rad,
                             max_cell_diff=1, task_label="Pregrasp",
                             ignore_obstacles=False):
        return plan_astar_segments(
            self._voxel_adapter, arm, q_start_rad, q_pregrasp_rad, q_target_rad,
            fine_step_deg=self._astar_pregrasp_fine_step_deg,
            debug_print_both_paths=self._debug_print_both_paths,
            logger=self._logger,
            max_cell_diff=max_cell_diff, task_label=task_label,
            ignore_obstacles=ignore_obstacles,
        )

    def _plan_astar_segments_berry(self, arm, q_start_rad, q_pregrasp_rad,
                                   q_target_rad):
        # Berry A* pre-path (home -> pregrasp), planned WITHOUT any voxel obstacle
        # input (ignore_obstacles=True): the A* runs on the bare reachability grid
        # and ignores the depth/obstacle voxels entirely. The pregrasp is only a
        # staging point below the berry, so the relaxed goal tolerance still
        # applies (the no-obstacle planner only warns on an unreachable goal).
        return self._plan_astar_segments(
            arm, q_start_rad, q_pregrasp_rad, q_target_rad,
            max_cell_diff=self._pick_berry_astar_max_cell_diff,
            task_label="BerryPregrasp",
            ignore_obstacles=True,
        )

    def _debug_log_voxel_vs_no_obstacles(
        self, adapter, arm, q_start_deg, q_pregrasp_deg, voxel_path_deg, voxel_info,
    ):
        return debug_log_voxel_vs_no_obstacles(
            adapter, arm, q_start_deg, q_pregrasp_deg, voxel_path_deg, voxel_info,
            logger=self._logger,
        )

    def _interpolate_path_deg(self, path_deg, fine_step_deg=1.0):
        return interpolate_path_deg(path_deg, fine_step_deg=fine_step_deg)

    def _build_horizontal_pre_to_target_waypoints(
        self,
        arm,
        pregrasp_pose,
        grasp_pose,
        grasp_joints,
        waypoint_count=None,
        solve_ik=None,
    ):
        if waypoint_count is None:
            waypoint_count = self._pre_to_target_waypoint_count
        # solve_ik defaults to the bottle-grasp IK; the berry context injects
        # _build_horizontal_pre_to_target_waypoints_berry so this Cartesian
        # pregrasp->grasp leg is solved with the berry model too.
        if solve_ik is None:
            solve_ik = self._solve_ai_ik_robust
        return build_horizontal_pre_to_target_waypoints(
            arm,
            pregrasp_pose,
            grasp_pose,
            grasp_joints,
            waypoint_count,
            solve_ik=solve_ik,
            check_limits=self._check_joint_limits,
            merge_thresh_rad=self._astar_merge_joint_thresh_rad,
            logger=self._logger,
        )

    def _build_horizontal_pre_to_target_waypoints_berry(
        self,
        arm,
        pregrasp_pose,
        grasp_pose,
        grasp_joints,
        waypoint_count=None,
    ):
        # Berry Cartesian pregrasp->grasp leg: same builder, berry IK model.
        return self._build_horizontal_pre_to_target_waypoints(
            arm,
            pregrasp_pose,
            grasp_pose,
            grasp_joints,
            waypoint_count=waypoint_count,
            solve_ik=self._solve_ai_ik_robust_berry,
        )

    def _downsample_waypoints_keep_last(self, waypoints_rad, stride):
        return downsample_waypoints_keep_last(waypoints_rad, stride)

    def _plan_astar_path_between_joints(self, arm, q_start_rad, q_goal_rad,
                                        append_exact_goal=False,
                                        max_cell_diff=1,
                                        task_label="Pregrasp"):
        return plan_astar_path_between_joints(
            self._voxel_adapter, arm, q_start_rad, q_goal_rad,
            merge_thresh_rad=self._astar_merge_joint_thresh_rad,
            append_exact_goal=append_exact_goal,
            max_cell_diff=max_cell_diff,
            task_label=task_label,
            logger=self._logger,
        )

    def _execute_waypoint_path(self, arm, waypoints_rad, wait=True, timeout=None):
        return execute_waypoint_path(
            arm, waypoints_rad,
            self._controller, self._get_current_arm_q,
            self._position_tolerance, self._default_timeout,
            wait=wait, timeout=timeout,
        )

    def _wait_until_dual_reached(self, q_left_target, q_right_target, timeout):
        return wait_until_dual_reached(
            q_left_target, q_right_target, timeout,
            self._get_current_arm_q, self._position_tolerance,
        )

    def _execute_dual_waypoint_paths(self, left_waypoints_rad, right_waypoints_rad, wait=True, timeout=None):
        return execute_dual_waypoint_paths(
            left_waypoints_rad, right_waypoints_rad,
            self._controller, self._get_current_arm_q,
            self._position_tolerance, self._default_timeout,
            wait=wait, timeout=timeout,
        )

    # =========================================================
    # start / shutdown
    # =========================================================
    def start(self):
        if not self._controller.start(
            connect_retries=self._connect_retries,
            connect_retry_interval=self._connect_retry_interval,
        ):
            self._logger.error("Failed to start ArmController")
            return False

        if self._init_to_home_on_start:
            if not self._controller.acquire_control():
                self._logger.error("Init-to-home: failed to take over arm control")
                return False
            # Use direct joint targets (no A*) for the init move. A* would
            # grid-snap the rest pose and run voxel avoidance, both manifest
            # as a visible detour. Also CRITICAL: the controller's internal
            # _q_target is initialized to zeros for both arms. If we send
            # left-home first then right-home, the right arm tracks zero-pose
            # (velocity-clipped) for ~1 s — visible as "right arm moves
            # forward, then reverses to home". Command BOTH arms at once so
            # neither tracks zero.
            try:
                left_home = self._get_home_joints_rad("left")
                right_home = self._get_home_joints_rad("right")
                q_dual = np.concatenate(
                    [np.asarray(left_home, dtype=float),
                     np.asarray(right_home, dtype=float)]
                )
                self._controller.ctrl_dual_arm(q_dual)
                ok_dual, q_left_now, q_right_now, msg_dual = (
                    self._wait_until_dual_reached(
                        left_home, right_home, self._default_timeout,
                    )
                )
                if not ok_dual:
                    # Non-fatal: a tolerance miss / motion timeout on the
                    # startup home move must not take down the node and every
                    # /g1_arm_simple service with it. Log and continue so the
                    # planning + service surface comes up regardless of where
                    # the arms settled.
                    self._logger.warning(
                        f"Init-to-home(dual) did not reach tolerance: {msg_dual}; "
                        "continuing startup"
                    )
                else:
                    self._logger.info("Init-to-home completed (dual direct, no A*)")
            except Exception as e:
                self._logger.exception(f"Init-to-home exception: {e}")
                return False
            if self._release_arm_control_after_init_home:
                # Hand the arms back to the robot's own controller until the
                # first motion service takes them over again.
                self._controller.release_control()

        self._logger.info("ArmNodeAbs started")
        return True

    def shutdown(self):
        self._logger.info("Shutting down ArmNodeAbs...")
        if self._controller and self._controller.is_running:
            self._controller.stop(go_home=False, release_control=True)
        self._logger.info("ArmNodeAbs shutdown complete")

    # =========================================================
    # 基础 joint movement
    # =========================================================
    def _validate_joint_request(self, arm, joint_angles):
        return validate_joint_request(arm, joint_angles)

    def _get_current_arm_q(self, arm):
        if arm == "left":
            return self._controller.get_left_arm_q()
        return self._controller.get_right_arm_q()

    def _send_target(self, arm, q_target):
        if arm == "left":
            self._controller.ctrl_left_arm(q_target)
        else:
            self._controller.ctrl_right_arm(q_target)

    def _wait_until_reached(self, arm, q_target, timeout):
        return wait_until_reached(
            arm, q_target, timeout,
            self._get_current_arm_q, self._position_tolerance,
        )

    def _execute_joint_target(self, arm, q_target, wait=True, timeout=None):
        return execute_joint_target(
            arm, q_target,
            self._controller, self._get_current_arm_q,
            self._position_tolerance, self._default_timeout,
            wait=wait, timeout=timeout,
        )

    # =========================================================
    # SDK takeover / hand-back
    # =========================================================
    def _ensure_arm_control(self):
        """Take the arms over from the robot's own controller if needed.

        Every motion service calls this first. The takeover ramps the
        arm_sdk weight 0->1 over ~control_switch_delay seconds and then
        holds ~control_switch_settle seconds before the first target is
        issued (no-op when already held). Returns (ok, message).
        """
        if self._controller.has_control:
            return True, "arm control already held"
        if self._controller.acquire_control():
            return True, "arm control acquired"
        return False, "failed to take over arm control"

    def _handle_acquire_arm_control(self, request):
        if not self._controller.is_running:
            return TriggerResponse(success=False, message="Controller not running")
        ok, msg = self._ensure_arm_control()
        return TriggerResponse(success=ok, message=msg)

    def _arms_at_home_error(self):
        """Return (ok, detail). ok is True when both arms measure within
        ~release_arm_control_home_tolerance (per-joint max abs, rad) of
        their ~{arm}_home_joints_deg; detail names the worst arm/error."""
        worst_arm, worst_err = None, -1.0
        for arm in ("left", "right"):
            q_now = self._get_current_arm_q(arm)
            if q_now is None:
                return False, f"{arm} arm state unavailable"
            err = float(np.max(np.abs(
                np.asarray(q_now, dtype=float) - self._get_home_joints_rad(arm)
            )))
            if err > worst_err:
                worst_arm, worst_err = arm, err
        ok = worst_err <= self._release_arm_control_home_tolerance
        return ok, (
            f"{worst_arm} arm max joint error {worst_err:.3f} rad "
            f"(tolerance {self._release_arm_control_home_tolerance:.3f})"
        )

    def _handle_release_arm_control(self, request):
        """Hand the arms back to the robot's own controller (weight 1->0
        over ~control_switch_delay seconds, then a ~control_switch_settle
        hold before replying). Refuses while either arm is away from its
        home joints (~release_arm_control_require_home) so the robot's own
        arm swing never takes over an extended or loaded arm; the
        orchestrator must home the arm first."""
        if not self._controller.is_running:
            return TriggerResponse(success=False, message="Controller not running")
        if not self._controller.has_control:
            return TriggerResponse(success=True, message="arm control already released")
        if self._release_arm_control_require_home and not self._simulation_mode:
            at_home, detail = self._arms_at_home_error()
            if not at_home:
                self._logger.warning(
                    f"release_arm_control refused: arm not at home ({detail})"
                )
                return TriggerResponse(
                    success=False, message=f"arm not at home, keeping control: {detail}"
                )
        if self._controller.release_control():
            return TriggerResponse(success=True, message="arm control released")
        return TriggerResponse(success=False, message="failed to release arm control")

    def _handle_move_arm_joints(self, request):
        response = MoveArmJointsResponse()

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            response.current_joint_angles = []
            return response
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            response.success = False
            response.message = msg_ctrl
            response.current_joint_angles = []
            return response

        ok, msg, arm, q_target = self._validate_joint_request(
            request.arm,
            request.joint_angles
        )
        if not ok:
            response.success = False
            response.message = msg
            response.current_joint_angles = []
            return response

        try:
            self._logger.info(f"Received {arm} target joints: {q_target.tolist()}")

            success, q_current, message = self._execute_joint_target(
                arm=arm,
                q_target=q_target,
                wait=request.wait,
                timeout=request.timeout,
            )
            response.success = success
            response.message = message
            response.current_joint_angles = q_current.tolist()
            return response

        except Exception as e:
            self._logger.exception("move_arm_joints failed")
            response.success = False
            response.message = f"Exception: {e}"
            response.current_joint_angles = []
            return response

    # =========================================================
    # GoHome
    # =========================================================
    def _homing_ctx(self):
        return HomingContext(
            logger=self._logger,
            default_timeout=self._default_timeout,
            use_astar_for_go_home=self._use_astar_for_go_home,
            release_non_blocking_on_home=self._release_non_blocking_on_home,
            left_home_joints_deg=self._left_home_joints_deg,
            right_home_joints_deg=self._right_home_joints_deg,
            get_home_joints_rad=self._get_home_joints_rad,
            get_start_joint_for_astar=self._get_start_joint_for_astar,
            plan_astar_path_between_joints=self._plan_astar_path_between_joints,
            fire_release_background=self._fire_release_background,
            release_both_hands=self._release_both_hands,
            release_single_hand=self._release_single_hand,
            execute_dual_waypoint_paths=self._execute_dual_waypoint_paths,
            execute_joint_target=self._execute_joint_target,
            execute_waypoint_path=self._execute_waypoint_path,
        )

    def _go_home_impl(self, wait=True, timeout=None, do_release=True):
        return go_home_dual(self._homing_ctx(), wait=wait, timeout=timeout, do_release=do_release)

    def _handle_go_home(self, request):
        """
        双臂都回到配置的 home joints，并执行手部 release
        """
        if not self._controller.is_running:
            return TriggerResponse(
                success=False,
                message="Controller not running"
            )
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            return TriggerResponse(success=False, message=msg_ctrl)

        try:
            self._logger.info("Received GoHome request")
            success, message = self._go_home_impl(
                wait=True,
                timeout=self._default_timeout,
                do_release=True,
            )
            return TriggerResponse(success=success, message=message)

        except Exception as e:
            self._logger.exception("GoHome failed")
            return TriggerResponse(
                success=False,
                message=f"Exception: {e}"
            )

    def _go_home_arm_impl(self, arm, wait=True, timeout=None, do_release=True,
                          use_astar=None):
        return go_home_arm(
            self._homing_ctx(), arm,
            wait=wait, timeout=timeout, do_release=do_release, use_astar=use_astar,
        )

    def _handle_go_home_arm(self, request):
        """
        单臂回 home joints，只释放该侧手。未使用的另一侧保持原状。
        """
        if not self._controller.is_running:
            return GoHomeArmResponse(
                success=False,
                message="Controller not running",
            )
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            return GoHomeArmResponse(success=False, message=msg_ctrl)

        arm = str(getattr(request, "arm", "")).strip().lower()
        if arm not in ("left", "right"):
            return GoHomeArmResponse(
                success=False,
                message=f"invalid arm '{request.arm}'; expected 'left' or 'right'",
            )

        astar_override = int(getattr(request, "astar_override", 0) or 0)
        if astar_override == GoHomeArmRequest.ASTAR_FORCE_ON:
            use_astar = True
        elif astar_override == GoHomeArmRequest.ASTAR_FORCE_OFF:
            use_astar = False
        else:
            use_astar = None  # respect ~use_astar_for_go_home

        try:
            self._logger.info(
                f"Received GoHomeArm request arm={arm} "
                f"astar_override={astar_override}"
            )
            success, message = self._go_home_arm_impl(
                arm=arm,
                wait=True,
                timeout=self._default_timeout,
                do_release=True,
                use_astar=use_astar,
            )
            return GoHomeArmResponse(success=success, message=message)

        except Exception as e:
            self._logger.exception("GoHomeArm failed")
            return GoHomeArmResponse(
                success=False,
                message=f"Exception: {e}",
            )

    # =========================================================
    # Hand service helpers
    # =========================================================
    def _call_trigger_service(self, service_name, client, wait_available=True):
        return call_trigger_service(
            service_name, client, self._hand_service_timeout,
            wait_available=wait_available,
        )

    def _call_trigger_service_with_retry(
        self,
        service_name,
        client,
        retries,
        interval,
        wait_available=True,
    ):
        return call_trigger_service_with_retry(
            service_name, client, retries, interval, self._hand_service_timeout,
            wait_available=wait_available,
        )

    # --- Per-arm hand-service accessors -------------------------------------
    def _pre_grasp_service_for(self, arm):
        return (
            self._pre_grasp_service_name_right if arm == "right"
            else self._pre_grasp_service_name_left
        )

    def _pre_grasp_client_for(self, arm):
        return self._pre_grasp_clients.get(arm, self._pre_grasp_clients["right"])

    def _grasp_5f_service_for(self, arm):
        return (
            self._grasp_5f_service_name_right if arm == "right"
            else self._grasp_5f_service_name_left
        )

    def _grasp_5f_client_for(self, arm):
        return self._grasp_5f_clients.get(arm, self._grasp_5f_clients["right"])

    def _grasp_berry_service_for(self, arm):
        return (
            self._grasp_berry_service_name_right if arm == "right"
            else self._grasp_berry_service_name_left
        )

    def _grasp_berry_client_for(self, arm):
        return self._grasp_berry_clients.get(arm, self._grasp_berry_clients["right"])

    def _pre_grasp_berry_service_for(self, arm):
        return (
            self._pre_grasp_berry_service_name_right if arm == "right"
            else self._pre_grasp_berry_service_name_left
        )

    def _pre_grasp_berry_client_for(self, arm):
        return self._pre_grasp_berry_clients.get(
            arm, self._pre_grasp_berry_clients["right"])

    def _release_service_names(self):
        return {
            "right": self._release_service_name_right,
            "left": self._release_service_name_left,
        }

    def _release_both_hands(self):
        return release_both_hands(self._release_clients, self._release_service_names())

    def _release_single_hand(self, side):
        return release_single_hand(side, self._release_clients, self._release_service_names())

    def _fire_release_background(self, release_fn, label):
        fire_release_background(release_fn, label, self._logger)

    def _fire_delayed_point_gesture(self, arm, delay_sec):
        fire_delayed_point_gesture(
            arm,
            delay_sec,
            self._point_gesture_service_for(arm),
            self._point_gesture_client_for(arm),
            self._call_trigger_service,
            self._logger,
        )

    def _get_object_position(self, label, index):
        return get_object_position(
            self._object_position_client,
            self._object_position_service_name,
            self._object_position_timeout,
            label,
            index,
        )

    def _get_berries(self, ripeness=""):
        return get_berries(
            self._berry_position_client,
            self._berry_position_service_name,
            self._berry_position_timeout,
            ripeness,
        )

    # =========================================================
    # NN helpers
    # =========================================================
    def _load_ai_tools(self, model_path, scaler_path, hidden_dims=None, dropout=0.0):
        return load_ai_tools(
            model_path, scaler_path, self._device,
            hidden_dims=hidden_dims, dropout=dropout,
            raise_on_missing=True,
        )

    def _load_berry_ai_tools(self, model_path, scaler_path, hidden_dims=None, dropout=0.0):
        # raise_on_missing=False: a missing berry weight must NOT abort startup
        # for the bottle-grasp / elevator-press paths. _solve_ai_ik_berry raises
        # at pick time if the model is absent.
        return load_ai_tools(
            model_path, scaler_path, self._device,
            hidden_dims=hidden_dims, dropout=dropout,
            raise_on_missing=False,
        )

    def _compensate_pose(self, x, y, z, hand="right"):
        return compensate_pose(x, y, z, hand, self._pose_comp_cfg)

    def _check_target_workspace(self, target, max_x=None, min_z=None):
        # max_x / min_z default to the grasp-policy bounds
        # (~workspace_max_x / ~workspace_min_z). The press pipeline passes its
        # own ~press_elevator_button_workspace_{max_x,min_z} so the press
        # fingertip can reach farther forward than the grasp palm without
        # relaxing the grasp gate.
        if max_x is None:
            max_x = self._workspace_max_x
        if min_z is None:
            min_z = self._workspace_min_z

        return check_target_workspace(target, WorkspaceBounds(max_x=max_x, min_z=min_z))

    def _check_target_workspace_berry(self, target):
        # Berry pick uses a lower floor (~pick_berry_workspace_min_z) than the
        # grasp ~workspace_min_z because berries hang lower, and a farther
        # ~pick_berry_workspace_max_x because berries can sit farther out.
        return self._check_target_workspace(
            target, max_x=self._pick_berry_workspace_max_x,
            min_z=self._pick_berry_workspace_min_z,
        )

    def _compute_grasp_and_pregrasp(self, target, hand="right"):
        # Single source of truth: g1_arm_abs.solve_grasp_pose.compute_grasp_and_pregrasp.
        # rospy params still control delta_rz_deg / pre_dist on the real robot.
        grasp_pose, pregrasp_pose = _shared_compute_grasp_and_pregrasp(
            target,
            hand=hand,
            delta_rz_deg=self._delta_rz_deg,
            pre_dist=self._pre_dist,
        )
        return (np.array(grasp_pose, dtype=float),
                np.array(pregrasp_pose, dtype=float))

    def _compute_grasp_and_pregrasp_berry(self, target, hand="right"):
        # Berry grasp orientation differs from the bottle grasp: the wrist tilts
        # to rx = +pi/2 (right) / -pi/2 (left) instead of the bottle rx = 0. The
        # grasp pose xyz (= the detected berry) and the rz rule come from the
        # shared geometry; rx is overridden on both poses.
        #
        # Berry pregrasp is its OWN straight-DOWN staging (NOT the bottle's fanned
        # pre_dist/delta_rz): place the pregrasp directly below the berry at the
        # same x,y by ~pick_berry_pregrasp_dz, then rise into it. Positional
        # accuracy is handled by _berry_pose_comp_cfg in _solve_ai_ik_berry --
        # the berry path uses NO bottle ~r_/~l_ offset or press offset.
        grasp_pose, pregrasp_pose = self._compute_grasp_and_pregrasp(
            target, hand=hand
        )
        rx = math.pi / 2.0 if hand == "right" else -math.pi / 2.0
        grasp_pose[3] = rx
        pregrasp_pose[3] = rx
        # Override the pregrasp to sit straight below the berry with the SAME
        # orientation as the grasp. The shared compute leaves the pregrasp yaw at
        # the bottle's fanned pre_rz_base (atan2(y+/-0.07, x)), which for a berry
        # straddling the +/-0.07 vs +/-0.17 band can flip sign relative to the
        # grasp rz (atan2(y+/-0.17, x)) and throw the IK shoulder_roll off-grid.
        # A directly-below pregrasp must share the grasp's rx AND rz.
        tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
        pregrasp_pose[0] = tx
        pregrasp_pose[1] = ty
        pregrasp_pose[2] = tz - self._pick_berry_pregrasp_dz
        pregrasp_pose[4] = grasp_pose[4]
        return grasp_pose, pregrasp_pose

    def _solve_ai_ik(self, pose5, hand="right"):
        return solve_ai_ik(
            pose5,
            hand,
            {"left": self._left_model, "right": self._right_model},
            {"left": self._left_scaler, "right": self._right_scaler},
            self._pose_comp_cfg,
            self._device,
        )

    def _check_joint_limits(self, arm, q):
        return check_joint_limits(arm, q)

    def _clip_joints_to_limits(self, arm, q):
        return clip_joints_to_limits(arm, q)

    def _solve_ai_ik_robust(self, pose5, hand="right"):
        # Use one NN inference only; joint-limit validation is handled by callers.
        return self._solve_ai_ik(pose5, hand=hand)

    def _solve_ai_ik_berry(self, pose5, hand="right"):
        # Berry-pick IK: same device as _solve_ai_ik but a CLEAN berry pipeline
        # -- the grasp_berry weight tree AND the berry-specific pose compensation
        # (~pick_berry_{r,l}_{x,y,z}_offset / ~pick_berry_tilt_deg), NOT the
        # bottle grasp_bottle weights or the bottle ~r_/~l_ offsets and NOT any
        # press offset. Used only by the ~pick_berry path. Raises RuntimeError if
        # the berry model failed to load (see solve_ai_ik).
        joints = solve_ai_ik(
            pose5,
            hand,
            {"left": self._left_berry_model, "right": self._right_berry_model},
            {"left": self._left_berry_scaler, "right": self._right_berry_scaler},
            self._berry_pose_comp_cfg,
            self._device,
        )
        # Log the EXACT MLP input (xyz after berry pose compensation, rx/rz
        # unchanged) and the resulting joints, so the grasp/pregrasp IK can be
        # inspected against the A* / planner cells.
        if self._berry_pose_comp_cfg.use_compensation:
            mx, my, mz = compensate_pose(pose5[0], pose5[1], pose5[2], hand,
                                         self._berry_pose_comp_cfg)
        else:
            mx, my, mz = float(pose5[0]), float(pose5[1]), float(pose5[2])
        rospy.logwarn(
            "[berry IK] hand=%s MLP in (x,y,z,rx,rz)="
            "[%.4f, %.4f, %.4f, %.4f, %.4f] -> joints(deg)=%s",
            hand, mx, my, mz, float(pose5[3]), float(pose5[4]),
            [round(math.degrees(float(j)), 2) for j in joints],
        )
        return joints

    def _solve_ai_ik_robust_berry(self, pose5, hand="right"):
        # Berry counterpart to _solve_ai_ik_robust (single NN inference;
        # joint-limit validation handled by callers).
        return self._solve_ai_ik_berry(pose5, hand=hand)

    # =========================================================
    # infer only
    # =========================================================
    def _handle_infer_grasp_from_point(self, request):
        response = InferGraspFromPointResponse()

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response

            target = np.array([request.x, request.y, request.z], dtype=float)
            if not np.all(np.isfinite(target)):
                response.success = False
                response.message = "target contains NaN or Inf"
                return response

            res = infer_grasp_joints(
                arm, target,
                compute_grasp_and_pregrasp=self._compute_grasp_and_pregrasp,
                solve_ik=self._solve_ai_ik_robust,
                check_limits=self._check_joint_limits,
                clip_limits=self._clip_joints_to_limits,
                clip_enabled=self._clip_inferred_joints_to_limits,
            )
            pregrasp_pose = res.pregrasp_pose
            grasp_pose = res.grasp_pose
            pregrasp_joints = res.pregrasp_joints
            grasp_joints = res.grasp_joints
            clip_messages = res.clip_messages

            if res.clip_errors:
                response.success = False
                response.message = "failed to clip inferred joints | " + " | ".join(
                    res.clip_errors
                )
                response.pregrasp_pose = pregrasp_pose.tolist()
                response.grasp_pose = grasp_pose.tolist()
                response.pregrasp_joint_angles = pregrasp_joints.tolist()
                response.grasp_joint_angles = grasp_joints.tolist()
                return response

            if not res.pre_ok or not res.grasp_ok:
                details = []
                if not res.pre_ok:
                    details.append("pregrasp: " + "; ".join(res.pre_violations))
                if not res.grasp_ok:
                    details.append("grasp: " + "; ".join(res.grasp_violations))
                response.success = False
                response.message = "inference produced out-of-limit joints | " + " | ".join(details)
                response.pregrasp_pose = pregrasp_pose.tolist()
                response.grasp_pose = grasp_pose.tolist()
                response.pregrasp_joint_angles = pregrasp_joints.tolist()
                response.grasp_joint_angles = grasp_joints.tolist()
                return response

            response.success = True
            response.message = (
                "inference success"
                if not clip_messages
                else "inference success with clipping | " + " | ".join(clip_messages)
            )
            response.pregrasp_pose = pregrasp_pose.tolist()
            response.grasp_pose = grasp_pose.tolist()
            response.pregrasp_joint_angles = pregrasp_joints.tolist()
            response.grasp_joint_angles = grasp_joints.tolist()
            return response

        except Exception as e:
            self._logger.exception("infer_grasp_from_point failed")
            response.success = False
            response.message = f"Exception: {e}"
            response.pregrasp_pose = []
            response.grasp_pose = []
            response.pregrasp_joint_angles = []
            response.grasp_joint_angles = []
            return response

    # =========================================================
    # Shared planning helper (used by both _execute_grasp_pipeline
    # and the plan_grasp_from_point service)
    # =========================================================
    def _grasp_ctx(self):
        return GraspContext(
            logger=self._logger,
            clip_inferred_joints_to_limits=self._clip_inferred_joints_to_limits,
            use_astar_for_grasp=self._use_astar_for_grasp,
            astar_merge_joint_thresh_rad=self._astar_merge_joint_thresh_rad,
            lift_after_grasp=self._lift_after_grasp,
            lift_dz=self._lift_dz,
            retract_after_grasp=self._retract_after_grasp,
            retract_x_target=self._retract_x_target,
            post_grasp_home_roll_offset_deg=self._post_grasp_home_roll_offset_deg,
            hand_service_timeout=self._hand_service_timeout,
            move_arm_service_name=self._move_arm_service_name,
            enable_pregrasp_breakpoint=self._enable_pregrasp_breakpoint,
            pregrasp_breakpoint_seconds=self._pregrasp_breakpoint_seconds,
            hand_settle_time=self._hand_settle_time,
            grasp_5f_retries=self._grasp_5f_retries,
            grasp_retry_interval=self._grasp_retry_interval,
            post_grasp_hold_time=self._post_grasp_hold_time,
            default_timeout=self._default_timeout,
            check_target_workspace=self._check_target_workspace,
            compute_grasp_and_pregrasp=self._compute_grasp_and_pregrasp,
            solve_ai_ik_robust=self._solve_ai_ik_robust,
            check_joint_limits=self._check_joint_limits,
            clip_joints_to_limits=self._clip_joints_to_limits,
            build_horizontal_pre_to_target_waypoints=self._build_horizontal_pre_to_target_waypoints,
            get_start_joint_for_astar=self._get_start_joint_for_astar,
            plan_astar_segments=self._plan_astar_segments,
            get_home_joints_rad=self._get_home_joints_rad,
            execute_waypoint_path=self._execute_waypoint_path,
            execute_joint_target=self._execute_joint_target,
            get_current_arm_q=self._get_current_arm_q,
            call_trigger_service=self._call_trigger_service,
            call_trigger_service_with_retry=self._call_trigger_service_with_retry,
            pre_grasp_service_for=self._pre_grasp_service_for,
            pre_grasp_client_for=self._pre_grasp_client_for,
            grasp_5f_service_for=self._grasp_5f_service_for,
            grasp_5f_client_for=self._grasp_5f_client_for,
            lift_move=self._lift_move,
        )

    def _lift_move(self, arm, joint_angles, wait, timeout):
        return self._move_arm_client(
            MoveArmJointsRequest(
                arm=arm,
                joint_angles=joint_angles,
                wait=wait,
                timeout=timeout,
            )
        )

    def _plan_grasp_from_target(
        self,
        arm,
        target,
        force_astar=None,
        pre_to_target_waypoint_count=None,
        log_tag="plan",
    ):
        return plan_grasp(
            self._grasp_ctx(), arm, target,
            force_astar=force_astar,
            pre_to_target_waypoint_count=pre_to_target_waypoint_count,
            log_tag=log_tag,
        )

    # =========================================================
    # infer + execute + hand services + lift
    # =========================================================
    def _execute_grasp_pipeline(self, arm, target, wait=True, timeout=None, force_astar=None):
        return execute_grasp(
            self._grasp_ctx(), arm, target,
            wait=wait, timeout=timeout, force_astar=force_astar,
        )

    def _berry_grasp_ctx(self):
        # Reuse the grasp context (NN IK -> A* pre-path -> pregrasp->grasp
        # Cartesian leg), swapping the hand poses to the berry-specific services
        # so the fingers form a berry approach/pinch instead of the bottle
        # pre_grasp/5-finger grip. Lift is disabled (no lift for berries); the
        # in-pipeline retract is also off — the berry retract is driven at the
        # handler level by _retract_berry, which mirrors the HANDOVER action
        # (drive to a fixed per-arm delivery joint pose). The workspace gate uses
        # a berry-specific lower floor; A* is toggled by ~pick_berry_use_astar.
        return replace(
            self._grasp_ctx(),
            pre_grasp_service_for=self._pre_grasp_berry_service_for,
            pre_grasp_client_for=self._pre_grasp_berry_client_for,
            grasp_5f_service_for=self._grasp_berry_service_for,
            grasp_5f_client_for=self._grasp_berry_client_for,
            # Berry-specific IK model (grasp_berry tree); bottle keeps its own.
            # Both the pregrasp/grasp IK and the Cartesian pregrasp->grasp leg
            # must use the berry model, so swap the waypoint builder too.
            solve_ai_ik_robust=self._solve_ai_ik_robust_berry,
            build_horizontal_pre_to_target_waypoints=self._build_horizontal_pre_to_target_waypoints_berry,
            # Berry grasp pose tilts the wrist (rx = +/-pi/2) unlike bottle (rx=0).
            compute_grasp_and_pregrasp=self._compute_grasp_and_pregrasp_berry,
            # Berry-specific lower workspace floor (~pick_berry_workspace_min_z).
            check_target_workspace=self._check_target_workspace_berry,
            # Berry A* pre-path (home -> pregrasp) with a relaxed goal tolerance.
            plan_astar_segments=self._plan_astar_segments_berry,
            # Settle a fixed time after the grasp move before closing the hand
            # (instead of verifying the arm reached, which was unreliable on
            # repeated picks).
            grasp_hand_settle_time=self._pick_berry_grasp_settle_sec,
            lift_after_grasp=False,
            retract_after_grasp=False,
        )

    def _execute_berry_pick_pipeline(self, arm, target, wait=True, timeout=None,
                                     force_astar=None):
        return execute_grasp(
            self._berry_grasp_ctx(), arm, target,
            wait=wait, timeout=timeout, force_astar=force_astar,
        )

    def _move_arm_joints_rad(self, arm, joints_rad, wait, timeout, tag):
        """Drive one arm to a 5-DOF joint target (radians) via MoveArmJoints."""
        joints_rad = np.asarray(joints_rad, dtype=float).reshape(-1).tolist()
        try:
            resp = self._move_arm_client(
                MoveArmJointsRequest(
                    arm=arm, joint_angles=joints_rad, wait=wait, timeout=timeout,
                )
            )
        except (rospy.ServiceException, rospy.ROSException) as e:
            return False, f"{tag} move failed: {e}"
        return bool(resp.success), f"{tag}: {resp.message}"

    def _move_berry_pose(self, arm, pose5, wait, timeout, tag):
        """Solve the berry IK for a 5-D pose, clip to limits, and drive there via
        MoveArmJoints. Returns (ok, message)."""
        joints = self._solve_ai_ik_robust_berry(pose5, hand=arm)
        ok, violations = self._check_joint_limits(arm, joints)
        if not ok and self._clip_inferred_joints_to_limits:
            joints, _adj, clip_errs = self._clip_joints_to_limits(arm, joints)
            if clip_errs:
                return False, f"{tag} clip failed | " + " | ".join(clip_errs)
            ok, violations = self._check_joint_limits(arm, joints)
        if not ok:
            return False, f"{tag} joints out-of-limit | " + "; ".join(violations)
        return self._move_arm_joints_rad(arm, joints, wait, timeout, tag)

    def _retract_berry(self, arm, grasp_pose, wait, timeout):
        """Post-grasp retract: ONE waypoint to a pull-back pose, then return to
        the home joint configuration.

        For a negative-yaw grasp (rz < 0) the make_retract_pose pull-back lands in
        an unreachable arm config, so the retract instead goes to a fixed side
        pose: (x=grasp x, y=+/-0.17 by arm, z=grasp z, rz=0), keeping the berry
        wrist rx. For rz >= 0 it uses the make_retract_pose pull-back (slide back
        along the yaw to x=~pick_berry_retract_x_target, y from rz). After the
        retract waypoint (berry IK), drive the arm to its home joints. Returns
        (ok, message)."""
        grasp_pose = np.asarray(grasp_pose, dtype=float)
        tx, tz = float(grasp_pose[0]), float(grasp_pose[2])
        rx, rz = float(grasp_pose[3]), float(grasp_pose[4])

        if rz < 0.0:
            y = 0.17 if arm == "left" else -0.17
            retract_pose = np.array([tx, y, tz, rx, 0.0], dtype=float)
        else:
            retract_pose, meta = make_retract_pose(
                grasp_pose, grasp_pose, tz, self._pick_berry_retract_x_target
            )

        if retract_pose is not None:
            ok, msg = self._move_berry_pose(arm, retract_pose, wait, timeout,
                                            "retract")
            if not ok:
                return False, msg
            note = msg
        else:
            note = f"retract skipped ({meta.get('reason')})"

        # Then return to the home joint configuration for this arm.
        home_rad = self._get_home_joints_rad(arm)
        ok, msg = self._move_arm_joints_rad(arm, home_rad, wait, timeout, "home")
        return ok, f"{note}; {msg}"

    def _handle_execute_grasp_from_point(self, request):
        response = ExecuteGraspFromPointResponse()

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            return response
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            response.success = False
            response.message = msg_ctrl
            return response

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response

            target = np.array([request.x, request.y, request.z], dtype=float)
            if not np.all(np.isfinite(target)):
                response.success = False
                response.message = "target contains NaN or Inf"
                return response

            result = self._execute_grasp_pipeline(
                arm=arm,
                target=target,
                wait=request.wait,
                timeout=request.timeout,
            )
            response.success = result["success"]
            response.message = (
                "execute_grasp_from_point success"
                if result["success"]
                else result["message"]
            )
            response.pregrasp_pose = result["pregrasp_pose"]
            response.grasp_pose = result["grasp_pose"]
            response.pregrasp_joint_angles = result["pregrasp_joint_angles"]
            response.grasp_joint_angles = result["grasp_joint_angles"]
            response.final_joint_angles = result["final_joint_angles"]
            return response

        except Exception as e:
            self._logger.exception("execute_grasp_from_point failed")
            response.success = False
            response.message = f"Exception: {e}"
            response.pregrasp_pose = []
            response.grasp_pose = []
            response.pregrasp_joint_angles = []
            response.grasp_joint_angles = []
            response.final_joint_angles = []
            return response

    def _handle_execute_astar_path_from_point(self, request):
        response = ExecuteAstarPathFromPointResponse()

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            return response
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            response.success = False
            response.message = msg_ctrl
            return response

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response

            target = np.array([request.x, request.y, request.z], dtype=float)
            if not np.all(np.isfinite(target)):
                response.success = False
                response.message = "target contains NaN or Inf"
                return response

            result = self._execute_grasp_pipeline(
                arm=arm,
                target=target,
                wait=request.wait,
                timeout=request.timeout,
                force_astar=True,
            )
            response.success = result["success"]
            response.message = (
                "execute_astar_path_from_point success"
                if result["success"]
                else result["message"]
            )
            response.pregrasp_pose = result["pregrasp_pose"]
            response.grasp_pose = result["grasp_pose"]
            response.pregrasp_joint_angles = result["pregrasp_joint_angles"]
            response.grasp_joint_angles = result["grasp_joint_angles"]
            response.path_joint_angles_flat = result["path_joint_angles_flat"]
            response.waypoint_count = int(result["waypoint_count"])
            response.final_joint_angles = result["final_joint_angles"]
            return response

        except Exception as e:
            self._logger.exception("execute_astar_path_from_point failed")
            response.success = False
            response.message = f"Exception: {e}"
            response.pregrasp_pose = []
            response.grasp_pose = []
            response.pregrasp_joint_angles = []
            response.grasp_joint_angles = []
            response.path_joint_angles_flat = []
            response.waypoint_count = 0
            response.final_joint_angles = []
            return response

    def _handle_plan_grasp_from_point(self, request):
        """
        Plan-only endpoint: returns the full grasp plan (pregrasp/grasp poses,
        NN joints, A* pre-path, Cartesian target-path, optional lift joints)
        WITHOUT moving the arm or touching the hand. main_process uses this
        to execute each waypoint itself via move_arm_joints.
        """
        response = PlanGraspFromPointResponse()

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            return response

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response

            target = np.array([request.x, request.y, request.z], dtype=float)
            if not np.all(np.isfinite(target)):
                response.success = False
                response.message = "target contains NaN or Inf"
                return response

            # Tri-state A* override, matching PlanGraspFromPointRequest constants.
            astar_override = int(getattr(request, "astar_override", 0) or 0)
            if astar_override == PlanGraspFromPointRequest.ASTAR_FORCE_ON:
                force_astar = True
            elif astar_override == PlanGraspFromPointRequest.ASTAR_FORCE_OFF:
                force_astar = False
            else:
                force_astar = None

            wp_count = int(getattr(request, "pre_to_target_waypoint_count", 0) or 0)
            wp_count = None if wp_count <= 0 else wp_count

            plan = self._plan_grasp_from_target(
                arm=arm,
                target=target,
                force_astar=force_astar,
                pre_to_target_waypoint_count=wp_count,
                log_tag="plan",
            )

            # Populate debug fields regardless of plan success.
            if plan["pregrasp_pose"] is not None:
                response.pregrasp_pose = plan["pregrasp_pose"].tolist()
            if plan["grasp_pose"] is not None:
                response.grasp_pose = plan["grasp_pose"].tolist()
            if plan["pregrasp_joints"] is not None:
                response.pregrasp_joint_angles = plan["pregrasp_joints"].tolist()
            if plan["grasp_joints"] is not None:
                response.grasp_joint_angles = plan["grasp_joints"].tolist()

            if not plan["success"]:
                response.success = False
                response.message = plan["message"]
                response.pre_path_flat = []
                response.pre_waypoint_count = 0
                response.target_path_flat = []
                response.target_waypoint_count = 0
                response.lift_joint_angles = []
                response.retract_joint_angles = []
                response.home_joint_angles = []
                return response

            pre_path = plan["pre_path_rad"] or []
            target_path = plan["target_path_rad"] or []

            response.pre_path_flat = self._flatten_path(pre_path)
            response.pre_waypoint_count = int(len(pre_path))
            response.target_path_flat = self._flatten_path(target_path)
            response.target_waypoint_count = int(len(target_path))
            response.lift_joint_angles = (
                plan["lift_joints"].tolist() if plan["lift_joints"] is not None else []
            )
            response.retract_joint_angles = (
                plan["retract_joints"].tolist() if plan["retract_joints"] is not None else []
            )
            response.home_joint_angles = (
                plan["home_joints"].tolist() if plan["home_joints"] is not None else []
            )

            response.success = True
            response.message = plan["message"]
            return response

        except Exception as e:
            self._logger.exception("plan_grasp_from_point failed")
            response.success = False
            response.message = f"Exception: {e}"
            response.pregrasp_pose = []
            response.grasp_pose = []
            response.pregrasp_joint_angles = []
            response.grasp_joint_angles = []
            response.pre_path_flat = []
            response.pre_waypoint_count = 0
            response.target_path_flat = []
            response.target_waypoint_count = 0
            response.lift_joint_angles = []
            response.retract_joint_angles = []
            response.home_joint_angles = []
            return response

    # =========================================================
    # generic plan-to-cartesian-pose
    #
    # Plans a collision-free joint-space path from the current arm pose
    # to an arbitrary 5-D end-effector pose, reusing the same NN IK +
    # A* (voxel-aware) infrastructure as the grasp planner. No grasp /
    # pregrasp / hand action runs here -- this is purely a motion
    # primitive used by the handover leg in main_process.
    # =========================================================
    def _pose_ctx(self):
        return PoseContext(
            logger=self._logger,
            solve_ai_ik=self._solve_ai_ik,
            clip_joints_to_limits=self._clip_joints_to_limits,
            get_current_arm_q=self._get_current_arm_q,
            plan_astar_path_between_joints=self._plan_astar_path_between_joints,
        )

    def _handle_plan_to_pose(self, request):
        response = PlanToPoseResponse()
        response.target_joint_angles = []
        response.target_pose = []
        response.pre_path_flat = []
        response.pre_waypoint_count = 0

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            return response

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response

            x = float(request.x)
            y = float(request.y)
            z = float(request.z)
            if not all(math.isfinite(v) for v in (x, y, z)):
                response.success = False
                response.message = "target xyz contains NaN or Inf"
                return response

            rx_req = float(request.rx)
            rz_req = float(request.rz)

            astar_override = int(getattr(request, "astar_override", 0) or 0)
            if astar_override == PlanToPoseRequest.ASTAR_FORCE_ON:
                force_astar = True
            elif astar_override == PlanToPoseRequest.ASTAR_FORCE_OFF:
                force_astar = False
            else:
                force_astar = self._use_astar_for_grasp

            plan = plan_to_pose(
                self._pose_ctx(), arm, x, y, z, rx_req, rz_req, force_astar
            )
            if not plan.ok:
                response.success = False
                response.message = plan.message
                return response

            path_rad = plan.path_rad
            response.target_joint_angles = plan.target_joints.tolist()
            response.target_pose = plan.pose5.tolist()
            response.pre_path_flat = self._flatten_path(path_rad)
            response.pre_waypoint_count = int(len(path_rad))
            response.success = True
            response.message = (
                f"plan_to_pose ok: arm={arm} pose=({x:.3f},{y:.3f},{z:.3f},"
                f"rx={plan.rx:.3f},rz={plan.rz:.3f}) waypoints={len(path_rad)} "
                f"astar={'on' if force_astar else 'off'} "
                f"yaw={'derived' if plan.derived_yaw else 'override'}"
            )
            self._logger.info(response.message)
            return response

        except Exception as e:
            self._logger.exception("plan_to_pose failed")
            response.success = False
            response.message = f"Exception: {e}"
            return response

    def _handle_grasp_bottle(self, request):
        if not self._controller.is_running:
            return TriggerResponse(success=False, message="Controller not running")
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            return TriggerResponse(success=False, message=msg_ctrl)

        arm = self._grasp_bottle_arm
        if arm not in ("left", "right"):
            return TriggerResponse(
                success=False,
                message=f"invalid ~grasp_bottle_arm: {arm}"
            )

        ok, target, object_msg = self._get_object_position(
            label=self._grasp_bottle_label,
            index=self._grasp_bottle_index,
        )
        if not ok:
            return TriggerResponse(success=False, message=object_msg)

        self._logger.info(
            f"[grasp_bottle] target from camera: {target.tolist()} "
            f"(label={self._grasp_bottle_label}, index={self._grasp_bottle_index})"
        )

        grasp_pose, pregrasp_pose = self._compute_grasp_and_pregrasp(target, hand=arm)
        pregrasp_joints = self._solve_ai_ik_robust(pregrasp_pose, hand=arm)
        grasp_joints = self._solve_ai_ik_robust(grasp_pose, hand=arm)
        pregrasp_joints_deg = rad_list_to_deg(pregrasp_joints)
        grasp_joints_deg = rad_list_to_deg(grasp_joints)

        debug_msg = (
            f"dry_run={self._grasp_bottle_dry_run} | "
            f"target_position={np.round(target, 4).tolist()} | "
            f"pregrasp_position={np.round(pregrasp_pose[:3], 4).tolist()} | "
            f"pregrasp_joint_deg={np.round(pregrasp_joints_deg, 2).tolist()} | "
            f"target_joint_deg={np.round(grasp_joints_deg, 2).tolist()}"
        )
        self._logger.info(f"[grasp_bottle] {debug_msg}")

        if self._grasp_bottle_dry_run:
            return TriggerResponse(success=True, message=debug_msg)

        try:
            result = self._execute_grasp_pipeline(
                arm=arm,
                target=target,
                wait=self._grasp_bottle_wait,
                timeout=self._grasp_bottle_timeout,
            )
        except Exception as e:
            self._logger.exception("grasp_bottle failed")
            return TriggerResponse(success=False, message=f"Exception: {e}")

        if not result["success"]:
            return TriggerResponse(success=False, message=result["message"])

        return TriggerResponse(
            success=True,
            message=(
                f"grasp_bottle success; target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
                f"| object_service: {object_msg}"
            ),
        )

    def _handle_pick_berry(self, request):
        """Pick ONE berry per call. Queries g1_camera/GetBerries for every
        berry, selects the nearest of the target ripeness, chooses the arm by
        the berry's y (base_link, +y = left), and runs the grasp pipeline with
        the berry-specific hand close (A* + pregrasp reused from the bottle
        grasp). One berry per call by design — re-call to pick the next."""
        if not self._controller.is_running:
            return TriggerResponse(success=False, message="Controller not running")

        ripeness = self._pick_berry_ripeness
        ok, berries, berry_msg = self._get_berries(ripeness=ripeness)
        if not ok:
            return TriggerResponse(success=False, message=berry_msg)

        # One berry per call: the nearest of the requested ripeness.
        berry = nearest_berry(berries, ripeness or None)
        if berry is None:
            return TriggerResponse(
                success=False,
                message=(f"no berry to pick (ripeness='{ripeness}', "
                         f"detected={len(berries)})"),
            )

        target = berry.position
        # Arm selection by the berry's y sign (torso_link, +y = left side):
        # y > 0 -> left hand, otherwise right hand.
        arm = "left" if target[1] > 0.0 else "right"

        self._logger.info(
            f"[pick_berry] target={np.round(target, 4).tolist()} "
            f"ripeness={berry.ripeness}({berry.ripeness_score:.2f}) "
            f"det={berry.det_score:.2f} -> arm={arm} (y>0 -> left); "
            f"{len(berries)} detected | berry_service: {berry_msg}"
        )

        if self._pick_berry_dry_run:
            return TriggerResponse(
                success=True,
                message=(f"dry_run | target={np.round(target, 4).tolist()} | "
                         f"arm={arm} | ripeness={berry.ripeness} | "
                         f"berry_service: {berry_msg}"),
            )

        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            return TriggerResponse(success=False, message=msg_ctrl)

        try:
            result = self._execute_berry_pick_pipeline(
                arm=arm,
                target=target,
                wait=self._pick_berry_wait,
                timeout=self._pick_berry_timeout,
                force_astar=self._pick_berry_use_astar,
            )
        except Exception as e:
            self._logger.exception("pick_berry failed")
            return TriggerResponse(success=False, message=f"Exception: {e}")

        if not result["success"]:
            return TriggerResponse(success=False, message=result["message"])

        # Retract: one make_retract_pose waypoint (x=~pick_berry_retract_x_target,
        # y from rz, z=grasp z), then return to home joints.
        retract_note = ""
        if self._pick_berry_retract and result.get("grasp_pose"):
            r_ok, r_msg = self._retract_berry(
                arm, result["grasp_pose"],
                wait=self._pick_berry_wait, timeout=self._pick_berry_timeout,
            )
            retract_note = f" | retract: {r_msg}"
            if not r_ok:
                return TriggerResponse(
                    success=False,
                    message=(f"berry grasped but retract failed; arm={arm} "
                             f"ripeness={berry.ripeness}{retract_note}"),
                )

        return TriggerResponse(
            success=True,
            message=(
                f"pick_berry success; arm={arm} ripeness={berry.ripeness} "
                f"target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
                f"| berry_service: {berry_msg}{retract_note}"
            ),
        )


    # =========================================================
    # Elevator-button press pipeline
    #
    # Kept deliberately separate from _execute_grasp_pipeline:
    #   - uses elevator_button_ik (separate MLP / model files)
    #   - rx is fixed to -90 deg, rz reuses grasp atan2(...) rule
    #   - hand service is /point_gesture (index + middle extended)
    #   - planning reuses _plan_astar_path_between_joints, so the voxel
    #     A* path applies as for the grasp pipeline
    # =========================================================
    def _point_gesture_service_for(self, arm):
        return (
            self._point_gesture_service_name_right if arm == "right"
            else self._point_gesture_service_name_left
        )

    def _point_gesture_client_for(self, arm):
        return self._point_gesture_clients.get(arm, self._point_gesture_clients["right"])

    def _press_ctx(self):
        return PressContext(
            logger=self._logger,
            press_workspace_max_x=self._press_workspace_max_x,
            press_workspace_min_z=self._press_workspace_min_z,
            press_x_offset_left=self._press_elevator_l_x_offset,
            press_x_offset_right=self._press_elevator_r_x_offset,
            press_y_offset_left=self._press_elevator_l_y_offset,
            press_y_offset_right=self._press_elevator_r_y_offset,
            press_z_offset_left=self._press_elevator_l_z_offset,
            press_z_offset_right=self._press_elevator_r_z_offset,
            retract_dx=self._press_elevator_default_retract_dx,
            clip_inferred_joints_to_limits=self._clip_inferred_joints_to_limits,
            check_target_workspace=self._check_target_workspace,
            clip_joints_to_limits=self._clip_joints_to_limits,
            check_joint_limits=self._check_joint_limits,
        )

    def _press_exec_ctx(self):
        return PressExecContext(
            logger=self._logger,
            press_elevator_hand_settle_sec=self._press_elevator_hand_settle_sec,
            press_elevator_post_settle_sec=self._press_elevator_post_settle_sec,
            get_current_arm_q=self._get_current_arm_q,
            plan_astar_path_between_joints=self._plan_astar_path_between_joints,
            flatten_path=self._flatten_path,
            fire_delayed_point_gesture=self._fire_delayed_point_gesture,
            execute_waypoint_path=self._execute_waypoint_path,
            execute_joint_target=self._execute_joint_target,
            get_home_joints_rad=self._get_home_joints_rad,
            release_single_hand=self._release_single_hand,
        )

    def _handle_press_elevator_button(self, request):
        response = PressElevatorButtonResponse()
        response.arm_used = ""
        response.target_xyz = []
        response.rx_rad = 0.0
        response.rz_rad = 0.0
        response.mlp_input_pose5 = []
        response.target_joint_angles = []
        response.pre_press_joint_angles = []
        response.pre_path_flat = []
        response.pre_waypoint_count = 0
        response.press_path_flat = []
        response.press_waypoint_count = 0
        response.final_joint_angles = []

        if not self._controller.is_running:
            response.success = False
            response.message = "Controller not running"
            return response
        ok_ctrl, msg_ctrl = self._ensure_arm_control()
        if not ok_ctrl:
            response.success = False
            response.message = msg_ctrl
            return response

        try:
            arm = request.arm.lower().strip() if isinstance(request.arm, str) else ""
            if not arm:
                arm = "right"
            if arm not in ("left", "right"):
                response.success = False
                response.message = "arm must be 'left' or 'right'"
                return response
            response.arm_used = arm

            target = np.array([request.x, request.y, request.z], dtype=float)
            if not np.all(np.isfinite(target)):
                response.success = False
                response.message = "target xyz contains NaN or Inf"
                return response
            response.target_xyz = target.tolist()

            # pre_press_dx <= 0 in the request -> fall back to the configured
            # default (~press_elevator_button_default_pre_press_dx, 0.05 m).
            pre_press_dx_req = float(getattr(request, "pre_press_dx", 0.0) or 0.0)
            pre_press_dx = (
                pre_press_dx_req if pre_press_dx_req > 1e-4
                else self._press_elevator_default_pre_press_dx
            )
            # The press pipeline is inherently sequential at the leg
            # boundaries (press leg must run *after* pre-path arrival;
            # retract after press; home after retract; release after home).
            # ROS srv bool fields default to False when unset by the caller,
            # which would turn every `_execute_*` call into fire-and-forget
            # and the press leg would chase a still-moving arm. Force
            # wait=True regardless of request.wait so this can't happen.
            # (point_gesture runs in parallel with the A* approach via a
            # delayed daemon thread; it is NOT sequenced behind the arm.)
            wait = True
            timeout = float(getattr(request, "timeout", 0.0) or 0.0)
            if timeout <= 0.0:
                timeout = self._default_timeout

            # Per-arm y/z compensation, workspace gating of all four waypoint
            # families, rx/rz, and the elevator-button MLP IK are computed by
            # the press planning core. response.target_xyz was already set from
            # the RAW request above; the planner compensates an internal copy.
            plan = plan_press_waypoints(
                self._press_ctx(), arm, target, pre_press_dx
            )
            response.rx_rad = plan.rx_rad
            response.rz_rad = plan.rz_rad
            response.mlp_input_pose5 = plan.pose5
            if not plan.ok:
                response.success = False
                response.message = plan.message
                return response

            response.target_joint_angles = plan.press_joints.tolist()
            response.pre_press_joint_angles = plan.pre_press_joints.tolist()

            # A* to pre-press, point-gesture daemon, direct press leg, then
            # settle / retract / home / release. Builds the path + waypoint
            # fields and the final state / message.
            exec_res = execute_press(self._press_exec_ctx(), arm, plan, wait, timeout)
            response.pre_path_flat = exec_res.pre_path_flat
            response.pre_waypoint_count = exec_res.pre_waypoint_count
            response.press_path_flat = exec_res.press_path_flat
            response.press_waypoint_count = exec_res.press_waypoint_count
            response.final_joint_angles = exec_res.final_joint_angles
            response.success = exec_res.ok
            response.message = exec_res.message
            return response

        except Exception as e:
            self._logger.exception("press_elevator_button failed")
            response.success = False
            response.message = f"Exception: {e}"
            return response


def main():
    node = ArmNodeAbs()

    try:
        if node.start():
            rospy.spin()
        else:
            rospy.logerr("Failed to start ArmNodeAbs")
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
