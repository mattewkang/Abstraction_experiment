# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/grasp_plan_math.py

Cartesian geometry for the grasp plan's post-pregrasp legs, extracted from
``arm_node_abs.py``.

ROS-free. The IK and joint-limit checks are injected as callables so this
module never touches the NN models or ROS directly:

- :func:`build_horizontal_pre_to_target_waypoints` interpolates Cartesian
  poses from pregrasp to grasp at (nearly) constant z and solves IK at each.
- :func:`make_lift_pose` builds the pure-vertical lift pose.
- :func:`make_retract_pose` builds the post-lift retract pose (or signals a
  skip), keeping the caller responsible for IK, clipping, and logging.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from g1_arm_abs.joint_index import ARM_DOF
from g1_arm_abs.utils import merge_close_points_rad

__all__ = [
    "InferGraspResult",
    "infer_grasp_joints",
    "build_horizontal_pre_to_target_waypoints",
    "make_lift_pose",
    "make_retract_pose",
]


@dataclass
class InferGraspResult:
    """Result of :func:`infer_grasp_joints`.

    ``*_raw`` are the pre-clip IK joints (callers log these); ``pregrasp_joints``
    / ``grasp_joints`` are the final (post-clip) joints. ``clip_errors`` is
    non-empty only when clipping itself failed (wrong arm / dimension), in which
    case the recheck is skipped and ``pre_ok`` / ``grasp_ok`` stay at their
    pre-clip values.
    """

    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    pregrasp_joints_raw: np.ndarray
    grasp_joints_raw: np.ndarray
    pregrasp_joints: np.ndarray
    grasp_joints: np.ndarray
    clip_messages: List[str] = field(default_factory=list)
    clip_errors: List[str] = field(default_factory=list)
    pre_ok: bool = True
    grasp_ok: bool = True
    pre_violations: List[str] = field(default_factory=list)
    grasp_violations: List[str] = field(default_factory=list)


def infer_grasp_joints(
    arm: str,
    target,
    *,
    compute_grasp_and_pregrasp: Callable,
    solve_ik: Callable,
    check_limits: Callable,
    clip_limits: Callable,
    clip_enabled: bool,
) -> InferGraspResult:
    """Shared NN-IK core for grasp inference: compute the pregrasp/grasp poses,
    solve IK for both, and (when enabled) clip out-of-limit joints and recheck.

    This is the byte-identical mechanic shared by the ``infer_grasp_from_point``
    handler and :func:`~g1_arm_abs.grasp_pipeline.plan_grasp`; both keep their
    own workspace gating, logging, message formatting, and response/plan
    packing. The injected callables match the node's ``_compute_grasp_and_pregrasp``
    / ``_solve_ai_ik_robust`` / ``_check_joint_limits`` / ``_clip_joints_to_limits``.
    """
    grasp_pose, pregrasp_pose = compute_grasp_and_pregrasp(target, hand=arm)
    pregrasp_joints = solve_ik(pregrasp_pose, hand=arm)
    grasp_joints = solve_ik(grasp_pose, hand=arm)
    pregrasp_joints_raw = pregrasp_joints
    grasp_joints_raw = grasp_joints

    pre_ok, pre_violations = check_limits(arm, pregrasp_joints)
    grasp_ok, grasp_violations = check_limits(arm, grasp_joints)
    clip_messages: List[str] = []
    clip_errors: List[str] = []
    if (not pre_ok or not grasp_ok) and clip_enabled:
        pregrasp_joints, pre_adjustments, pre_clip_errors = clip_limits(arm, pregrasp_joints)
        grasp_joints, grasp_adjustments, grasp_clip_errors = clip_limits(arm, grasp_joints)
        clip_errors = pre_clip_errors + grasp_clip_errors
        if not clip_errors:
            if pre_adjustments:
                clip_messages.append("pregrasp clipped: " + "; ".join(pre_adjustments))
            if grasp_adjustments:
                clip_messages.append("grasp clipped: " + "; ".join(grasp_adjustments))
            pre_ok, pre_violations = check_limits(arm, pregrasp_joints)
            grasp_ok, grasp_violations = check_limits(arm, grasp_joints)

    return InferGraspResult(
        pregrasp_pose=pregrasp_pose,
        grasp_pose=grasp_pose,
        pregrasp_joints_raw=pregrasp_joints_raw,
        grasp_joints_raw=grasp_joints_raw,
        pregrasp_joints=pregrasp_joints,
        grasp_joints=grasp_joints,
        clip_messages=clip_messages,
        clip_errors=clip_errors,
        pre_ok=pre_ok,
        grasp_ok=grasp_ok,
        pre_violations=pre_violations,
        grasp_violations=grasp_violations,
    )


def build_horizontal_pre_to_target_waypoints(
    arm: str,
    pregrasp_pose,
    grasp_pose,
    grasp_joints,
    waypoint_count: int,
    solve_ik: Callable,
    check_limits: Callable,
    merge_thresh_rad: float,
    logger=None,
) -> List[List[float]]:
    """Build Cartesian intermediate poses between pregrasp and grasp while
    keeping z as constant as possible, then solve IK for each point.

    ``solve_ik(pose, hand=arm)`` and ``check_limits(arm, q) -> (ok, _)`` are
    injected. The exact grasp joints are always appended as the final waypoint;
    the result is then merged with ``merge_close_points_rad`` using
    ``merge_thresh_rad``. IK / limit failures at an intermediate point are
    logged (if ``logger`` given) and that point is skipped, never fatal.
    """
    waypoint_count = max(0, int(waypoint_count))

    pre = np.array(pregrasp_pose, dtype=float).reshape(-1)
    grasp = np.array(grasp_pose, dtype=float).reshape(-1)
    grasp_q = np.array(grasp_joints, dtype=float).reshape(-1)
    if len(pre) < 5 or len(grasp) < 5 or len(grasp_q) != ARM_DOF:
        return [grasp_q.tolist()]

    if waypoint_count <= 0:
        return [grasp_q.tolist()]

    z_hold = float(pre[2])
    rx = float(grasp[3])
    rz = float(grasp[4])
    waypoints_rad = []

    for i in range(1, waypoint_count + 1):
        t = i / float(waypoint_count + 1)
        x = float(pre[0] + t * (grasp[0] - pre[0]))
        y = float(pre[1] + t * (grasp[1] - pre[1]))
        pose = np.array([x, y, z_hold, rx, rz], dtype=float)

        try:
            q_mid = solve_ik(pose, hand=arm)
            ok, _ = check_limits(arm, q_mid)
            if ok:
                waypoints_rad.append(np.array(q_mid, dtype=float).reshape(-1).tolist())
            elif logger is not None:
                logger.warning(
                    f"[execute] skip invalid intermediate waypoint {i}/{waypoint_count} for {arm}"
                )
        except Exception as e:
            if logger is not None:
                logger.warning(
                    f"[execute] IK failed at intermediate waypoint {i}/{waypoint_count} for {arm}: {e}"
                )

    # Always end at the exact grasp target.
    waypoints_rad.append(grasp_q.tolist())
    return merge_close_points_rad(
        waypoints_rad,
        joint_thresh_rad=merge_thresh_rad,
    )


def make_lift_pose(grasp_pose, lift_dz: float) -> np.ndarray:
    """Pure-vertical lift: keep grasp xy + orientation, add ``lift_dz`` to z."""
    lift_pose = np.array(grasp_pose, dtype=float).copy()
    lift_pose[2] = float(grasp_pose[2]) + float(lift_dz)
    return lift_pose


def make_retract_pose(
    grasp_pose,
    lift_pose,
    target_z: float,
    retract_x_target: float,
) -> Tuple[Optional[np.ndarray], dict]:
    """Build the post-lift retract pose along the opposite-yaw direction.

    Returns ``(retract_pose, meta)``. ``retract_pose`` is ``None`` (skip) when
    the retract direction is perpendicular to x (``|cos(rz)| < 1e-3``) or the
    lift x is already past ``retract_x_target`` (``s <= 0``). ``meta`` carries
    ``reason`` ("perpendicular" / "past_target" / None) plus the computed
    ``rz`` / ``cos_rz`` / ``s`` / ``lift_x`` so the caller can log exactly.
    """
    rx = float(grasp_pose[3])
    rz = float(grasp_pose[4])
    cos_rz = math.cos(rz)
    sin_rz = math.sin(rz)
    lift_x = float(lift_pose[0])
    lift_y = float(lift_pose[1])

    if abs(cos_rz) < 1e-3:
        return None, {"reason": "perpendicular", "rz": rz, "cos_rz": cos_rz}

    s = (lift_x - retract_x_target) / cos_rz
    if s <= 0.0:
        return None, {"reason": "past_target", "s": s, "cos_rz": cos_rz, "lift_x": lift_x}

    retract_pose = np.array(
        [
            retract_x_target,
            lift_y - s * sin_rz,
            target_z,
            rx,
            rz,
        ],
        dtype=float,
    )
    return retract_pose, {"reason": None, "s": s}
