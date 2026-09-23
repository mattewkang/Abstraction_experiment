# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/pose_planning.py

Planning core for the generic plan-to-pose service, extracted from
``arm_node_abs.py``.

Given an arm and a 5-D Cartesian target (with optional rx/rz override), this
derives the wrist yaw, runs NN IK + joint-limit clipping, and plans a path
(voxel A* pre-path or single-target fallback). It returns a :class:`PosePlan`;
the node handler keeps request parsing and ``PlanToPoseResponse`` building.

ROS-free: IK / clip / current-joint reader / A* planner are injected via
:class:`PoseContext`.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from g1_arm_abs.joint_index import ARM_DOF

__all__ = [
    "PoseContext",
    "PosePlan",
    "plan_to_pose",
]


@dataclass
class PoseContext:
    """Dependencies the plan-to-pose core needs from the node."""

    logger: object
    solve_ai_ik: Callable
    clip_joints_to_limits: Callable
    get_current_arm_q: Callable
    plan_astar_path_between_joints: Callable


@dataclass
class PosePlan:
    ok: bool
    message: str = ""
    target_joints: Optional[np.ndarray] = None
    pose5: Optional[np.ndarray] = None
    path_rad: Optional[List] = None
    rx: float = 0.0
    rz: float = 0.0
    derived_yaw: bool = False


def plan_to_pose(
    ctx: PoseContext,
    arm: str,
    x: float,
    y: float,
    z: float,
    rx_req: float,
    rz_req: float,
    force_astar: bool,
) -> PosePlan:
    """Plan a collision-free path to an arbitrary 5-D end-effector pose.

    ``rx_req`` / ``rz_req`` NaN -> derive an NN-style yaw (rx=0, rz from xyz),
    matching ``compute_grasp_and_pregrasp`` so the IK input distribution stays
    aligned with the grasp NN's training data. Returns a :class:`PosePlan`
    (``ok=False`` with a message on any IK / clip / current-state / A* failure).
    """
    if not (math.isfinite(rx_req) and math.isfinite(rz_req)):
        rx = 0.0
        y_offset = 0.16 if arm == "right" else -0.16
        rz = math.atan2(y + y_offset, x)
        derived_yaw = True
    else:
        rx = rx_req
        rz = rz_req
        derived_yaw = False

    pose5 = np.array([x, y, z, rx, rz], dtype=float)

    try:
        target_joints = ctx.solve_ai_ik(pose5, hand=arm)
    except Exception as ik_exc:  # noqa: BLE001
        ctx.logger.exception(f"plan_to_pose: NN IK failed: {ik_exc}")
        return PosePlan(False, f"NN IK failed: {ik_exc}",
                        pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw)

    target_joints = np.array(target_joints, dtype=float).reshape(-1)
    if len(target_joints) != ARM_DOF:
        return PosePlan(
            False,
            f"NN IK returned {len(target_joints)} joints, expected {ARM_DOF}",
            pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw,
        )

    clipped, adjustments, errors = ctx.clip_joints_to_limits(arm, target_joints)
    if errors:
        return PosePlan(
            False, "joint-limit clipping error: " + "; ".join(errors),
            pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw,
        )
    if adjustments:
        ctx.logger.info(
            f"plan_to_pose: clipped joints to limits ({arm}): "
            + "; ".join(adjustments)
        )
    target_joints = clipped

    q_start = ctx.get_current_arm_q(arm)
    if q_start is None:
        return PosePlan(False, f"current {arm} arm joints unavailable",
                        pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw)
    q_start = np.array(q_start, dtype=float).reshape(-1)

    # Path planning: voxel A* pre_path or single-target fallback, mirrors what
    # plan_grasp does for its pre_path leg.
    if force_astar:
        try:
            path_rad = ctx.plan_astar_path_between_joints(
                arm=arm,
                q_start_rad=q_start,
                q_goal_rad=target_joints,
                append_exact_goal=True,
            )
        except Exception as plan_exc:  # noqa: BLE001
            ctx.logger.exception(f"plan_to_pose: A* planning failed: {plan_exc}")
            return PosePlan(False, f"A* planning failed: {plan_exc}",
                            pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw)
        if not path_rad:
            return PosePlan(False, "A* returned empty path",
                            pose5=pose5, rx=rx, rz=rz, derived_yaw=derived_yaw)
    else:
        path_rad = [list(target_joints)]

    return PosePlan(
        ok=True,
        target_joints=target_joints,
        pose5=pose5,
        path_rad=path_rad,
        rx=rx,
        rz=rz,
        derived_yaw=derived_yaw,
    )
