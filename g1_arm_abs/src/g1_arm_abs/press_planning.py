# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/press_planning.py

Pure planning core for the elevator-button press, extracted from
``arm_node_abs.py``.

Given an arm, a raw button-face target, and a pre-press pull-back distance, this
computes everything up to (but not including) A* / execution:
  - per-arm y/z compensation of the target,
  - workspace gating of press / pre-press / mid-press / retract points,
  - the fixed-per-arm rx and grasp-shared rz,
  - elevator-button MLP IK + joint-limit clip/validate for every waypoint.

It returns a :class:`PressPlan` (``ok`` plus all joints/targets, or a failure
message). The node keeps the A* pre-path planning, motion execution, the hand
point-gesture / release, and the ROS response building. Dependencies are
bundled into :class:`PressContext`; the elevator-button IK helpers are imported
directly (they are ROS-free).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from g1_arm_abs.joint_index import ARM_DOF
from g1_arm_abs.utils import rad_list_to_deg
from g1_arm_abs.elevator_button_ik import (
    compute_rx_rad as elevator_compute_rx_rad,
    compute_rz_rad as elevator_compute_rz_rad,
    get_model_paths as elevator_get_model_paths,
    solve_elevator_button_ik,
)

__all__ = [
    "PressContext",
    "PressPlan",
    "plan_press_waypoints",
]


@dataclass
class PressContext:
    """Dependencies the press planner needs from the node."""

    logger: object
    press_workspace_max_x: float
    press_workspace_min_z: float
    press_x_offset_left: float
    press_x_offset_right: float
    press_y_offset_left: float
    press_y_offset_right: float
    press_z_offset_left: float
    press_z_offset_right: float
    retract_dx: float
    clip_inferred_joints_to_limits: bool
    check_target_workspace: Callable
    clip_joints_to_limits: Callable
    check_joint_limits: Callable


@dataclass
class PressPlan:
    """Result of the press planning core."""

    ok: bool
    message: str = ""
    rx_rad: float = 0.0
    rz_rad: float = 0.0
    pose5: List[float] = field(default_factory=list)
    press_target: Optional[np.ndarray] = None
    press_joints: Optional[np.ndarray] = None
    pre_press_joints: Optional[np.ndarray] = None
    mid_press_joints: List[np.ndarray] = field(default_factory=list)
    retract_joints: Optional[np.ndarray] = None
    pre_press_dx: float = 0.0
    retract_dx: float = 0.0


def plan_press_waypoints(
    ctx: PressContext,
    arm: str,
    raw_target: np.ndarray,
    pre_press_dx: float,
) -> PressPlan:
    """Compute rx/rz + press/pre-press/mid-press/retract joints for the press.

    ``raw_target`` is the button-face xyz as sent by the caller (the node keeps
    the raw value for the response); compensation is applied to an internal
    copy. Returns a :class:`PressPlan` — ``ok=False`` with a message on any
    workspace or IK failure.
    """
    target = np.array(raw_target, dtype=float).reshape(-1)

    # Apply per-arm press x / y / z compensation BEFORE the workspace check and
    # the derived waypoints. All three keep the grasp ~{r,l}_{*}_offset sign
    # convention (subtracted). Because +x points forward into the button face,
    # a NEGATIVE press_x_offset drives the fingertip that many metres deeper
    # than the detected face (a deliberate over-press), so "press 0.5 cm
    # further forward" is press_x_offset = -0.005.
    press_x_offset = (
        ctx.press_x_offset_left if arm == "left" else ctx.press_x_offset_right
    )
    press_y_offset = (
        ctx.press_y_offset_left if arm == "left" else ctx.press_y_offset_right
    )
    press_z_offset = (
        ctx.press_z_offset_left if arm == "left" else ctx.press_z_offset_right
    )
    if press_x_offset != 0.0 or press_y_offset != 0.0 or press_z_offset != 0.0:
        target = target.copy()
        target[0] = target[0] - press_x_offset
        target[1] = target[1] - press_y_offset
        target[2] = target[2] - press_z_offset

    def _check_ws(point):
        return ctx.check_target_workspace(
            point,
            max_x=ctx.press_workspace_max_x,
            min_z=ctx.press_workspace_min_z,
        )

    # 1) Workspace sanity check on the press point (button face).
    in_workspace, workspace_msg = _check_ws(target)
    if not in_workspace:
        ctx.logger.warning(
            f"[press_elevator_button] press point | arm={arm} | {workspace_msg}"
        )
        return PressPlan(ok=False, message=f"press point: {workspace_msg}")

    # 2) Compute rz the same way as the grasp pose; rx is fixed per arm.
    rx_rad = elevator_compute_rx_rad(arm)
    rz_rad = elevator_compute_rz_rad(arm, target.tolist())
    pose5 = [float(target[0]), float(target[1]), float(target[2]),
             float(rx_rad), float(rz_rad)]

    # 3) Pre-press point: pull back along -x by pre_press_dx.
    pre_press_target = np.array(
        [float(target[0]) - pre_press_dx,
         float(target[1]),
         float(target[2])],
        dtype=float,
    )
    in_ws_pre, ws_msg_pre = _check_ws(pre_press_target)
    if not in_ws_pre:
        ctx.logger.warning(
            f"[press_elevator_button] pre-press point | "
            f"arm={arm} | {ws_msg_pre}"
        )
        return PressPlan(
            ok=False, message=f"pre-press point: {ws_msg_pre}",
            rx_rad=float(rx_rad), rz_rad=float(rz_rad), pose5=pose5,
        )

    # Mid-press points: split pre_press -> press into five equal cartesian
    # steps (fractions 1/5..4/5).
    mid_press_fractions = (1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0)
    mid_press_targets = [
        pre_press_target + frac * (target - pre_press_target)
        for frac in mid_press_fractions
    ]
    for idx, mid_xyz in enumerate(mid_press_targets, start=1):
        in_ws_mid, ws_msg_mid = _check_ws(mid_xyz)
        if not in_ws_mid:
            ctx.logger.warning(
                f"[press_elevator_button] mid-press point {idx} | "
                f"arm={arm} | {ws_msg_mid}"
            )
            return PressPlan(
                ok=False, message=f"mid-press point {idx}: {ws_msg_mid}",
                rx_rad=float(rx_rad), rz_rad=float(rz_rad), pose5=pose5,
            )

    # Retract point: deeper -x pull-back than pre-press.
    retract_dx = ctx.retract_dx
    retract_target = np.array(
        [float(target[0]) - retract_dx,
         float(target[1]),
         float(target[2])],
        dtype=float,
    )
    in_ws_retract, ws_msg_retract = _check_ws(retract_target)
    if not in_ws_retract:
        ctx.logger.warning(
            f"[press_elevator_button] retract point | "
            f"arm={arm} | {ws_msg_retract}"
        )
        return PressPlan(
            ok=False, message=f"retract point: {ws_msg_retract}",
            rx_rad=float(rx_rad), rz_rad=float(rz_rad), pose5=pose5,
        )

    model_path, scaler_path = elevator_get_model_paths(arm)
    ctx.logger.info(
        f"[press_elevator_button] arm={arm} "
        f"press=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) "
        f"mid_press=["
        + ", ".join(
            f"({m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f})"
            for m in mid_press_targets
        )
        + "] "
        f"pre_press=({pre_press_target[0]:.4f}, {pre_press_target[1]:.4f}, "
        f"{pre_press_target[2]:.4f}) "
        f"retract=({retract_target[0]:.4f}, {retract_target[1]:.4f}, "
        f"{retract_target[2]:.4f}) "
        f"pre_press_dx={pre_press_dx:.3f} m retract_dx={retract_dx:.3f} m "
        f"rx={np.degrees(rx_rad):+.2f} deg "
        f"rz={np.degrees(rz_rad):+.2f} deg | "
        f"mlp_model='{model_path}' scaler='{scaler_path}'"
    )

    # 4) Solve IK at every waypoint with the dedicated elevator-button MLP.
    def _solve_and_validate(xyz, label):
        """Returns (joints_rad ndarray, err_message_or_None)."""
        try:
            joints = solve_elevator_button_ik(
                arm=arm,
                target_xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                rx_rad=rx_rad,
            )
        except Exception as ik_exc:  # noqa: BLE001
            ctx.logger.exception(
                f"[press_elevator_button] {label} MLP IK failed: {ik_exc}"
            )
            return None, f"{label} IK failed: {ik_exc}"
        joints = np.asarray(joints, dtype=float).reshape(-1)
        if len(joints) != ARM_DOF:
            return None, (
                f"{label} IK returned {len(joints)} joints, "
                f"expected {ARM_DOF}"
            )
        if ctx.clip_inferred_joints_to_limits:
            clipped, adjustments, clip_errors = ctx.clip_joints_to_limits(arm, joints)
            if clip_errors:
                return None, (
                    f"{label} IK clipping error | " + " | ".join(clip_errors)
                )
            if adjustments:
                ctx.logger.warning(
                    f"[press_elevator_button] {label} joints clipped: "
                    + "; ".join(adjustments)
                )
            joints = clipped
        ok_lim, viol = ctx.check_joint_limits(arm, joints)
        if not ok_lim:
            return None, (
                f"{label} IK joints out-of-limit | " + "; ".join(viol)
            )
        return joints, None

    def _fail(msg):
        return PressPlan(
            ok=False, message=msg,
            rx_rad=float(rx_rad), rz_rad=float(rz_rad), pose5=pose5,
        )

    press_joints, err = _solve_and_validate(target, "press")
    if err is not None:
        return _fail(err)
    ctx.logger.info(
        f"[press_elevator_button] press joints (deg) = "
        f"{[round(v, 2) for v in rad_list_to_deg(press_joints)]}"
    )

    pre_press_joints, err = _solve_and_validate(pre_press_target, "pre-press")
    if err is not None:
        return _fail(err)
    ctx.logger.info(
        f"[press_elevator_button] pre_press joints (deg) = "
        f"{[round(v, 2) for v in rad_list_to_deg(pre_press_joints)]} "
        f"| commanded pre_press position (m) = "
        f"({pre_press_target[0]:.4f}, {pre_press_target[1]:.4f}, "
        f"{pre_press_target[2]:.4f})"
    )

    mid_press_joints_list = []
    for idx, mid_xyz in enumerate(mid_press_targets, start=1):
        mid_joints, err = _solve_and_validate(mid_xyz, f"mid-press-{idx}")
        if err is not None:
            return _fail(err)
        mid_press_joints_list.append(mid_joints)
        ctx.logger.info(
            f"[press_elevator_button] mid_press[{idx}] joints (deg) = "
            f"{[round(v, 2) for v in rad_list_to_deg(mid_joints)]}"
        )

    retract_joints, err = _solve_and_validate(retract_target, "retract")
    if err is not None:
        return _fail(err)
    ctx.logger.info(
        f"[press_elevator_button] retract joints (deg) = "
        f"{[round(v, 2) for v in rad_list_to_deg(retract_joints)]} "
        f"(retract_dx={retract_dx:.3f} m)"
    )

    return PressPlan(
        ok=True,
        rx_rad=float(rx_rad),
        rz_rad=float(rz_rad),
        pose5=pose5,
        press_target=target,
        press_joints=press_joints,
        pre_press_joints=pre_press_joints,
        mid_press_joints=mid_press_joints_list,
        retract_joints=retract_joints,
        pre_press_dx=pre_press_dx,
        retract_dx=retract_dx,
    )
