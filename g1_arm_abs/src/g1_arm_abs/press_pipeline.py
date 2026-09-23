# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/press_pipeline.py

Execution orchestration for the elevator-button press, extracted from
``arm_node_abs._handle_press_elevator_button``.

Given a validated :class:`~g1_arm_abs.press_planning.PressPlan` (joints already
solved), this runs the two-leg motion: A* current -> pre-press (relaxed 3-cell
tolerance), the point-gesture daemon fired in parallel, the direct
pre-press -> mid-press[1..N] -> press leg, then settle / retract / home /
release. It returns a :class:`PressExecResult`; the node handler maps it onto
``PressElevatorButtonResponse``.

ROS-light: motion / hand / A* / gesture callables are injected via
:class:`PressExecContext`.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from g1_arm_abs.utils import rad_list_to_deg

__all__ = [
    "PressExecContext",
    "PressExecResult",
    "execute_press",
]


@dataclass
class PressExecContext:
    """Dependencies the press execution leg needs from the node."""

    logger: object
    press_elevator_hand_settle_sec: float
    press_elevator_post_settle_sec: float
    get_current_arm_q: Callable
    plan_astar_path_between_joints: Callable
    flatten_path: Callable
    fire_delayed_point_gesture: Callable
    execute_waypoint_path: Callable
    execute_joint_target: Callable
    get_home_joints_rad: Callable
    release_single_hand: Callable


@dataclass
class PressExecResult:
    ok: bool
    message: str = ""
    pre_path_flat: List[float] = field(default_factory=list)
    pre_waypoint_count: int = 0
    press_path_flat: List[float] = field(default_factory=list)
    press_waypoint_count: int = 0
    final_joint_angles: List[float] = field(default_factory=list)


def execute_press(ctx: PressExecContext, arm, plan, wait, timeout) -> PressExecResult:
    """Run the press motion legs for a solved :class:`PressPlan`.

    Returns a :class:`PressExecResult`; ``ok=False`` with a message on any
    A* / motion / hand failure (the path / waypoint fields populated so far are
    still returned so the caller can surface them).
    """
    result = PressExecResult(ok=False)

    target = plan.press_target
    rx_rad = plan.rx_rad
    rz_rad = plan.rz_rad
    press_joints = plan.press_joints
    pre_press_joints = plan.pre_press_joints
    mid_press_joints_list = plan.mid_press_joints
    retract_joints = plan.retract_joints
    pre_press_dx = plan.pre_press_dx
    retract_dx = plan.retract_dx

    # 5) A* ONLY for current -> pre_press_joints. The press leg is a direct
    # multi-waypoint motion pre_press_joints -> mid_press[1..N] -> press_joints;
    # no A* is run on that segment. The hand gesture fires in a daemon thread
    # with a small delay so the fingers shape into the pointing pose *during*
    # the A* approach swing.
    q_start = ctx.get_current_arm_q(arm)
    if q_start is None:
        result.message = f"current {arm} arm joints unavailable"
        return result
    q_start = np.asarray(q_start, dtype=float).reshape(-1)

    try:
        # Relax the IK-goal-vs-final tolerance for the pre-press approach: the
        # press leg is driven by the exact IK joints, so a 3-cell (~12 deg)
        # landing at the pre-press cell still leaves the finger aimed at the
        # button face. Grasp's pregrasp keeps the strict 1-cell default.
        path_rad = ctx.plan_astar_path_between_joints(
            arm=arm,
            q_start_rad=q_start,
            q_goal_rad=pre_press_joints,
            append_exact_goal=True,
            max_cell_diff=3,
            task_label="Prepress",
        )
    except Exception as plan_exc:  # noqa: BLE001
        ctx.logger.exception(
            f"[press_elevator_button] A* planning to pre-press failed: "
            f"{plan_exc}"
        )
        result.message = f"voxel A* planning failed: {plan_exc}"
        return result

    if not path_rad:
        result.message = "voxel A* returned empty path to pre-press"
        ctx.logger.warning(result.message)
        return result

    ctx.logger.info(
        f"[press_elevator_button] A* current -> pre_press ok | "
        f"waypoints={len(path_rad)} "
        f"start_deg={[round(v, 2) for v in rad_list_to_deg(q_start)]} "
        f"goal_deg={[round(v, 2) for v in rad_list_to_deg(pre_press_joints)]}"
    )
    result.pre_path_flat = ctx.flatten_path(path_rad)
    result.pre_waypoint_count = int(len(path_rad))

    # The press leg path: mid_press waypoints then press.
    press_path = [m.tolist() for m in mid_press_joints_list] + [
        press_joints.tolist()
    ]
    result.press_path_flat = ctx.flatten_path(press_path)
    result.press_waypoint_count = len(press_path)

    # 7) Fire point_gesture in a daemon thread with a small delay so the hand
    # reshapes during the A* approach, then execute the A* pre-path foreground.
    ctx.fire_delayed_point_gesture(
        arm=arm,
        delay_sec=ctx.press_elevator_hand_settle_sec,
    )

    # Execute the A* pre-path: current -> pre_press_joints.
    ok_exec, q_after, exec_msg = ctx.execute_waypoint_path(
        arm=arm,
        waypoints_rad=path_rad,
        wait=wait,
        timeout=timeout,
    )
    ctx.logger.info(
        f"[press_elevator_button] pre-path execute: ok={ok_exec} "
        f"msg='{exec_msg}'"
    )
    if not ok_exec:
        result.message = f"pre-path execution failed: {exec_msg}"
        result.final_joint_angles = q_after.tolist()
        return result

    # 8) Direct press leg: pre_press -> mid_press[1..N] -> press, no A*.
    ok_press_exec, q_after, press_msg = ctx.execute_waypoint_path(
        arm=arm,
        waypoints_rad=press_path,
        wait=wait,
        timeout=timeout,
    )
    ctx.logger.info(
        f"[press_elevator_button] press leg execute "
        f"(pre_press -> {len(mid_press_joints_list)} mid_press -> press, "
        f"+{pre_press_dx:.3f} m in x): "
        f"ok={ok_press_exec} msg='{press_msg}'"
    )
    if not ok_press_exec:
        result.message = f"press leg execution failed: {press_msg}"
        result.final_joint_angles = q_after.tolist()
        return result

    # 9) Post-press: settle, retract (back to pre_press), home, release.
    settle_sec = ctx.press_elevator_post_settle_sec
    if settle_sec > 0:
        ctx.logger.info(
            f"[press_elevator_button] post-press settle {settle_sec:.2f} s"
        )
        time.sleep(settle_sec)

    # Retract: drive the finger clear of the button face along -x by retract_dx
    # before the home leg so the home sweep doesn't drag across the panel.
    ok_retract, q_after, retract_exec_msg = ctx.execute_joint_target(
        arm=arm,
        q_target=retract_joints,
        wait=wait,
        timeout=timeout,
    )
    ctx.logger.info(
        f"[press_elevator_button] post-press retract ({arm}, "
        f"-{retract_dx:.3f} m): ok={ok_retract} msg='{retract_exec_msg}'"
    )
    if not ok_retract:
        result.message = f"post-press retract failed: {retract_exec_msg}"
        result.final_joint_angles = q_after.tolist()
        return result

    home_rad = ctx.get_home_joints_rad(arm)
    ok_home, q_after, home_msg = ctx.execute_joint_target(
        arm=arm,
        q_target=home_rad,
        wait=wait,
        timeout=timeout,
    )
    ctx.logger.info(
        f"[press_elevator_button] post-press home ({arm}): "
        f"ok={ok_home} msg='{home_msg}'"
    )
    if not ok_home:
        result.message = f"post-press home failed: {home_msg}"
        result.final_joint_angles = q_after.tolist()
        return result

    release_ok, release_msg = ctx.release_single_hand(arm)
    ctx.logger.info(
        f"[press_elevator_button] post-press release ({arm}): "
        f"ok={release_ok} msg='{release_msg}'"
    )
    if not release_ok:
        result.message = f"post-press release failed: {release_msg}"
        result.final_joint_angles = q_after.tolist()
        return result

    result.ok = True
    result.message = (
        f"press_elevator_button success | arm={arm} "
        f"press=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) "
        f"pre_press_dx={pre_press_dx:.3f} m "
        f"rx={math.degrees(rx_rad):+.2f}deg "
        f"rz={math.degrees(rz_rad):+.2f}deg "
        f"pre_waypoints={result.pre_waypoint_count} "
        f"press_waypoints={result.press_waypoint_count} "
        f"| post-press: settle={settle_sec:.2f}s retract=ok home=ok "
        f"release='{release_msg}'"
    )
    result.final_joint_angles = q_after.tolist()
    ctx.logger.info(result.message)
    return result
