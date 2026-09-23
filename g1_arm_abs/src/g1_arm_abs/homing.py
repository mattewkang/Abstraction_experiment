# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/homing.py

Dual-arm and single-arm GoHome orchestration, extracted from
``arm_node_abs.py``.

The pipeline calls back into many node capabilities (A* planning, hand release,
motion execution) and reads several config values. Rather than a long argument
list, those dependencies are bundled into :class:`HomingContext`, built by the
node and passed in. The functions own no state and add no ROS coupling beyond
what the injected callables already carry.

Order of operations (preserved exactly): plan first (A* when enabled), then
release hands only after planning succeeds (so a failed plan never drops a held
object), then move.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from g1_arm_abs.utils import rad_list_to_deg

__all__ = [
    "HomingContext",
    "go_home_dual",
    "go_home_arm",
]


@dataclass
class HomingContext:
    """Dependencies the homing pipeline needs from the node."""

    logger: object
    default_timeout: float
    use_astar_for_go_home: bool
    release_non_blocking_on_home: bool
    left_home_joints_deg: object
    right_home_joints_deg: object
    get_home_joints_rad: Callable
    get_start_joint_for_astar: Callable
    plan_astar_path_between_joints: Callable
    fire_release_background: Callable
    release_both_hands: Callable
    release_single_hand: Callable
    execute_dual_waypoint_paths: Callable
    execute_joint_target: Callable
    execute_waypoint_path: Callable


def go_home_dual(
    ctx: HomingContext,
    wait: bool = True,
    timeout: Optional[float] = None,
    do_release: bool = True,
) -> Tuple[bool, str]:
    """Move BOTH arms to their configured home joints.

    Order of operations:
      1. Plan the path  (A* dual-arm when enabled; skipped for the
                         direct-target fallback, which has nothing to plan).
      2. Release hands  (only after planning succeeds, so a failed plan
                         doesn't drop a held object).
      3. Move the arms  (execute the path or command the joint targets).
    """
    if timeout is None or timeout <= 0:
        timeout = ctx.default_timeout

    left_home = ctx.get_home_joints_rad("left")
    right_home = ctx.get_home_joints_rad("right")
    release_msg = ""

    if ctx.use_astar_for_go_home:
        # ---- Step 1: plan both arms BEFORE touching hands --------------
        left_start = ctx.get_start_joint_for_astar("left")
        try:
            left_path = ctx.plan_astar_path_between_joints(
                arm="left",
                q_start_rad=left_start,
                q_goal_rad=left_home,
                append_exact_goal=True,
            )
        except Exception as e:
            return False, f"[GoHome] left A* planning failed: {e}"
        ctx.logger.info(
            f"[GoHome] left A* waypoints={len(left_path)} "
            f"start_deg={np.round(rad_list_to_deg(left_start), 2).tolist()} "
            f"goal_deg={np.round(rad_list_to_deg(left_home), 2).tolist()}"
        )

        right_start = ctx.get_start_joint_for_astar("right")
        try:
            right_path = ctx.plan_astar_path_between_joints(
                arm="right",
                q_start_rad=right_start,
                q_goal_rad=right_home,
                append_exact_goal=True,
            )
        except Exception as e:
            return False, f"[GoHome] right A* planning failed: {e}"
        ctx.logger.info(
            f"[GoHome] right A* waypoints={len(right_path)} "
            f"start_deg={np.round(rad_list_to_deg(right_start), 2).tolist()} "
            f"goal_deg={np.round(rad_list_to_deg(right_home), 2).tolist()}"
        )

        # ---- Step 2: release hands now that the plan is in hand --------
        if do_release:
            if ctx.release_non_blocking_on_home:
                ctx.fire_release_background(ctx.release_both_hands, "GoHome")
                release_msg = "both: dispatched (non-blocking)"
                ctx.logger.info(f"[GoHome] hand release (non-blocking): {release_msg}")
            else:
                hand_ok, hand_msg = ctx.release_both_hands()
                if not hand_ok:
                    return False, f"release before GoHome motion failed: {hand_msg}"
                ctx.logger.info(f"[GoHome] hand release (post-plan): {hand_msg}")
                release_msg = hand_msg

        # ---- Step 3: execute the planned dual-arm path -----------------
        dual_ok, dual_msg = ctx.execute_dual_waypoint_paths(
            left_waypoints_rad=left_path,
            right_waypoints_rad=right_path,
            wait=wait,
            timeout=timeout,
        )
        if not dual_ok:
            return False, f"Dual-arm GoHome failed: {dual_msg}"
    else:
        # Direct single-target mode: no plan to defer. Keep the original
        # "release first, then move" ordering.
        if do_release:
            if ctx.release_non_blocking_on_home:
                ctx.fire_release_background(ctx.release_both_hands, "GoHome")
                release_msg = "both: dispatched (non-blocking)"
                ctx.logger.info(f"[GoHome] hand release (non-blocking): {release_msg}")
            else:
                hand_ok, hand_msg = ctx.release_both_hands()
                if not hand_ok:
                    return False, f"release before GoHome failed: {hand_msg}"
                ctx.logger.info(f"[GoHome] hand release: {hand_msg}")
                release_msg = hand_msg

        left_success, left_q, left_msg = ctx.execute_joint_target(
            arm="left",
            q_target=left_home,
            wait=wait,
            timeout=timeout,
        )
        if not left_success:
            return False, f"Left arm GoHome failed: {left_msg}"

        right_success, right_q, right_msg = ctx.execute_joint_target(
            arm="right",
            q_target=right_home,
            wait=wait,
            timeout=timeout,
        )
        if not right_success:
            return False, f"Right arm GoHome failed: {right_msg}"

    summary = (
        "Path planned, hands released, arms moved to configured home joints"
        if do_release
        else "Path planned, arms moved to configured home joints"
    )
    return True, (
        f"{summary} "
        f"(left_deg={np.round(ctx.left_home_joints_deg, 2).tolist()}, "
        f"right_deg={np.round(ctx.right_home_joints_deg, 2).tolist()})"
        + (f" | release: {release_msg}" if release_msg else "")
    )


def go_home_arm(
    ctx: HomingContext,
    arm: str,
    wait: bool = True,
    timeout: Optional[float] = None,
    do_release: bool = True,
    use_astar: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Single-arm GoHome: plan + move only ``arm`` to its configured home
    joints, and release only that side's hand. The other arm is untouched.

    ``use_astar`` overrides ``~use_astar_for_go_home`` for this call:
      None  -> respect ``~use_astar_for_go_home`` (default behaviour).
      True  -> force A* on (priority A* through reachable map / voxel).
      False -> force A* off (direct joint target, skips planning entirely).
    """
    if timeout is None or timeout <= 0:
        timeout = ctx.default_timeout

    t_phase_start = time.time()
    t_start_total = t_phase_start

    home_rad = ctx.get_home_joints_rad(arm)
    release_msg = ""

    effective_use_astar = (
        ctx.use_astar_for_go_home if use_astar is None else bool(use_astar)
    )

    if effective_use_astar:
        start_rad = ctx.get_start_joint_for_astar(arm)
        try:
            path_rad = ctx.plan_astar_path_between_joints(
                arm=arm,
                q_start_rad=start_rad,
                q_goal_rad=home_rad,
                append_exact_goal=True,
            )
        except Exception as e:
            return False, f"[GoHomeArm:{arm}] A* planning failed: {e}"
        t_plan = time.time()
        ctx.logger.info(
            f"[GoHomeArm:{arm}] A* waypoints={len(path_rad)} "
            f"plan_ms={(t_plan - t_phase_start) * 1000.0:.0f} "
            f"start_deg={np.round(rad_list_to_deg(start_rad), 2).tolist()} "
            f"goal_deg={np.round(rad_list_to_deg(home_rad), 2).tolist()}"
        )
        t_phase_start = t_plan

        if do_release:
            if ctx.release_non_blocking_on_home:
                ctx.fire_release_background(
                    lambda a=arm: ctx.release_single_hand(a),
                    f"GoHomeArm:{arm}",
                )
                release_msg = f"{arm}: dispatched (non-blocking)"
                ctx.logger.info(
                    f"[GoHomeArm:{arm}] hand release (non-blocking): {release_msg}"
                )
            else:
                hand_ok, hand_msg = ctx.release_single_hand(arm)
                if not hand_ok:
                    return False, f"release before GoHomeArm motion failed: {hand_msg}"
                ctx.logger.info(f"[GoHomeArm:{arm}] hand release (post-plan): {hand_msg}")
                release_msg = hand_msg

        t_release_done = time.time()
        ctx.logger.info(
            f"[GoHomeArm:{arm}] release_dispatch_ms="
            f"{(t_release_done - t_phase_start) * 1000.0:.0f}"
        )
        t_phase_start = t_release_done
        ok, _q, msg = ctx.execute_waypoint_path(
            arm=arm,
            waypoints_rad=path_rad,
            wait=wait,
            timeout=timeout,
        )
        t_exec_done = time.time()
        ctx.logger.info(
            f"[GoHomeArm:{arm}] execute_ms="
            f"{(t_exec_done - t_phase_start) * 1000.0:.0f} "
            f"total_ms={(t_exec_done - t_start_total) * 1000.0:.0f}"
        )
        if not ok:
            return False, f"Single-arm GoHome failed: {msg}"
    else:
        if do_release:
            if ctx.release_non_blocking_on_home:
                ctx.fire_release_background(
                    lambda a=arm: ctx.release_single_hand(a),
                    f"GoHomeArm:{arm}",
                )
                release_msg = f"{arm}: dispatched (non-blocking)"
                ctx.logger.info(
                    f"[GoHomeArm:{arm}] hand release (non-blocking): {release_msg}"
                )
            else:
                hand_ok, hand_msg = ctx.release_single_hand(arm)
                if not hand_ok:
                    return False, f"release before GoHomeArm failed: {hand_msg}"
                ctx.logger.info(f"[GoHomeArm:{arm}] hand release: {hand_msg}")
                release_msg = hand_msg

        ok, _q, msg = ctx.execute_joint_target(
            arm=arm,
            q_target=home_rad,
            wait=wait,
            timeout=timeout,
        )
        if not ok:
            return False, f"Single-arm GoHome failed: {msg}"

    home_deg = (
        ctx.left_home_joints_deg if arm == "left" else ctx.right_home_joints_deg
    )
    summary = (
        f"{arm} arm moved to home joints, {arm} hand released"
        if do_release
        else f"{arm} arm moved to home joints"
    )
    return True, (
        f"{summary} ({arm}_deg={np.round(home_deg, 2).tolist()})"
        + (f" | release: {release_msg}" if release_msg else "")
    )
