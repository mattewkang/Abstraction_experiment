# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/motion_exec.py

Joint-target / waypoint-path execution helpers extracted from
``arm_node_abs.py``.

ROS-light: these use ``rospy.Rate(50)`` and ``rospy.is_shutdown()`` for the
convergence polling loops, but own no state. The ``ArmController`` and a
``q_getter(arm) -> q|None`` (current-joint reader) are injected, along with the
position tolerance and default timeout, so the caller stays the sole owner of
the controller and its config.

All motion is routed through ``controller.ctrl_left_arm`` /
``ctrl_right_arm`` / ``ctrl_dual_arm`` so the per-cycle velocity clip inside the
controller is preserved.
"""

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import rospy

from g1_arm_abs.joint_index import ARM_DOF

__all__ = [
    "wait_until_reached",
    "wait_until_dual_reached",
    "execute_joint_target",
    "execute_waypoint_path",
    "execute_dual_waypoint_paths",
]


def wait_until_reached(
    arm: str,
    q_target,
    timeout: float,
    q_getter: Callable,
    position_tolerance: float,
) -> Tuple[bool, np.ndarray, str]:
    """Poll a single arm at 50 Hz until it converges within
    ``position_tolerance`` of ``q_target`` or ``timeout`` elapses.
    """
    start_time = time.time()
    rate = rospy.Rate(50)
    last_err = float("inf")

    while not rospy.is_shutdown():
        q_current = q_getter(arm)
        if q_current is not None:
            err = np.max(np.abs(q_current - q_target))
            last_err = float(err)
            if err < position_tolerance:
                return True, q_current, f"{arm} arm reached target"

        if time.time() - start_time > timeout:
            q_current = q_getter(arm)
            if q_current is None:
                q_current = np.zeros(ARM_DOF)
            return (
                False,
                q_current,
                (
                    f"{arm} arm motion timeout "
                    f"(max_err={last_err:.4f}, tol={position_tolerance:.4f}, "
                    f"target={np.round(q_target, 4).tolist()}, "
                    f"current={np.round(q_current, 4).tolist()})"
                ),
            )

        rate.sleep()

    q_current = q_getter(arm)
    if q_current is None:
        q_current = np.zeros(ARM_DOF)
    return False, q_current, "ROS shutdown while waiting"


def wait_until_dual_reached(
    q_left_target,
    q_right_target,
    timeout: float,
    q_getter: Callable,
    position_tolerance: float,
) -> Tuple[bool, np.ndarray, np.ndarray, str]:
    """Poll both arms at 50 Hz until both converge within ``position_tolerance``
    of their targets or ``timeout`` elapses.
    """
    start_time = time.time()
    rate = rospy.Rate(50)
    last_left_err = float("inf")
    last_right_err = float("inf")

    q_left_target = np.array(q_left_target, dtype=float).reshape(-1)
    q_right_target = np.array(q_right_target, dtype=float).reshape(-1)

    while not rospy.is_shutdown():
        q_left = q_getter("left")
        q_right = q_getter("right")

        if q_left is not None and q_right is not None:
            left_err = np.max(np.abs(q_left - q_left_target))
            right_err = np.max(np.abs(q_right - q_right_target))
            last_left_err = float(left_err)
            last_right_err = float(right_err)
            if left_err < position_tolerance and right_err < position_tolerance:
                return True, q_left, q_right, "both arms reached target"

        if time.time() - start_time > timeout:
            q_left = q_getter("left")
            q_right = q_getter("right")
            if q_left is None:
                q_left = np.zeros(ARM_DOF, dtype=float)
            if q_right is None:
                q_right = np.zeros(ARM_DOF, dtype=float)
            return (
                False,
                q_left,
                q_right,
                (
                    f"dual-arm motion timeout "
                    f"(left_max_err={last_left_err:.4f}, right_max_err={last_right_err:.4f}, "
                    f"tol={position_tolerance:.4f})"
                ),
            )
        rate.sleep()

    q_left = q_getter("left")
    q_right = q_getter("right")
    if q_left is None:
        q_left = np.zeros(ARM_DOF, dtype=float)
    if q_right is None:
        q_right = np.zeros(ARM_DOF, dtype=float)
    return False, q_left, q_right, "ROS shutdown while waiting"


def execute_joint_target(
    arm: str,
    q_target,
    controller,
    q_getter: Callable,
    position_tolerance: float,
    default_timeout: float,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> Tuple[bool, np.ndarray, str]:
    """Send one 5-DOF joint target to the given arm and optionally wait for it.

    Routes through ``controller.ctrl_{left,right}_arm`` so the velocity clip
    applies.
    """
    if timeout is None or timeout <= 0:
        timeout = default_timeout

    if arm == "left":
        controller.ctrl_left_arm(q_target)
    else:
        controller.ctrl_right_arm(q_target)

    if wait:
        return wait_until_reached(arm, q_target, timeout, q_getter, position_tolerance)
    else:
        q_current = q_getter(arm)
        if q_current is None:
            q_current = np.zeros(ARM_DOF)
        return True, q_current, f"{arm} arm target sent"


def execute_waypoint_path(
    arm: str,
    waypoints_rad,
    controller,
    q_getter: Callable,
    position_tolerance: float,
    default_timeout: float,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> Tuple[bool, np.ndarray, str]:
    """Drive one arm through a list of joint waypoints, waiting at each."""
    if timeout is None or timeout <= 0:
        timeout = default_timeout

    last_q = q_getter(arm)
    if last_q is None:
        last_q = np.zeros(ARM_DOF, dtype=float)

    if not waypoints_rad:
        return True, np.array(last_q, dtype=float), "empty path"

    for idx, q_target in enumerate(waypoints_rad):
        q_target = np.array(q_target, dtype=float).reshape(-1)
        if len(q_target) != ARM_DOF:
            return False, np.array(last_q, dtype=float), f"invalid waypoint dim at {idx}"

        success, q_current, msg = execute_joint_target(
            arm,
            q_target,
            controller,
            q_getter,
            position_tolerance,
            default_timeout,
            wait=wait,
            timeout=timeout,
        )
        last_q = q_current
        if not success:
            return False, np.array(q_current, dtype=float), f"waypoint[{idx}] failed: {msg}"

    return True, np.array(last_q, dtype=float), "path execution success"


def execute_dual_waypoint_paths(
    left_waypoints_rad,
    right_waypoints_rad,
    controller,
    q_getter: Callable,
    position_tolerance: float,
    default_timeout: float,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> Tuple[bool, str]:
    """Drive both arms through parallel waypoint lists in lock-step.

    The shorter list holds its final waypoint while the longer one finishes.
    Each step issues a single ``ctrl_dual_arm`` command and (optionally) waits
    for both arms to converge.
    """
    if timeout is None or timeout <= 0:
        timeout = default_timeout

    left_waypoints = [np.array(q, dtype=float).reshape(-1) for q in (left_waypoints_rad or [])]
    right_waypoints = [np.array(q, dtype=float).reshape(-1) for q in (right_waypoints_rad or [])]

    if not left_waypoints:
        q_left = q_getter("left")
        if q_left is None:
            q_left = np.zeros(ARM_DOF, dtype=float)
        left_waypoints = [np.array(q_left, dtype=float)]
    if not right_waypoints:
        q_right = q_getter("right")
        if q_right is None:
            q_right = np.zeros(ARM_DOF, dtype=float)
        right_waypoints = [np.array(q_right, dtype=float)]

    for q in left_waypoints + right_waypoints:
        if len(q) != ARM_DOF:
            return False, "invalid waypoint dimension for dual-arm path"

    n_steps = max(len(left_waypoints), len(right_waypoints))
    left_last = left_waypoints[-1]
    right_last = right_waypoints[-1]

    for i in range(n_steps):
        q_left = left_waypoints[i] if i < len(left_waypoints) else left_last
        q_right = right_waypoints[i] if i < len(right_waypoints) else right_last
        q_dual = np.concatenate([q_left, q_right], axis=0)
        controller.ctrl_dual_arm(q_dual)

        if wait:
            ok, _, _, msg = wait_until_dual_reached(
                q_left, q_right, timeout, q_getter, position_tolerance
            )
            if not ok:
                return False, f"dual waypoint[{i}] failed: {msg}"

    return True, "dual path execution success"
