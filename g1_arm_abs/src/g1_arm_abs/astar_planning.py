# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/astar_planning.py

Thin glue over the voxel A* adapter, extracted from ``arm_node_abs.py``.

ROS-free: the ``VoxelPlannerAdapter`` instance is injected as ``adapter`` and a
``logger`` is passed in; rad/deg conversion, path interpolation, and waypoint
merging are reused from the existing ROS-free helpers. The adapter itself owns
the depth subscription + per-arm planner state (it is ROS-light), but these
functions add no ROS coupling of their own.
"""

from typing import List, Optional

import numpy as np

from g1_arm_abs.utils import (
    rad_list_to_deg,
    deg_list_to_rad,
    merge_close_points_rad,
)
from g1_arm_abs.path_ops import interpolate_path_deg

__all__ = [
    "plan_astar_segments",
    "plan_astar_path_between_joints",
    "debug_log_voxel_vs_no_obstacles",
]


def plan_astar_segments(
    adapter,
    arm: str,
    q_start_rad,
    q_pregrasp_rad,
    q_target_rad,
    fine_step_deg: float,
    debug_print_both_paths: bool,
    logger,
    max_cell_diff: int = 1,
    task_label: str = "Pregrasp",
    ignore_obstacles: bool = False,
) -> dict:
    """Plan the pre_path (current -> pregrasp) using the voxel A* adapter.

    Returns a dict with ``pre_path_rad`` (interpolated waypoints from start to
    pregrasp) and ``target_path_rad`` (one-element list with the grasp target).
    The caller merges the two into the final ``full_path_rad``. ``max_cell_diff``
    / ``task_label`` are forwarded to the adapter's goal-reachability check (the
    grasp keeps the strict 1-cell default; berry relaxes it, like the press leg).
    ``ignore_obstacles`` plans WITHOUT any voxel obstacle input (the berry path):
    the A* runs on the bare reachability grid, ignores depth, and only WARNS (not
    raises) on an unreachable goal, so ``max_cell_diff`` / ``task_label`` are
    unused in that mode.
    """
    if adapter is None:
        raise RuntimeError("voxel A* adapter is not available")

    q_start_deg = rad_list_to_deg(q_start_rad)
    q_pregrasp_deg = rad_list_to_deg(q_pregrasp_rad)

    if ignore_obstacles:
        path_deg, info = adapter.plan_joint_path_deg_no_obstacles(
            arm=arm, q_start_deg=q_start_deg, q_goal_deg=q_pregrasp_deg,
        )
    else:
        path_deg, info = adapter.plan_joint_path_deg(
            arm=arm, q_start_deg=q_start_deg, q_goal_deg=q_pregrasp_deg,
            max_cell_diff=max_cell_diff, task_label=task_label,
        )
    logger.info(f"[A* backend=voxel] {info}")

    if debug_print_both_paths:
        debug_log_voxel_vs_no_obstacles(
            adapter=adapter,
            arm=arm,
            q_start_deg=q_start_deg,
            q_pregrasp_deg=q_pregrasp_deg,
            voxel_path_deg=path_deg,
            voxel_info=info,
            logger=logger,
        )

    pre_deg_path = interpolate_path_deg(path_deg, fine_step_deg=fine_step_deg)
    pre_path_rad = [deg_list_to_rad(q_deg) for q_deg in pre_deg_path]
    target_path_rad = [np.array(q_target_rad, dtype=float).reshape(-1).tolist()]

    return {
        "pre_path_rad": pre_path_rad,
        "target_path_rad": target_path_rad,
    }


def debug_log_voxel_vs_no_obstacles(
    adapter,
    arm: str,
    q_start_deg,
    q_pregrasp_deg,
    voxel_path_deg,
    voxel_info,
    logger,
) -> None:
    """Diagnostic: replan the same start->pregrasp without obstacles and log
    both joint paths side-by-side. Logging-only — never affects the returned
    path.
    """
    try:
        no_obs_path_deg, no_obs_info = adapter.plan_joint_path_deg_no_obstacles(
            arm=arm, q_start_deg=q_start_deg, q_goal_deg=q_pregrasp_deg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[A* debug] no-obstacle replanning failed: {exc}")
        return

    def _fmt(path):
        return [
            "[" + ", ".join(f"{a:7.2f}" for a in q) + "]"
            for q in path
        ]

    logger.info(
        "[A* debug] === pre_path comparison (degrees, "
        f"arm={arm}, start_deg={[round(a, 2) for a in q_start_deg]}, "
        f"pregrasp_deg={[round(a, 2) for a in q_pregrasp_deg]}) ==="
    )
    logger.info(f"[A* debug] WITH-voxel    : {voxel_info}")
    for i, line in enumerate(_fmt(voxel_path_deg)):
        logger.info(f"[A* debug]   voxel[{i:02d}] {line}")
    logger.info(f"[A* debug] WITHOUT-voxel : {no_obs_info}")
    for i, line in enumerate(_fmt(no_obs_path_deg)):
        logger.info(f"[A* debug]   nobst[{i:02d}] {line}")
    logger.info(
        f"[A* debug] === counts: with={len(voxel_path_deg)} "
        f"without={len(no_obs_path_deg)} ==="
    )


def plan_astar_path_between_joints(
    adapter,
    arm: str,
    q_start_rad,
    q_goal_rad,
    merge_thresh_rad: float,
    append_exact_goal: bool = False,
    max_cell_diff: int = 1,
    task_label: str = "Pregrasp",
    logger=None,
) -> List[List[float]]:
    """Plan a generic joint-to-joint A* path via the adapter.

    Optionally appends the exact goal (when the A* landing differs from it) and
    merges near-coincident waypoints with ``merge_thresh_rad``. ``max_cell_diff``
    / ``task_label`` are forwarded to the adapter's reachability check.
    """
    if adapter is None:
        raise RuntimeError("voxel A* adapter is not available")
    q_start_deg = rad_list_to_deg(q_start_rad)
    q_goal_deg = rad_list_to_deg(q_goal_rad)
    path_deg, info = adapter.plan_joint_path_deg(
        arm=arm, q_start_deg=q_start_deg, q_goal_deg=q_goal_deg,
        max_cell_diff=max_cell_diff, task_label=task_label,
    )
    if logger is not None:
        logger.info(f"[A* backend=voxel] (joint-to-joint) {info}")
    path_rad = [deg_list_to_rad(q_deg) for q_deg in path_deg]
    if append_exact_goal:
        goal_rad = list(q_goal_rad)
        if not path_rad or np.max(np.abs(
            np.array(path_rad[-1]) - np.array(goal_rad)
        )) > 1e-9:
            path_rad = path_rad + [goal_rad]
    path_rad = merge_close_points_rad(
        path_rad,
        joint_thresh_rad=merge_thresh_rad,
    )
    return path_rad
