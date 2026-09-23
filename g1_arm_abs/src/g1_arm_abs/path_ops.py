# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/path_ops.py

Pure joint-space path geometry helpers extracted from ``arm_node_abs.py``.

These functions operate purely on numeric joint paths (lists of joint vectors)
and carry no robot, ROS, or model state. They are fully ROS-free so they can be
exercised offline.

Conventions:
- A "path" is a list of joint vectors; each vector is a list/array of joint
  angles. Degrees vs radians is encoded by the function name (``_deg`` suffix)
  and is otherwise irrelevant to the math.
- ``flatten_path`` packs a list of N joint vectors into a single flat list
  (row-major), matching the ``N*5`` flat float schema the ROS services expose.
"""

from typing import List, Sequence

import numpy as np

__all__ = [
    "flatten_path",
    "interpolate_path_deg",
    "downsample_waypoints_keep_last",
]


def flatten_path(path_rad: Sequence[Sequence[float]]) -> List[float]:
    """Flatten a list of joint vectors into a single row-major flat list."""
    if not path_rad:
        return []
    arr = np.array(path_rad, dtype=float).reshape(-1)
    return arr.tolist()


def interpolate_path_deg(
    path_deg: Sequence[Sequence[float]],
    fine_step_deg: float = 1.0,
) -> List[List[float]]:
    """Linearly densify a joint path so no per-step joint delta exceeds
    ``fine_step_deg`` degrees.

    The first waypoint is always kept; each segment is subdivided into the
    fewest equal steps that bound the max-axis delta by ``fine_step_deg``.
    A single-waypoint path is returned unchanged.
    """
    if not path_deg:
        return []
    fine_step_deg = max(0.1, float(fine_step_deg))
    arr = np.asarray(path_deg, dtype=float)
    if arr.shape[0] == 1:
        return [arr[0].tolist()]

    smooth_path = [arr[0]]
    for i in range(1, arr.shape[0]):
        q_prev = arr[i - 1]
        diff = arr[i] - q_prev
        n_steps = max(1, int(np.ceil(np.max(np.abs(diff)) / fine_step_deg)))
        for k in range(1, n_steps + 1):
            smooth_path.append(q_prev + diff * (k / n_steps))
    return [q.tolist() for q in smooth_path]


def downsample_waypoints_keep_last(
    waypoints_rad: List,
    stride: int,
) -> List:
    """Stride-subsample a waypoint list while always keeping the first and
    last waypoints. Paths of length <= 2 (or stride <= 1) are returned as-is.
    """
    stride = max(1, int(stride))
    if stride <= 1 or len(waypoints_rad) <= 2:
        return waypoints_rad
    kept = [waypoints_rad[0]]
    i = 1
    while i < len(waypoints_rad) - 1:
        kept.append(waypoints_rad[i])
        i += stride
    kept.append(waypoints_rad[-1])
    return kept
