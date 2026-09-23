# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/pose_compensation.py

Per-arm Cartesian pose compensation applied to the NN-IK input, extracted from
``arm_node_abs.py``.

This is the GRASP-pipeline per-arm compensation and is intentionally DISTINCT
from ``solve_grasp_pose.compensate_pose`` (which applies a single tilt + single
z-offset). Here each arm carries independent x/y/z offsets that are subtracted
from the raw target before a fixed forward tilt rotation about the y axis.

ROS-free: the per-arm offsets + tilt are carried in :class:`PoseCompensationConfig`,
built once by the node from its ``~{r,l}_{x,y,z}_offset`` / ``~tilt_deg`` params.
"""

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "PoseCompensationConfig",
    "compensate_pose",
]


@dataclass(frozen=True)
class PoseCompensationConfig:
    """Per-arm grasp pose-compensation offsets and the shared forward tilt.

    ``use_compensation`` mirrors ``~use_pose_compensation``; it is the gate the
    caller honours before applying compensation (``compensate_pose`` itself
    always applies, matching the original ``_compensate_pose`` behaviour).
    """

    use_compensation: bool
    tilt_deg: float
    r_x_offset: float
    r_y_offset: float
    r_z_offset: float
    l_x_offset: float
    l_y_offset: float
    l_z_offset: float


def compensate_pose(
    x: float,
    y: float,
    z: float,
    hand: str,
    cfg: PoseCompensationConfig,
) -> Tuple[float, float, float]:
    """Apply per-arm offsets + forward tilt to a raw target before NN IK.

    Each per-arm offset is SUBTRACTED from the raw target (positive offset ->
    NN aims back toward the body / downward; negative -> further out / upward),
    then the point is rotated by ``tilt_deg`` about the y axis.
    """
    if hand == "left":
        x_offset = cfg.l_x_offset
        y_offset = cfg.l_y_offset
        z_offset = cfg.l_z_offset
    else:
        x_offset = cfg.r_x_offset
        y_offset = cfg.r_y_offset
        z_offset = cfg.r_z_offset
    x = x - x_offset
    y = y - y_offset
    z = z - z_offset
    theta = math.radians(cfg.tilt_deg)

    x_new = x * math.cos(theta) + z * math.sin(theta)
    y_new = y
    z_new = -x * math.sin(theta) + z * math.cos(theta)
    return x_new, y_new, z_new
