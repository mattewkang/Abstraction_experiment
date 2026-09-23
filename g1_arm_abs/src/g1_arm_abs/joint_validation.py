# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/joint_validation.py

Pure joint-space and workspace validation helpers extracted from
``arm_node_abs.py``.

All functions are ROS-free. Per-arm joint limits come from
``joint_index.ARM_JOINT_LIMITS`` (5-DOF per arm); callers may pass a different
limits dict for testing. Cartesian workspace gating takes its bounds as an
explicit :class:`WorkspaceBounds` so the grasp pipeline and the press pipeline
can supply independent limits.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from g1_arm_abs.joint_index import ARM_JOINT_LIMITS, ARM_DOF

__all__ = [
    "WorkspaceBounds",
    "validate_joint_request",
    "check_target_workspace",
    "check_joint_limits",
    "clip_joints_to_limits",
]


@dataclass(frozen=True)
class WorkspaceBounds:
    """Cartesian workspace gate: target x must be <= max_x, z must be >= min_z."""

    max_x: float
    min_z: float


def validate_joint_request(
    arm,
    joint_angles,
) -> Tuple[bool, str, str, Optional[np.ndarray]]:
    """Validate an arm name + joint-angle request.

    Returns ``(ok, message, normalized_arm, q)``. An empty/blank arm defaults
    to ``"right"``. ``q`` is the float ndarray on success, else ``None``.
    """
    arm = arm.lower().strip() if isinstance(arm, str) else ""
    if not arm:
        arm = "right"
    if arm not in ("left", "right"):
        return False, "arm must be 'left' or 'right'", arm, None

    q = np.array(joint_angles, dtype=float).reshape(-1)
    if len(q) != ARM_DOF:
        return False, f"Expected {ARM_DOF} joint angles, got {len(q)}", arm, None

    if not np.all(np.isfinite(q)):
        return False, "joint_angles contains NaN or Inf", arm, None

    return True, "ok", arm, q


def check_target_workspace(
    target: Sequence[float],
    bounds: WorkspaceBounds,
) -> Tuple[bool, str]:
    """Gate a Cartesian target against ``bounds`` (max forward x, min height z)."""
    target = np.array(target, dtype=float).reshape(-1)
    if len(target) < 3:
        return False, "invalid target dimension"

    x = float(target[0])
    z = float(target[2])
    violations = []
    if x > bounds.max_x:
        violations.append(f"x={x:.4f} > max_x={bounds.max_x:.4f}")
    if z < bounds.min_z:
        violations.append(f"z={z:.4f} < min_z={bounds.min_z:.4f}")

    if violations:
        return False, "target out of workspace | " + " | ".join(violations)
    return True, "target within workspace"


def check_joint_limits(
    arm: str,
    q,
    limits: dict = ARM_JOINT_LIMITS,
) -> Tuple[bool, List[str]]:
    """Check a joint vector against per-arm limits. Returns ``(ok, violations)``."""
    arm_limits = limits.get(arm)
    if arm_limits is None:
        return False, [f"unknown arm '{arm}'"]

    q = np.array(q, dtype=float).reshape(-1)
    if len(q) != ARM_DOF:
        return False, [f"Expected {ARM_DOF} joints, got {len(q)}"]

    violations = []
    for i, angle in enumerate(q):
        lo, hi = arm_limits[i]
        if not (lo <= angle <= hi):
            violations.append(
                f"j{i}={angle:.4f} out of [{lo:.4f}, {hi:.4f}]"
            )

    return len(violations) == 0, violations


def clip_joints_to_limits(
    arm: str,
    q,
    limits: dict = ARM_JOINT_LIMITS,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Clamp a joint vector into per-arm limits.

    Returns ``(clipped_q, adjustments, errors)``. ``adjustments`` lists each
    axis that was moved; ``errors`` is non-empty only for an unknown arm or a
    wrong-dimension input (in which case the input is returned unmodified).
    """
    arm_limits = limits.get(arm)
    if arm_limits is None:
        return np.array(q, dtype=float).reshape(-1), [], [f"unknown arm '{arm}'"]

    q = np.array(q, dtype=float).reshape(-1)
    if len(q) != ARM_DOF:
        return q, [], [f"Expected {ARM_DOF} joints, got {len(q)}"]

    clipped = q.copy()
    adjustments = []
    for i, angle in enumerate(q):
        lo, hi = arm_limits[i]
        new_angle = float(np.clip(angle, lo, hi))
        if abs(new_angle - angle) > 1e-9:
            adjustments.append(
                f"j{i}: {angle:.4f} -> {new_angle:.4f} (limit [{lo:.4f}, {hi:.4f}])"
            )
        clipped[i] = new_angle

    return clipped, adjustments, []
