# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/berry_lookup.py

Berry-position lookup wrapper for the picking pipeline.

ROS-light: wraps the ``g1_camera/GetBerries`` service, which returns EVERY
detected berry in one call (parallel arrays of torso_link xyz + ripeness +
scores). The ServiceProxy client + service name + timeout are injected; this
owns no state. Mirrors ``camera_lookup.get_object_position`` but adapted to the
multi-detection response, plus a pick-ordering helper so the arm can choose
which berry to grasp first.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy

from g1_camera.srv import GetBerriesRequest

__all__ = ["Berry", "get_berries", "sort_for_picking", "nearest_berry"]


@dataclass
class Berry:
    """One detected berry with a valid 3D fix, in the robot base frame."""

    position: np.ndarray   # (3,) torso_link xyz, metres
    ripeness: str          # "red" / "black" / "white"
    det_score: float       # detector confidence
    ripeness_score: float  # ripeness-classifier probability

    @property
    def distance(self) -> float:
        """Euclidean distance from the torso_link origin (pick-order key)."""
        return float(np.linalg.norm(self.position))


def get_berries(
    client,
    service_name: str,
    timeout: float,
    ripeness: str = "",
) -> Tuple[bool, List[Berry], str]:
    """Query ``/perception/get_berries`` for every berry in view.

    ``ripeness`` optionally filters server-side ("" = all, or "red" / "black" /
    "white"). Returns ``(ok, berries, message)``. A missing service, a failed
    call, or an unsuccessful response returns ``ok=False`` with an empty list
    and a descriptive message. ``ok=True`` with an empty list is a valid
    "nothing detected" result. Berries with non-finite coordinates are dropped.
    """
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
    except rospy.ROSException as e:
        return False, [], f"service {service_name} not available: {e}"

    try:
        req = GetBerriesRequest(ripeness=ripeness)
        resp = client(req)
    except rospy.ServiceException as e:
        return False, [], f"call get_berries failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, [], f"call get_berries exception: {e}"

    if not resp.success:
        return False, [], f"get_berries failed: {resp.message}"

    berries: List[Berry] = []
    count = int(resp.count)
    for i in range(count):
        pos = np.array([resp.x[i], resp.y[i], resp.z[i]], dtype=float)
        if not np.all(np.isfinite(pos)):
            continue
        berries.append(Berry(
            position=pos,
            ripeness=(resp.ripeness_labels[i]
                      if i < len(resp.ripeness_labels) else ""),
            det_score=(float(resp.det_scores[i])
                       if i < len(resp.det_scores) else 0.0),
            ripeness_score=(float(resp.ripeness_scores[i])
                            if i < len(resp.ripeness_scores) else 0.0),
        ))
    return True, berries, resp.message


def sort_for_picking(
    berries: List[Berry],
    ripeness: Optional[str] = None,
) -> List[Berry]:
    """Order berries nearest-first (smallest torso_link distance) — the natural
    pick order for the arm. Pass ``ripeness`` to keep only that ripeness
    (client-side filter, complementary to the server-side request filter)."""
    out = berries
    if ripeness:
        out = [b for b in out if b.ripeness == ripeness]
    return sorted(out, key=lambda b: b.distance)


def nearest_berry(
    berries: List[Berry],
    ripeness: Optional[str] = None,
) -> Optional[Berry]:
    """The closest berry to grasp first, or ``None`` when none match."""
    ordered = sort_for_picking(berries, ripeness)
    return ordered[0] if ordered else None
