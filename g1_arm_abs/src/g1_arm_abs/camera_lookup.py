# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/camera_lookup.py

Camera object-position lookup wrapper, extracted from ``arm_node_abs.py``.

ROS-light: wraps the ``g1_camera/GetObjectPosition`` service call (used by the
legacy ``grasp_bottle`` path). The ServiceProxy client + service name + timeout
are injected; this owns no state.
"""

from typing import Optional, Tuple

import numpy as np
import rospy

from g1_camera.srv import GetObjectPositionRequest

__all__ = ["get_object_position"]


def get_object_position(
    client,
    service_name: str,
    timeout: float,
    label: str,
    index: int,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Query ``/perception/get_object_position`` for one object's xyz.

    Returns ``(ok, target_xyz_or_None, message)``. A missing service, a failed
    call, an unsuccessful response, or a non-finite result all return
    ``ok=False`` with a descriptive message.
    """
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
    except rospy.ROSException as e:
        return False, None, f"service {service_name} not available: {e}"

    try:
        req = GetObjectPositionRequest(label=label, index=int(index))
        resp = client(req)
    except rospy.ServiceException as e:
        return False, None, f"call get_object_position failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, None, f"call get_object_position exception: {e}"

    if not resp.success:
        return False, None, f"get_object_position failed: {resp.message}"

    target = np.array([resp.x, resp.y, resp.z], dtype=float)
    if not np.all(np.isfinite(target)):
        return False, None, "get_object_position returned NaN or Inf"

    return True, target, resp.message
