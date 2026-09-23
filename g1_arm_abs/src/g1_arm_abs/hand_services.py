# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/hand_services.py

Thin wrappers around the per-side ``std_srvs/Trigger`` hand services
(``pre_grasp`` / ``grasp_5f`` / ``release``), extracted from ``arm_node_abs.py``.

ROS-light: these touch ``rospy.wait_for_service`` and the ROS exception types,
but advertise nothing and own no state. ServiceProxy clients and service-name
strings are passed in by the caller (the node owns the per-arm client dicts).
"""

import time
from typing import Dict, Tuple

import rospy

__all__ = [
    "call_trigger_service",
    "call_trigger_service_with_retry",
    "release_both_hands",
    "release_single_hand",
]


def call_trigger_service(
    service_name: str,
    client,
    timeout: float,
    wait_available: bool = True,
) -> Tuple[bool, str]:
    """Call a ``std_srvs/Trigger`` service. Returns ``(success, message)``.

    When ``wait_available`` is true, first wait up to ``timeout`` seconds for
    the service to appear (a miss returns a not-available message rather than
    raising).
    """
    if wait_available:
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
        except rospy.ROSException as e:
            return False, f"service {service_name} not available: {e}"

    try:
        resp = client()
        return resp.success, resp.message
    except rospy.ServiceException as e:
        return False, f"service {service_name} call failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"service {service_name} exception: {e}"


def call_trigger_service_with_retry(
    service_name: str,
    client,
    retries: int,
    interval: float,
    timeout: float,
    wait_available: bool = True,
) -> Tuple[bool, str]:
    """Call a Trigger service up to ``retries`` times, sleeping ``interval``
    seconds between failed attempts. Returns ``(success, joined_messages)``.
    """
    retries = max(1, int(retries))
    messages = []
    for i in range(retries):
        ok, msg = call_trigger_service(
            service_name,
            client,
            timeout,
            wait_available=wait_available,
        )
        messages.append(f"attempt {i+1}/{retries}: {msg}")
        if ok:
            return True, " | ".join(messages)
        if i < retries - 1 and interval > 0:
            time.sleep(interval)
    return False, " | ".join(messages)


def release_both_hands(
    release_clients: Dict[str, object],
    service_names: Dict[str, str],
) -> Tuple[bool, str]:
    """Release both hands, best-effort.

    A side whose service is unavailable (``ServiceException``) is skipped — no
    ``wait_for_service`` probe, so a missing side costs ~0 instead of a full
    timeout. Returns ``(success, message)`` where ``success`` is true when
    nothing actively failed; false if no side responded at all.
    """
    messages = []
    failures = 0
    responded = 0
    for side in ("right", "left"):
        srv_name = service_names[side]
        client = release_clients[side]
        try:
            resp = client()
        except rospy.ServiceException as e:
            messages.append(f"{side}: service {srv_name} unavailable (skipped): {e}")
            continue
        responded += 1
        messages.append(f"{side}: {resp.message}")
        if not resp.success:
            failures += 1

    if responded == 0:
        return False, "no hand release service responded | " + " | ".join(messages)
    return failures == 0, " | ".join(messages)


def release_single_hand(
    side: str,
    release_clients: Dict[str, object],
    service_names: Dict[str, str],
) -> Tuple[bool, str]:
    """Release only ``side``. Mirrors :func:`release_both_hands` semantics
    (skip on ``ServiceException``, surface failure) for a single side.
    """
    if side not in ("left", "right"):
        return False, f"invalid side '{side}'"
    srv_name = service_names[side]
    client = release_clients[side]
    try:
        resp = client()
    except rospy.ServiceException as e:
        return False, f"{side}: service {srv_name} unavailable: {e}"
    return bool(resp.success), f"{side}: {resp.message}"
