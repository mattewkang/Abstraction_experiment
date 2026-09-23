# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/async_dispatch.py

Fire-and-forget daemon-thread helpers for hand services, extracted from
``arm_node_abs.py``.

ROS-light: these spawn a daemon thread and log the outcome, but own no state.
The release / trigger callables, clients, and logger are injected. The caller
never blocks on the result (outcomes are logged asynchronously).
"""

import threading
import time
from typing import Callable

__all__ = [
    "fire_release_background",
    "fire_delayed_point_gesture",
]


def fire_release_background(release_fn: Callable, label: str, logger) -> None:
    """Run ``release_fn()`` in a daemon thread and return immediately, so the
    per-hand finger-convergence wait inside the hand service doesn't stall the
    caller (e.g. GoHome). The outcome is logged async; never blocked on.
    """
    def _worker():
        try:
            ok, msg = release_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{label}] background release raised: {exc}")
            return
        if ok:
            logger.info(f"[{label}] background release ok: {msg}")
        else:
            logger.warning(f"[{label}] background release reported failure: {msg}")

    threading.Thread(
        target=_worker, name=f"{label}-release", daemon=True,
    ).start()


def fire_delayed_point_gesture(
    arm: str,
    delay_sec: float,
    service_name: str,
    client,
    call_trigger_service: Callable,
    logger,
) -> None:
    """Trigger the per-arm point_gesture service in a daemon thread after a
    short delay. Used by the press pipeline so the fingers shape into the
    pointing pose during the A* approach, offset from the arm motion start.
    Failures are logged async; the caller never blocks on the outcome.

    ``call_trigger_service(service_name, client, wait_available=True)`` is the
    injected Trigger-call helper.
    """
    delay = max(0.0, float(delay_sec))

    def _worker():
        if delay > 0:
            time.sleep(delay)
        try:
            ok, msg = call_trigger_service(service_name, client, wait_available=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[press_elevator_button] background point_gesture "
                f"({service_name}) raised: {exc}"
            )
            return
        level = logger.info if ok else logger.warning
        level(
            f"[press_elevator_button] background point_gesture "
            f"({service_name}, delay={delay:.2f}s): ok={ok} msg={msg}"
        )

    threading.Thread(
        target=_worker, name=f"point_gesture-{arm}", daemon=True,
    ).start()
