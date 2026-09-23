# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/arm_controller_proc.py

Subprocess-isolated ArmController.

``ArmControllerProxy`` exposes the same public surface the node uses
(``start`` / ``stop`` / ``acquire_control`` / ``release_control`` /
``has_control`` / ``ctrl_dual_arm`` / ``ctrl_left_arm`` /
``ctrl_right_arm`` / ``get_left_arm_q`` / ``get_right_arm_q`` /
``is_running``) but runs the real :class:`ArmController` (DDS pub/sub, the
250 Hz publish thread, the 500 Hz state subscriber) in a dedicated child
process. That removes the ROS node's other threads (torch inference, voxel
depth processing, rospy service handlers) from the control loop's GIL,
which is the measured source of the 6-98 ms period spikes on the GX10.

IPC design (all hot paths avoid pickling):
  parent -> child : one Pipe carrying rare, small command tuples
                    (per-waypoint ctrl targets, acquire/release, stop).
                    ~50-200 us each.
  child  -> parent: a shared double array holding the latest 35 motor q +
                    dq and a monotonically increasing sequence counter,
                    written by the child's state callback at 500 Hz and
                    read lock-briefly by the parent's 50 Hz wait loops.

The child is started with the ``forkserver`` method with ONLY this module
preloaded: the forkserver process is a clean interpreter (no rospy/DDS
state, no forked locks), and unlike ``spawn`` it never re-executes the
``__main__`` module -- re-running the catkin devel wrapper of
``arm_node_abs.py`` in a child would fail on ``g1_arm_abs.srv`` (generated
service bindings resolve only through the devel-space package path).
Environment is inherited, so ``G1_ARM_LOOP_TIMING_DUMP`` /
``G1_ARM_LOOP_TIMING`` / ``G1_ARM_CTRL_CPUS`` keep working inside the child
and the [loop-timing] lines appear on the node's stdout.

Selection: ``arm_node_abs.py`` picks this proxy when ``~controller_subprocess``
is true (and not in simulation mode); otherwise it uses the in-process
ArmController unchanged.
"""

import logging
import multiprocessing as mp
import os
import threading
import time
from typing import Optional

import numpy as np

from .joint_index import G1_23_NUM_MOTORS, TOTAL_ARM_DOF, ARM_DOF

logger = logging.getLogger(__name__)

_START_EXTRA_TIMEOUT_S = 30.0   # on top of retries * (state_timeout + interval)
_STOP_TIMEOUT_S = 20.0          # release ramp + go_home <= 5 s + margin
_SWITCH_EXTRA_TIMEOUT_S = 5.0   # on top of the ramp + settle of one switch


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def _child_main(conn, shm_state, seq, ctor_kwargs):
    """Entry point of the controller child process."""
    logging.basicConfig(
        level=logging.INFO,
        format="[arm-ctrl-proc %(process)d] %(name)s: %(message)s",
    )
    log = logging.getLogger("arm_ctrl_proc")

    # Optional CPU pinning for the controller process only (no privileges
    # needed). Set e.g. G1_ARM_CTRL_CPUS="5-9,15-19" to keep the control
    # loop on the GX10's big X925 cores. Unset = scheduler default.
    cpus_env = os.environ.get("G1_ARM_CTRL_CPUS")
    if cpus_env:
        try:
            cpus = set()
            for part in cpus_env.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    cpus.update(range(int(a), int(b) + 1))
                else:
                    cpus.add(int(part))
            os.sched_setaffinity(0, cpus)
            log.info("controller child pinned to CPUs %s", sorted(cpus))
        except (ValueError, OSError) as e:
            log.warning("G1_ARM_CTRL_CPUS=%r not applied: %s", cpus_env, e)

    # Import here: with spawn this is a fresh interpreter without rospy.
    from g1_arm_abs.arm_controller import ArmController

    n = G1_23_NUM_MOTORS
    shm_np = np.frombuffer(shm_state.get_obj(), dtype=np.float64)

    def state_cb(q, dq):
        # Called from the controller's 500 Hz subscribe thread; two slice
        # assignments at C speed instead of a 2*35-element Python loop.
        with shm_state.get_lock():
            shm_np[:n] = q
            shm_np[n:] = dq
            seq.value += 1

    controller = ArmController(state_callback=state_cb, **ctor_kwargs)

    running = True
    while running:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            # Parent died or Ctrl-C propagated: release control and exit.
            if controller.is_running:
                controller.stop(go_home=False, release_control=True)
            break

        op = msg[0]
        try:
            if op == "start":
                _, kwargs = msg
                ok = controller.start(**kwargs)
                conn.send(("start", bool(ok)))
            elif op == "ctrl_dual":
                controller.ctrl_dual_arm(np.asarray(msg[1]), msg[2] and np.asarray(msg[2]))
            elif op == "ctrl_left":
                controller.ctrl_left_arm(np.asarray(msg[1]), msg[2] and np.asarray(msg[2]))
            elif op == "ctrl_right":
                controller.ctrl_right_arm(np.asarray(msg[1]), msg[2] and np.asarray(msg[2]))
            elif op == "acquire":
                conn.send(("acquire", bool(controller.acquire_control())))
            elif op == "release":
                conn.send(("release", bool(controller.release_control())))
            elif op == "stop":
                _, go_home, release_control = msg
                controller.stop(go_home=go_home, release_control=release_control)
                conn.send(("stopped", True))
                running = False
            else:
                log.warning("unknown op: %r", op)
        except Exception as e:  # keep the command loop alive on bad input
            log.exception("op %r failed: %s", op, e)
            if op in ("start", "stop", "acquire", "release"):
                conn.send((op, False))
                if op == "stop":
                    running = False

    conn.close()
    log.info("controller child exiting")


# ---------------------------------------------------------------------------
# Parent-side proxy
# ---------------------------------------------------------------------------

class ArmControllerProxy:
    """Same public surface as ArmController; execution in a child process."""

    def __init__(
        self,
        motion_mode: bool = True,
        simulation_mode: bool = False,
        network_interface: Optional[str] = None,
        control_frequency: float = 250.0,
        velocity_limit: float = 20.0,
        control_switch_delay: float = 0.5,
        control_switch_settle: float = 0.5,
    ):
        if simulation_mode:
            raise ValueError(
                "ArmControllerProxy is for real-robot mode; use ArmController "
                "for simulation_mode=True"
            )
        self._ctor_kwargs = dict(
            motion_mode=motion_mode,
            simulation_mode=False,
            network_interface=network_interface,
            control_frequency=control_frequency,
            velocity_limit=velocity_limit,
            control_switch_delay=control_switch_delay,
            control_switch_settle=control_switch_settle,
        )
        self._switch_timeout = (
            float(control_switch_delay) + float(control_switch_settle)
            + _SWITCH_EXTRA_TIMEOUT_S
        )
        self._has_control = not motion_mode
        # forkserver, NOT spawn: spawn re-executes the node's __main__ in the
        # child, which crashes on catkin devel-space imports (g1_arm_abs.srv).
        # The forkserver preloads only this module and forks clean children.
        self._ctx = mp.get_context("forkserver")
        self._ctx.set_forkserver_preload(["g1_arm_abs.arm_controller_proc"])
        self._shm_state = self._ctx.Array("d", 2 * G1_23_NUM_MOTORS)  # q[35]+dq[35]
        self._seq = self._ctx.Value("L", 0)
        self._conn, self._child_conn = self._ctx.Pipe()
        self._conn_lock = threading.Lock()
        self._proc = None
        self._running = False
        logger.info("ArmControllerProxy initialized (subprocess controller)")

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        connect_retries: int = 1,
        connect_retry_interval: float = 2.0,
        state_timeout: float = 10.0,
    ) -> bool:
        if self._running:
            logger.warning("Controller already running")
            return True

        self._proc = self._ctx.Process(
            target=_child_main,
            args=(self._child_conn, self._shm_state, self._seq, self._ctor_kwargs),
            name="arm-controller",
            daemon=True,
        )
        self._start_without_main_reexec()

        kwargs = dict(
            connect_retries=connect_retries,
            connect_retry_interval=connect_retry_interval,
            state_timeout=state_timeout,
        )
        timeout = (
            max(1, int(connect_retries)) * (state_timeout + connect_retry_interval)
            + _START_EXTRA_TIMEOUT_S
        )
        with self._conn_lock:
            self._conn.send(("start", kwargs))
            if not self._conn.poll(timeout):
                logger.error("Controller child did not answer start within %.0fs", timeout)
                self._terminate_child()
                return False
            tag, ok = self._conn.recv()

        if tag != "start" or not ok:
            logger.error("Controller child failed to start")
            self._terminate_child()
            return False

        self._running = True
        logger.info("ArmControllerProxy started (child pid=%s)", self._proc.pid)
        return True

    def _start_without_main_reexec(self) -> None:
        """Start the child without letting it re-execute the node's __main__.

        multiprocessing's get_preparation_data() (parent side, at start())
        records __main__'s __spec__ name or __file__ and the child re-runs
        that module during prepare(). For a catkin devel-space node script
        that re-run crashes on generated bindings (g1_arm_abs.srv). Hiding
        both attributes for the duration of start() makes the preparation
        data omit init_main_from_* so the child skips the main fixup; the
        target function is still imported normally from this module.
        """
        import sys
        main = sys.modules.get("__main__")
        saved_spec = getattr(main, "__spec__", None) if main else None
        had_file = main is not None and hasattr(main, "__file__")
        saved_file = getattr(main, "__file__", None) if had_file else None
        try:
            if main is not None:
                main.__spec__ = None
                if had_file:
                    del main.__file__
            self._proc.start()
        finally:
            if main is not None:
                main.__spec__ = saved_spec
                if had_file:
                    main.__file__ = saved_file

    def stop(self, go_home: bool = True, release_control: bool = True) -> None:
        if not self._running or self._proc is None:
            return
        logger.info("Stopping ArmControllerProxy...")
        try:
            with self._conn_lock:
                self._conn.send(("stop", go_home, release_control))
                if self._conn.poll(_STOP_TIMEOUT_S):
                    self._conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            logger.warning("Controller child did not exit; terminating")
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._running = False
        logger.info("ArmControllerProxy stopped")

    def _terminate_child(self):
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.is_alive()

    # -- SDK takeover / hand-back ------------------------------------------

    def _switch(self, op: str) -> bool:
        """Round-trip an acquire/release to the child; blocks for the ramp."""
        if not self.is_running:
            logger.warning("%s_control: controller not running", op)
            return False
        with self._conn_lock:
            self._conn.send((op,))
            if not self._conn.poll(self._switch_timeout):
                logger.error(
                    "Controller child did not answer %s within %.1fs",
                    op, self._switch_timeout,
                )
                return False
            tag, ok = self._conn.recv()
        ok = bool(ok) and tag == op
        if ok:
            self._has_control = (op == "acquire")
        return ok

    def acquire_control(self) -> bool:
        if self._has_control:
            return True
        return self._switch("acquire")

    def release_control(self) -> bool:
        if not self._has_control:
            return True
        return self._switch("release")

    @property
    def has_control(self) -> bool:
        return self._has_control

    # -- commands ----------------------------------------------------------

    def _send(self, msg) -> None:
        with self._conn_lock:
            self._conn.send(msg)

    def ctrl_dual_arm(self, q_target, tauff_target=None) -> None:
        q = np.asarray(q_target, dtype=float).reshape(-1)
        if len(q) != TOTAL_ARM_DOF:
            raise ValueError(f"Expected {TOTAL_ARM_DOF} joint angles, got {len(q)}")
        self._send(("ctrl_dual", q.tolist(),
                    None if tauff_target is None else list(tauff_target)))

    def ctrl_left_arm(self, q_target, tauff_target=None) -> None:
        q = np.asarray(q_target, dtype=float).reshape(-1)
        if len(q) != ARM_DOF:
            raise ValueError(f"Expected {ARM_DOF} joint angles, got {len(q)}")
        self._send(("ctrl_left", q.tolist(),
                    None if tauff_target is None else list(tauff_target)))

    def ctrl_right_arm(self, q_target, tauff_target=None) -> None:
        q = np.asarray(q_target, dtype=float).reshape(-1)
        if len(q) != ARM_DOF:
            raise ValueError(f"Expected {ARM_DOF} joint angles, got {len(q)}")
        self._send(("ctrl_right", q.tolist(),
                    None if tauff_target is None else list(tauff_target)))

    # -- state getters -----------------------------------------------------

    def _read_state(self):
        with self._shm_state.get_lock():
            if self._seq.value == 0:
                return None
            return np.frombuffer(self._shm_state.get_obj(), dtype=np.float64).copy()

    def get_current_motor_q(self):
        s = self._read_state()
        return None if s is None else s[:G1_23_NUM_MOTORS]

    def get_current_dual_arm_q(self):
        s = self._read_state()
        if s is None:
            return None
        from .joint_index import G1_23_JointArmIndex
        return np.array([s[j] for j in G1_23_JointArmIndex])

    def get_current_dual_arm_dq(self):
        s = self._read_state()
        if s is None:
            return None
        from .joint_index import G1_23_JointArmIndex
        return np.array([s[G1_23_NUM_MOTORS + j] for j in G1_23_JointArmIndex])

    def get_left_arm_q(self):
        q = self.get_current_dual_arm_q()
        return None if q is None else q[:ARM_DOF]

    def get_right_arm_q(self):
        q = self.get_current_dual_arm_q()
        return None if q is None else q[ARM_DOF:]
