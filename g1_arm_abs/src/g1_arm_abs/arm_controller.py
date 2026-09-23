"""
Arm controller for Unitree G1 robot (23-DOF variant).

Provides DDS communication with the robot for arm control using SDK2.
Based on G1_23_ArmController pattern from xr_teleoperate.
"""

import logging
import os
import sys
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable

from .joint_index import (
    G1_23_JointIndex,
    G1_23_JointArmIndex,
    G1_23_NUM_MOTORS,
    TOTAL_ARM_DOF,
    ARM_DOF,
    WEIGHT_INDEX,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    check_joint_limit,
)
from .utils import DataBuffer, clip_arm_q_target

logger = logging.getLogger(__name__)

# DDS topics
TOPIC_LOW_STATE = "rt/lowstate"
TOPIC_ARM_SDK = "rt/arm_sdk"
TOPIC_LOW_CMD = "rt/lowcmd"

# =============================================================================
# BEGIN loop-timing instrumentation (diagnostic only; safe to delete).
#
# Measures the real cadence of the 250 Hz publish loop in _ctrl_motor_state:
#   T_compute = t(just before DDS Write) - t(loop iteration start)
#   T_send    = duration of the DDS Write call itself
#   T_period  = interval between successive command transmissions
# Samples are collected in memory and ONE summary line is logged every
# ~10 s (no per-cycle printing). Commanded values are never touched.
# Disable with env G1_ARM_LOOP_TIMING=0. To remove: delete this block and
# the five lines marked "loop-timing" inside _ctrl_motor_state.
# =============================================================================
_LOOP_TIMING_ENABLED = os.environ.get("G1_ARM_LOOP_TIMING", "1") != "0"


class _LoopTimingRecorder:
    """In-memory dt statistics for the control loop; one log line per window."""

    def __init__(self, target_dt_s: float, report_every_s: float = 10.0):
        self._target_ms = target_dt_s * 1e3
        self._report_every_ns = int(report_every_s * 1e9)
        self._period_ms = []
        self._compute_ms = []
        self._send_ms = []
        self._prev_send_ns = None
        self._window_start_ns = None
        # Optional raw-sample dump for offline A/B analysis. Set env
        # G1_ARM_LOOP_TIMING_DUMP=<path.csv> to append one "metric,ms" row
        # per sample at each window report (never per-cycle I/O).
        self._dump_path = os.environ.get("G1_ARM_LOOP_TIMING_DUMP") or None

    @staticmethod
    def _pct(sorted_v, p):
        k = int(round(p / 100.0 * (len(sorted_v) - 1)))
        return sorted_v[min(len(sorted_v) - 1, max(0, k))]

    def record(self, t_start_ns: int, t_pre_send_ns: int, t_post_send_ns: int) -> None:
        self._compute_ms.append((t_pre_send_ns - t_start_ns) / 1e6)
        self._send_ms.append((t_post_send_ns - t_pre_send_ns) / 1e6)
        if self._prev_send_ns is not None:
            self._period_ms.append((t_post_send_ns - self._prev_send_ns) / 1e6)
        self._prev_send_ns = t_post_send_ns

        if self._window_start_ns is None:
            self._window_start_ns = t_post_send_ns
        elif t_post_send_ns - self._window_start_ns >= self._report_every_ns:
            self._report()
            self._window_start_ns = t_post_send_ns

    def flush(self) -> None:
        """Report whatever is left in the current window (call at loop exit)."""
        self._report()

    def _dump(self) -> None:
        if self._dump_path is None:
            return
        try:
            with open(self._dump_path, "a") as f:
                f.writelines(f"period,{v:.6f}\n" for v in self._period_ms)
                f.writelines(f"compute,{v:.6f}\n" for v in self._compute_ms)
                f.writelines(f"send,{v:.6f}\n" for v in self._send_ms)
        except OSError as e:
            logger.warning("[loop-timing] dump failed: %s", e)
            self._dump_path = None

    def _report(self) -> None:
        p = sorted(self._period_ms)
        c = sorted(self._compute_ms)
        s = sorted(self._send_ms)
        if not p:
            return
        self._dump()
        tgt = self._target_ms
        n = len(p)
        mean = sum(p) / n
        var = sum((x - mean) ** 2 for x in p) / n
        logger.info(
            "[loop-timing] n=%d target=%.3fms | period mean=%.3f med=%.3f "
            "std=%.3f min=%.3f p95=%.3f p99=%.3f p99.9=%.3f max=%.3f | "
            "compute mean=%.3f p99=%.3f max=%.3f | "
            "send mean=%.3f p99=%.3f max=%.3f | "
            ">1.2x=%d >1.5x=%d >2x=%d | eff=%.1fHz",
            n, tgt,
            mean, self._pct(p, 50), var ** 0.5, p[0],
            self._pct(p, 95), self._pct(p, 99), self._pct(p, 99.9), p[-1],
            sum(c) / len(c), self._pct(c, 99), c[-1],
            sum(s) / len(s), self._pct(s, 99), s[-1],
            sum(1 for x in p if x > 1.2 * tgt),
            sum(1 for x in p if x > 1.5 * tgt),
            sum(1 for x in p if x > 2.0 * tgt),
            1000.0 / mean,
        )
        self._period_ms.clear()
        self._compute_ms.clear()
        self._send_ms.clear()
# =============================================================================
# END loop-timing instrumentation
# =============================================================================


def _ensure_local_unitree_sdk_path() -> bool:
    """Add bundled SDK locations to sys.path if present.

    Repo layout: this file lives at
    <repo>/g1_arm_abs/src/g1_arm_abs/arm_controller.py, and the bundled SDK
    sits at <repo>/external/unitree_sdk2_python. realpath resolves the
    catkin_ws/src symlink back to the source tree first, and a few extra
    parent levels are probed so other checkout depths still work. The
    supported path remains `pip install -e <repo>/external/unitree_sdk2_python`
    inside g1_intellect_env; this walk is only a fallback.
    """
    root = os.path.dirname(os.path.realpath(__file__))
    for _ in range(6):
        root = os.path.dirname(root)
        candidate_dirs = (
            os.path.join(root, "unitree_sdk2_python"),
            os.path.join(root, "external", "unitree_sdk2_python"),
        )
        for candidate in candidate_dirs:
            if os.path.isdir(os.path.join(candidate, "unitree_sdk2py")):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                    logger.info("Added bundled Unitree SDK path: %s", candidate)
                return True
    return False


# Precomputed index arrays for the numpy state buffers (see
# _subscribe_motor_state): one fancy-index pull instead of a Python loop.
_ALL_MOTOR_IDX = np.array([j.value for j in G1_23_JointIndex], dtype=np.intp)
_ARM_MOTOR_IDX = np.array([j.value for j in G1_23_JointArmIndex], dtype=np.intp)


class MotorState:
    """Motor state container."""

    def __init__(self):
        self.q = 0.0   # Joint position (rad)
        self.dq = 0.0  # Joint velocity (rad/s)


class LowState:
    """Low-level state container for all motors."""

    def __init__(self, num_motors: int = G1_23_NUM_MOTORS):
        self.motor_state = [MotorState() for _ in range(num_motors)]


class ArmController:
    """
    Arm controller for G1_23 robot using SDK2 DDS communication.

    Controls both arms (10 DOF total) at 250Hz with velocity limiting
    and weight parameter management for motion mode.

    In motion mode the controller starts RELEASED (arm_sdk weight 0): the
    robot's own arm controller keeps running (arm swing while walking).
    Call ``acquire_control()`` before issuing targets and
    ``release_control()`` when the task is done; both ramp the weight over
    ``control_switch_delay`` seconds so the handoff is smooth.
    """

    def __init__(
        self,
        motion_mode: bool = True,
        simulation_mode: bool = False,
        network_interface: Optional[str] = None,
        control_frequency: float = 250.0,
        velocity_limit: float = 20.0,
        state_callback: Optional[Callable] = None,
        kp_arm: float = 80.0,
        kd_arm: float = 3.0,
        kp_wrist: float = 40.0,
        kd_wrist: float = 1.5,
        kp_lock: float = 300.0,
        kd_lock: float = 3.0,
        control_switch_delay: float = 0.5,
        control_switch_settle: float = 0.5,
        weight_transition_interval: float = 0.02,
    ):
        """
        Initialize arm controller.

        Args:
            motion_mode: Use rt/arm_sdk topic (True) or rt/lowcmd (False)
            simulation_mode: Skip robot connection for testing
            network_interface: Network interface for DDS (e.g., 'eth0')
            control_frequency: Control loop frequency in Hz
            velocity_limit: Maximum joint velocity (rad/s)
            state_callback: Callback function called on state update
            kp_arm: Position gain for shoulder/elbow joints
            kd_arm: Damping gain for shoulder/elbow joints
            kp_wrist: Position gain for wrist joints
            kd_wrist: Damping gain for wrist joints
            kp_lock: Position gain for locked non-arm joints
            kd_lock: Damping gain for locked non-arm joints
            control_switch_delay: Duration (s) of the weight ramp used on
                every SDK takeover (0->1) and hand-back (1->0). The arm
                starts released; acquire_control() / release_control()
                switch between the robot's own arm controller and this SDK.
            control_switch_settle: Extra hold (s) after each ramp has
                finished, before acquire_control() / release_control()
                return. Nothing else is commanded during this window, so
                the next control command (SDK arm target or the robot's
                own gait) only starts once the handoff has settled.
            weight_transition_interval: Interval between weight ramp steps (s)
        """
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.control_frequency = control_frequency
        self.control_dt = 1.0 / control_frequency
        self.velocity_limit = velocity_limit
        self._state_callback = state_callback

        # Control gains
        self.kp_lock = kp_lock
        self.kd_lock = kd_lock
        self.kp_arm = kp_arm
        self.kd_arm = kd_arm
        self.kp_wrist = kp_wrist
        self.kd_wrist = kd_wrist

        # Weight ramp parameters (shared by takeover and hand-back)
        self._control_switch_delay = max(0.0, float(control_switch_delay))
        self._control_switch_settle = max(0.0, float(control_switch_settle))
        self._weight_transition_interval = max(1e-3, float(weight_transition_interval))
        self._weight_transition_steps = max(
            2, int(round(self._control_switch_delay / self._weight_transition_interval)) + 1
        )
        self._switch_lock = threading.Lock()

        # Target joint angles and torques
        self._q_target = np.zeros(TOTAL_ARM_DOF)
        self._tauff_target = np.zeros(TOTAL_ARM_DOF)
        self._ctrl_lock = threading.Lock()

        # State buffer
        self._lowstate_buffer = DataBuffer()
        self._all_motor_q: Optional[np.ndarray] = None

        # Weight for motion mode (0 = robot's own arm controller, 1 = SDK).
        # Always start released: the robot keeps its native arm behaviour
        # (arm swing while walking) until acquire_control() is called.
        self._weight = 0.0
        # True while this SDK owns the arms. In lowcmd mode (motion_mode
        # False) there is no weight channel, so the SDK always owns them.
        self._has_control = not motion_mode
        self._warned_ctrl_without_control = False

        # Thread control
        self._running = False
        self._subscribe_thread: Optional[threading.Thread] = None
        self._publish_thread: Optional[threading.Thread] = None

        # SDK components (initialized in start())
        self._crc = None
        self._msg = None
        self._lowcmd_publisher = None
        self._lowstate_subscriber = None

        # Network interface
        if network_interface is None:
            import os
            network_interface = os.environ.get('G1_NETWORK_INTERFACE')
            if network_interface is None:
                if not simulation_mode:
                    raise RuntimeError(
                        "G1_NETWORK_INTERFACE environment variable not set. "
                        "Set it to the network interface connected to the robot "
                        "(e.g., 'export G1_NETWORK_INTERFACE=eth0')"
                    )
                network_interface = 'lo'  # loopback for simulation
                logger.info("Using loopback interface for simulation mode")
        self._network_interface = network_interface

        logger.info(f"ArmController initialized (motion_mode={motion_mode}, "
                    f"simulation_mode={simulation_mode})")

    def start(
        self,
        connect_retries: int = 1,
        connect_retry_interval: float = 2.0,
        state_timeout: float = 10.0,
    ) -> bool:
        """
        Start the controller.

        Initializes DDS communication and starts control threads. Robot
        connection (DDS bring-up + first-state handshake) can be flaky right
        after power-on or while the network interface is settling, so the
        connection attempt is retried up to ``connect_retries`` times.

        Args:
            connect_retries: Max number of connection attempts (>=1).
            connect_retry_interval: Seconds to wait between attempts.
            state_timeout: Seconds to wait for the first robot state per
                attempt before giving up on that attempt.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Controller already running")
            return True

        if self.simulation_mode:
            logger.info("Starting in simulation mode (no robot connection)")
            self._running = True
            self._init_simulation_state()
            return True

        retries = max(1, int(connect_retries))
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                # DDS bring-up runs once. ChannelFactoryInitialize creates a
                # process-global participant, so re-running _init_sdk on a
                # retry would leak it. If a prior attempt failed before the
                # publisher was created, _lowcmd_publisher is still None and we
                # re-run it; once it succeeds we keep the channels and only
                # retry the state handshake.
                if self._lowcmd_publisher is None:
                    self._init_sdk()

                self._running = True

                # (Re)start subscription thread. A failed attempt sets
                # _running=False, which lets the previous thread exit; spawn a
                # fresh one here.
                if (self._subscribe_thread is None
                        or not self._subscribe_thread.is_alive()):
                    self._subscribe_thread = threading.Thread(
                        target=self._subscribe_motor_state,
                        daemon=True
                    )
                    self._subscribe_thread.start()

                # Wait for first state
                start_time = time.time()
                while not self._lowstate_buffer.has_data():
                    if time.time() - start_time > state_timeout:
                        raise RuntimeError(
                            f"Timeout waiting for robot state "
                            f"({state_timeout:.1f}s)"
                        )
                    time.sleep(0.1)
                    logger.debug("Waiting to subscribe to DDS...")

                logger.info("DDS subscription established")

                # Initialize message with current motor positions
                self._init_motor_commands()

                # Start control thread
                self._publish_thread = threading.Thread(
                    target=self._ctrl_motor_state,
                    daemon=True
                )
                self._publish_thread.start()

                logger.info("ArmController started successfully")
                return True

            except Exception as e:
                last_error = e
                # Drop _running so the subscribe thread (if any) winds down
                # before the next attempt re-spawns it.
                self._running = False
                logger.error(
                    f"Failed to start controller "
                    f"(attempt {attempt}/{retries}): {e}"
                )
                if attempt < retries:
                    logger.info(
                        f"Retrying robot connection in "
                        f"{connect_retry_interval:.1f}s..."
                    )
                    time.sleep(connect_retry_interval)

        logger.error(
            f"Failed to start controller after {retries} attempt(s): "
            f"{last_error}"
        )
        return False

    def stop(self, go_home: bool = True, release_control: bool = True) -> None:
        """
        Stop the controller.

        Args:
            go_home: Move arms to home position before stopping
            release_control: Gradually release control (weight 1->0)
        """
        if not self._running:
            return

        logger.info("Stopping ArmController...")

        if go_home and not self.simulation_mode:
            self.go_home(release_after=release_control)
        elif release_control and not self.simulation_mode:
            self.release_control()

        self._running = False

        # Wait for threads to finish with timeout
        thread_timeout = 2.0
        if self._subscribe_thread and self._subscribe_thread.is_alive():
            self._subscribe_thread.join(timeout=thread_timeout)
            if self._subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not terminate within timeout")

        if self._publish_thread and self._publish_thread.is_alive():
            self._publish_thread.join(timeout=thread_timeout)
            if self._publish_thread.is_alive():
                logger.warning("Publish thread did not terminate within timeout")

        logger.info("ArmController stopped")

    def _init_sdk(self) -> None:
        """Initialize SDK2 DDS communication."""
        try:
            from unitree_sdk2py.core.channel import (
                ChannelPublisher,
                ChannelSubscriber,
                ChannelFactoryInitialize,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as hg_LowCmd,
                LowState_ as hg_LowState,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
        except ModuleNotFoundError as exc:
            if exc.name == "unitree_sdk2py" and _ensure_local_unitree_sdk_path():
                from unitree_sdk2py.core.channel import (
                    ChannelPublisher,
                    ChannelSubscriber,
                    ChannelFactoryInitialize,
                )
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                    LowCmd_ as hg_LowCmd,
                    LowState_ as hg_LowState,
                )
                from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
                from unitree_sdk2py.utils.crc import CRC
            elif exc.name and exc.name.startswith("cyclonedds"):
                raise RuntimeError(
                    "Missing Python dependency 'cyclonedds'. "
                    "Install Unitree SDK dependencies in the ROS Python environment, "
                    "for example: `pip install cyclonedds==0.10.2 opencv-python` "
                    "and `pip install -e <repo>/unitree_sdk2_python`."
                ) from exc
            else:
                raise

        # Initialize DDS
        ChannelFactoryInitialize(0, self._network_interface)

        # Create publisher
        topic = TOPIC_ARM_SDK if self.motion_mode else TOPIC_LOW_CMD
        self._lowcmd_publisher = ChannelPublisher(topic, hg_LowCmd)
        self._lowcmd_publisher.Init()

        # Create subscriber
        self._lowstate_subscriber = ChannelSubscriber(TOPIC_LOW_STATE, hg_LowState)
        self._lowstate_subscriber.Init()

        # Initialize CRC and message
        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._msg.mode_pr = 0

        logger.info(f"SDK initialized on interface {self._network_interface}, "
                    f"topic={topic}")

    def _init_simulation_state(self) -> None:
        """Initialize simulation state."""
        self._lowstate_buffer.set(
            (np.zeros(G1_23_NUM_MOTORS), np.zeros(G1_23_NUM_MOTORS))
        )
        self._all_motor_q = np.zeros(G1_23_NUM_MOTORS)

    def _init_motor_commands(self) -> None:
        """Initialize motor command message with current positions and gains."""
        # Get mode_machine from robot
        state_msg = self._lowstate_subscriber.Read()
        if state_msg:
            self._msg.mode_machine = state_msg.mode_machine

        # Get current motor positions
        self._all_motor_q = self.get_current_motor_q()

        # Configure all joints
        arm_indices = set(member.value for member in G1_23_JointArmIndex)

        for joint_id in G1_23_JointIndex:
            self._msg.motor_cmd[joint_id].mode = 1

            if joint_id.value in arm_indices:
                # Arm joint
                if self._is_wrist_motor(joint_id):
                    self._msg.motor_cmd[joint_id].kp = self.kp_wrist
                    self._msg.motor_cmd[joint_id].kd = self.kd_wrist
                else:
                    self._msg.motor_cmd[joint_id].kp = self.kp_arm
                    self._msg.motor_cmd[joint_id].kd = self.kd_arm
            else:
                # Non-arm joint (lock at current position)
                if self._is_weak_motor(joint_id):
                    self._msg.motor_cmd[joint_id].kp = self.kp_arm
                    self._msg.motor_cmd[joint_id].kd = self.kd_arm
                else:
                    self._msg.motor_cmd[joint_id].kp = self.kp_lock
                    self._msg.motor_cmd[joint_id].kd = self.kd_lock

            self._msg.motor_cmd[joint_id].q = self._all_motor_q[joint_id]

        logger.info("Motor commands initialized")
        logger.debug(f"Current arm positions: {self.get_current_dual_arm_q()}")

    def _subscribe_motor_state(self) -> None:
        """Subscription thread: read motor state at 500Hz.

        Stores the state as a pair of numpy arrays ``(q, dq)`` instead of a
        LowState object tree: building 35 MotorState objects per message at
        500 Hz costs real CPU and GC churn, and the pure-Python copy loop
        holds the GIL long enough to stall the 250 Hz publish thread when
        the machine is under memory/CPU pressure.
        """
        n = G1_23_NUM_MOTORS
        while self._running:
            msg = self._lowstate_subscriber.Read()
            if msg is not None:
                q = np.empty(n)
                dq = np.empty(n)
                motor_state = msg.motor_state
                for i in range(n):
                    m = motor_state[i]
                    q[i] = m.q
                    dq[i] = m.dq
                self._lowstate_buffer.set((q, dq))

                # Callback for state updates
                if self._state_callback:
                    try:
                        self._state_callback(q, dq)
                    except Exception as e:
                        logger.warning(f"State callback error: {e}")

            time.sleep(0.002)  # 500Hz

    def _ctrl_motor_state(self) -> None:
        """Control thread: publish commands at control_frequency."""
        # Weight for motion mode (refreshed every cycle below)
        if self.motion_mode:
            self._msg.motor_cmd[WEIGHT_INDEX].q = self._weight

        # loop-timing (diagnostic; see _LoopTimingRecorder above)
        _timing = _LoopTimingRecorder(self.control_dt) if _LOOP_TIMING_ENABLED else None

        # Absolute-deadline pacing: each cycle sleeps until a fixed schedule
        # tick instead of `dt - elapsed`, so wake-up latency does not
        # accumulate into period drift. A stall longer than one period skips
        # the missed ticks (no catch-up burst) by re-anchoring the deadline.
        next_deadline = time.perf_counter()

        while self._running:
            if _timing is not None:  # loop-timing
                _t0 = time.perf_counter_ns()

            # Get target with lock
            with self._ctrl_lock:
                arm_q_target = self._q_target.copy()
                arm_tauff_target = self._tauff_target.copy()

            # Apply velocity clipping
            current_q = self.get_current_dual_arm_q()
            if current_q is not None:
                clipped_q_target = clip_arm_q_target(
                    arm_q_target, current_q,
                    self.velocity_limit, self.control_dt
                )
            else:
                clipped_q_target = arm_q_target

            # Set motor commands for arm joints
            for idx, joint_id in enumerate(G1_23_JointArmIndex):
                self._msg.motor_cmd[joint_id].q = clipped_q_target[idx]
                self._msg.motor_cmd[joint_id].dq = 0
                self._msg.motor_cmd[joint_id].tau = arm_tauff_target[idx]

            # Weight (0 = robot's own controller, 1 = SDK); ramped by
            # acquire_control / release_control.
            if self.motion_mode:
                self._msg.motor_cmd[WEIGHT_INDEX].q = self._weight

            # Compute CRC and publish
            self._msg.crc = self._crc.Crc(self._msg)
            if _timing is not None:  # loop-timing
                _t1 = time.perf_counter_ns()
            self._lowcmd_publisher.Write(self._msg)
            if _timing is not None:  # loop-timing
                _timing.record(_t0, _t1, time.perf_counter_ns())

            # Sleep until the next schedule tick (skip missed ticks on stall)
            next_deadline += self.control_dt
            now = time.perf_counter()
            if now >= next_deadline:
                next_deadline = now
            else:
                time.sleep(next_deadline - now)

        if _timing is not None:  # loop-timing: report the final partial window
            _timing.flush()

    def _is_weak_motor(self, motor_index: G1_23_JointIndex) -> bool:
        """Check if motor is a weak motor requiring lower gains (ankle joints)."""
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
        ]
        return motor_index.value in weak_motors

    def _is_wrist_motor(self, motor_index: G1_23_JointIndex) -> bool:
        """Check if motor is a wrist motor."""
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

    def _ramp_weight(self, start: float, end: float) -> None:
        """Linearly ramp the arm_sdk weight over control_switch_delay."""
        for weight in np.linspace(start, end, num=self._weight_transition_steps):
            self._weight = float(weight)
            time.sleep(self._weight_transition_interval)
        self._weight = float(end)
        # Safety settle: hold the new ownership for control_switch_settle
        # before any further command may be issued either way.
        if self._control_switch_settle > 0.0:
            time.sleep(self._control_switch_settle)

    @property
    def control_switch_duration(self) -> float:
        """Total blocking time of one acquire/release (ramp + settle)."""
        return self._control_switch_delay + self._control_switch_settle

    def acquire_control(self) -> bool:
        """Take over the arms from the robot's own controller.

        Seeds the joint targets with the measured arm pose (so the arms hold
        still during the handoff) and ramps the arm_sdk weight 0->1 over
        ``control_switch_delay`` seconds, then holds for
        ``control_switch_settle`` seconds. Blocking; idempotent.

        Returns:
            True once this SDK owns the arms.
        """
        with self._switch_lock:
            if self._has_control:
                return True
            if not self._running:
                logger.warning("acquire_control: controller not running")
                return False
            if self.simulation_mode or not self.motion_mode:
                self._has_control = True
                return True

            current_q = self.get_current_dual_arm_q()
            if current_q is None:
                logger.warning("acquire_control: no arm state yet")
                return False
            with self._ctrl_lock:
                self._q_target = np.array(current_q, dtype=float)
                self._tauff_target = np.zeros(TOTAL_ARM_DOF)

            logger.info(
                "Taking over arm control (weight 0->1 over %.2fs, settle %.2fs)...",
                self._control_switch_delay, self._control_switch_settle,
            )
            self._ramp_weight(0.0, 1.0)
            self._has_control = True
            self._warned_ctrl_without_control = False
            logger.info("Arm control acquired")
            return True

    def release_control(self) -> bool:
        """Hand the arms back to the robot's own controller.

        Ramps the arm_sdk weight 1->0 over ``control_switch_delay`` seconds,
        then holds for ``control_switch_settle`` seconds. Blocking; idempotent. In lowcmd mode there is nothing to hand back.

        Returns:
            True once the robot's own controller owns the arms (or when the
            mode has no weight channel).
        """
        with self._switch_lock:
            if not self.motion_mode:
                return True
            if not self._has_control:
                return True
            if self.simulation_mode or not self._running:
                self._has_control = False
                self._weight = 0.0
                return True

            logger.info(
                "Releasing arm control (weight 1->0 over %.2fs, settle %.2fs)...",
                self._control_switch_delay, self._control_switch_settle,
            )
            self._ramp_weight(1.0, 0.0)
            self._has_control = False
            logger.info("Arm control released")
            return True

    def _release_control(self) -> None:
        """Shutdown path: hand back if this SDK still owns the arms."""
        self.release_control()

    def _check_has_control(self) -> None:
        if self._has_control or self._warned_ctrl_without_control:
            return
        self._warned_ctrl_without_control = True
        logger.warning(
            "Arm target set while the SDK does not own the arms "
            "(weight 0); call acquire_control() first"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def ctrl_dual_arm(
        self,
        q_target: np.ndarray,
        tauff_target: Optional[np.ndarray] = None
    ) -> None:
        """
        Set target joint angles for both arms.

        Args:
            q_target: Target angles for 10 joints (5 left + 5 right)
            tauff_target: Target feedforward torques (optional)
        """
        if len(q_target) != TOTAL_ARM_DOF:
            raise ValueError(f"Expected {TOTAL_ARM_DOF} joint angles, got {len(q_target)}")

        self._check_has_control()
        with self._ctrl_lock:
            self._q_target = np.array(q_target)
            if tauff_target is not None:
                self._tauff_target = np.array(tauff_target)

    def ctrl_left_arm(
        self,
        q_target: np.ndarray,
        tauff_target: Optional[np.ndarray] = None
    ) -> None:
        """
        Set target joint angles for left arm only.

        Args:
            q_target: Target angles for 5 left arm joints
            tauff_target: Target feedforward torques (optional)
        """
        if len(q_target) != ARM_DOF:
            raise ValueError(f"Expected {ARM_DOF} joint angles, got {len(q_target)}")

        self._check_has_control()
        with self._ctrl_lock:
            self._q_target[:ARM_DOF] = q_target
            if tauff_target is not None:
                self._tauff_target[:ARM_DOF] = tauff_target

    def ctrl_right_arm(
        self,
        q_target: np.ndarray,
        tauff_target: Optional[np.ndarray] = None
    ) -> None:
        """
        Set target joint angles for right arm only.

        Args:
            q_target: Target angles for 5 right arm joints
            tauff_target: Target feedforward torques (optional)
        """
        if len(q_target) != ARM_DOF:
            raise ValueError(f"Expected {ARM_DOF} joint angles, got {len(q_target)}")

        self._check_has_control()
        with self._ctrl_lock:
            self._q_target[ARM_DOF:] = q_target
            if tauff_target is not None:
                self._tauff_target[ARM_DOF:] = tauff_target

    def set_joint_angle(
        self,
        arm: str,
        joint_idx: int,
        angle: float,
        check_limits: bool = True
    ) -> None:
        """
        Set single joint angle.

        Args:
            arm: 'left' or 'right'
            joint_idx: Joint index within arm (0-4)
            angle: Target angle in radians
            check_limits: If True, validate angle against joint limits
        """
        if joint_idx < 0 or joint_idx >= ARM_DOF:
            raise ValueError(f"Joint index must be 0-{ARM_DOF-1}")

        arm_lower = arm.lower()
        if arm_lower not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")

        # Validate joint limits
        if check_limits:
            is_valid, lower, upper = check_joint_limit(arm_lower, joint_idx, angle)
            if not is_valid:
                raise ValueError(
                    f"Angle {angle:.4f} rad exceeds joint limits [{lower:.4f}, {upper:.4f}] rad"
                )

        self._check_has_control()
        with self._ctrl_lock:
            if arm_lower == 'left':
                self._q_target[joint_idx] = angle
            else:
                self._q_target[ARM_DOF + joint_idx] = angle

    def get_current_motor_q(self) -> Optional[np.ndarray]:
        """Get current positions of all motors."""
        state = self._lowstate_buffer.get()
        if state is None:
            return None
        q, _dq = state
        return q[_ALL_MOTOR_IDX]

    def get_current_dual_arm_q(self) -> Optional[np.ndarray]:
        """Get current positions of both arm motors (10 DOF)."""
        state = self._lowstate_buffer.get()
        if state is None:
            return None
        q, _dq = state
        return q[_ARM_MOTOR_IDX]

    def get_current_dual_arm_dq(self) -> Optional[np.ndarray]:
        """Get current velocities of both arm motors (10 DOF)."""
        state = self._lowstate_buffer.get()
        if state is None:
            return None
        _q, dq = state
        return dq[_ARM_MOTOR_IDX]

    def get_left_arm_q(self) -> Optional[np.ndarray]:
        """Get current positions of left arm motors (5 DOF)."""
        q = self.get_current_dual_arm_q()
        if q is None:
            return None
        return q[:ARM_DOF]

    def get_right_arm_q(self) -> Optional[np.ndarray]:
        """Get current positions of right arm motors (5 DOF)."""
        q = self.get_current_dual_arm_q()
        if q is None:
            return None
        return q[ARM_DOF:]

    def go_home(
        self,
        release_after: bool = False,
        timeout: float = 5.0,
        tolerance: float = 0.05
    ) -> bool:
        """
        Move arms to home position (all zeros).

        Args:
            release_after: Release control after reaching home
            timeout: Maximum time to wait (seconds)
            tolerance: Position tolerance for home detection (rad)

        Returns:
            True if home position reached within timeout
        """
        logger.info("Moving arms to home position...")

        # Set zero target
        with self._ctrl_lock:
            self._q_target = np.zeros(TOTAL_ARM_DOF)

        # Wait for arms to reach home
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_q = self.get_current_dual_arm_q()
            if current_q is not None and np.all(np.abs(current_q) < tolerance):
                logger.info("Arms reached home position")
                if release_after:
                    self.release_control()
                return True
            time.sleep(0.05)

        logger.warning("Timeout waiting for arms to reach home")
        if release_after:
            self.release_control()
        return False

    def set_velocity_limit(self, limit: float) -> None:
        """Set maximum joint velocity limit (rad/s)."""
        self.velocity_limit = limit
        logger.info(f"Velocity limit set to {limit} rad/s")

    def set_gains(
        self,
        kp_arm: Optional[float] = None,
        kd_arm: Optional[float] = None,
        kp_wrist: Optional[float] = None,
        kd_wrist: Optional[float] = None,
        kp_lock: Optional[float] = None,
        kd_lock: Optional[float] = None
    ) -> None:
        """
        Set control gains.

        Args:
            kp_arm: Position gain for shoulder/elbow joints
            kd_arm: Damping gain for shoulder/elbow joints
            kp_wrist: Position gain for wrist joints
            kd_wrist: Damping gain for wrist joints
            kp_lock: Position gain for locked non-arm joints
            kd_lock: Damping gain for locked non-arm joints
        """
        if kp_arm is not None:
            self.kp_arm = kp_arm
        if kd_arm is not None:
            self.kd_arm = kd_arm
        if kp_wrist is not None:
            self.kp_wrist = kp_wrist
        if kd_wrist is not None:
            self.kd_wrist = kd_wrist
        if kp_lock is not None:
            self.kp_lock = kp_lock
        if kd_lock is not None:
            self.kd_lock = kd_lock

        # Update message if already initialized
        if self._msg is not None:
            for joint_id in G1_23_JointArmIndex:
                if self._is_wrist_motor(joint_id):
                    self._msg.motor_cmd[joint_id].kp = self.kp_wrist
                    self._msg.motor_cmd[joint_id].kd = self.kd_wrist
                else:
                    self._msg.motor_cmd[joint_id].kp = self.kp_arm
                    self._msg.motor_cmd[joint_id].kd = self.kd_arm

        logger.info(f"Gains updated: kp_arm={self.kp_arm}, kd_arm={self.kd_arm}, "
                    f"kp_wrist={self.kp_wrist}, kd_wrist={self.kd_wrist}")

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    @property
    def weight(self) -> float:
        """Get current weight parameter."""
        return self._weight

    @property
    def has_control(self) -> bool:
        """True while this SDK owns the arms (weight 1, or lowcmd mode)."""
        return self._has_control
