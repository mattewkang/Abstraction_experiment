#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float32

import asyncio
import random
import threading

from bc_stark_sdk import main_mod as libstark
from g1_hands.grasp_controller import TactileGraspController
from g1_hands.msg import TactileStatus


async def _robust_open_revo2(port_name, retries, interval, jitter, logger_prefix):
    """
    Retry wrapper around libstark.auto_detect_modbus_revo2 + modbus_open.

    Catches Exception AND SystemExit (older SDK revisions raised SystemExit
    on first failure), waits with small random jitter so two parallel hand
    nodes don't always hit the USB bus on the same tick, and raises
    cleanly only after every attempt fails.
    """
    retries = max(1, int(retries))
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            proto, port, baud, slave = await libstark.auto_detect_modbus_revo2(
                port_name, True  # quick=True — matches revo2_utils default
            )
            if proto != libstark.StarkProtocolType.Modbus:
                raise RuntimeError(f"unexpected protocol {proto}")
            client = await libstark.modbus_open(port, baud)
            info = await client.get_device_info(slave)
            rospy.loginfo(
                f"{logger_prefix} attempt {attempt}/{retries} ok: "
                f"port={port} baud={baud} slave_id={slave} info={info.description}"
            )
            return client, slave
        except BaseException as e:  # catch SystemExit too
            last_exc = e
            rospy.logwarn(
                f"{logger_prefix} attempt {attempt}/{retries} failed on "
                f"{port_name}: {e}"
            )
            if attempt < retries:
                backoff = interval + random.random() * max(0.0, jitter)
                await asyncio.sleep(backoff)

    raise RuntimeError(
        f"auto_detect_modbus_revo2 failed on {port_name} after "
        f"{retries} attempts: {last_exc}"
    )


class Revo2Node:

    def __init__(self):
        # Default node name is 'g1_hand' so the services expose themselves
        # as /g1_hand/pre_grasp, /g1_hand/grasp_5f, /g1_hand/release.
        # A `name=` attribute in a launch file overrides this.
        rospy.init_node('g1_hand')

        # -------------------------------
        # Parameters
        # -------------------------------
        self.port = rospy.get_param('~port', '/dev/ttyUSB1')
        self.slave_id = rospy.get_param('~slave_id', 127)
        # init_timeout must exceed (open_retries * (open_retry_interval +
        # open_retry_jitter) + per-attempt probe time). Defaults give us
        # 5 tries * ~1.3s + ~2s per probe = headroom up to ~25s.
        self.init_timeout = float(rospy.get_param('~init_timeout', 30.0))
        self.open_retries = int(rospy.get_param('~open_retries', 5))
        self.open_retry_interval = float(
            rospy.get_param('~open_retry_interval', 1.0)
        )
        self.open_retry_jitter = float(
            rospy.get_param('~open_retry_jitter', 0.5)
        )
        self.release_positions = list(rospy.get_param(
            '~release_positions',
            [180, 260, 140, 140, 140, 140]
        ))
        if len(self.release_positions) != 6:
            raise ValueError("~release_positions must have 6 values")
        self.release_positions = [max(0, min(1000, int(v))) for v in self.release_positions]

        # Per-service blocking timeouts (seconds). The Trigger callback waits
        # at most this long for the underlying coroutine to finish before
        # returning to the caller.
        self.pre_grasp_timeout = float(rospy.get_param('~pre_grasp_timeout', 3.0))
        self.grasp_5f_timeout  = float(rospy.get_param('~grasp_5f_timeout', 3.0))
        self.release_timeout   = float(rospy.get_param('~release_timeout', 3.0))
        self.point_gesture_timeout = float(rospy.get_param('~point_gesture_timeout', 3.0))
        self.grasp_berry_timeout = float(rospy.get_param('~grasp_berry_timeout', 3.0))
        self.pre_grasp_berry_timeout = float(rospy.get_param('~pre_grasp_berry_timeout', 3.0))

        # Tactile publishing. <=0 disables the background poll entirely so a
        # mis-configured launch can opt out without recompiling.
        self.tactile_publish_hz = float(rospy.get_param('~tactile_publish_hz', 30.0))
        # frame_id stamped on every TactileStatus; auto_release_monitor_node
        # subscribes per-arm and routes by topic, not by frame_id, but
        # downstream consumers may want it.
        node_name = rospy.get_name().lstrip('/')
        self.tactile_frame_id = rospy.get_param(
            '~tactile_frame_id', node_name or 'g1_hand'
        )

        # Finger positions used by /point_gesture (elevator-button pose):
        # index 0=Thumb base, 1=Thumb flex, 2=Index, 3=Middle, 4=Ring, 5=Pinky.
        # 0 = fully extended, 1000 = fully flexed. Index + Middle stay extended
        # while thumb / ring / pinky fold inward.
        self.point_gesture_positions = list(rospy.get_param(
            '~point_gesture_positions',
            [1000, 1000, 0, 0, 1000, 1000]
        ))
        if len(self.point_gesture_positions) != 6:
            raise ValueError("~point_gesture_positions must have 6 values")
        self.point_gesture_positions = [
            max(0, min(1000, int(v))) for v in self.point_gesture_positions
        ]

        # Finger positions used by /grasp_berry (small-object pinch):
        # index 0=Thumb base, 1=Thumb flex, 2=Index, 3=Middle, 4=Ring, 5=Pinky.
        # 0 = fully extended, 1000 = fully flexed. Thumb opposes and thumb-flex
        # + index + middle curl to a PARTIAL close onto a ~1 cm berry while
        # ring + pinky fold away. The flex values are the berry-size tuning
        # knob: too high crushes, too low drops. Fingers blocked by the berry
        # settle early and the wait helper accepts "motion settled" as success.
        self.grasp_berry_positions = list(rospy.get_param(
            '~grasp_berry_positions',
            [410, 900, 490, 490, 1000, 1000]
        ))
        if len(self.grasp_berry_positions) != 6:
            raise ValueError("~grasp_berry_positions must have 6 values")
        self.grasp_berry_positions = [
            max(0, min(1000, int(v))) for v in self.grasp_berry_positions
        ]

        # Finger positions used by /pre_grasp_berry (berry approach pose):
        # thumb + index + middle partially pre-curled (close to the grasp pose
        # but looser), ring + pinky folded away so they don't foul the approach.
        self.pre_grasp_berry_positions = list(rospy.get_param(
            '~pre_grasp_berry_positions',
            [300, 900, 300, 300, 1000, 1000]
        ))
        if len(self.pre_grasp_berry_positions) != 6:
            raise ValueError("~pre_grasp_berry_positions must have 6 values")
        self.pre_grasp_berry_positions = [
            max(0, min(1000, int(v))) for v in self.pre_grasp_berry_positions
        ]

        self.client = None
        self.controller = None
        # Serializes coroutines that share the Modbus client. The tactile
        # poll holds this while reading sensors so a concurrent /release
        # request can't fragment a transaction. Created on the asyncio
        # loop's thread inside async_init.
        self._client_lock = None

        # -------------------------------
        # Publishers
        # -------------------------------
        self.stiffness_pub = rospy.Publisher(
            'stiffness', Float32, queue_size=10
        )
        # ~tactile resolves to /<node_name>/tactile, e.g.
        # /g1_hand_right/tactile. auto_release_monitor_node subscribes
        # to both per-hand topics.
        self.tactile_pub = rospy.Publisher(
            '~tactile', TactileStatus, queue_size=5
        )

        # -------------------------------
        # Asyncio loop in thread
        # -------------------------------
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.thread.start()

        # -------------------------------
        # Init hardware async
        # -------------------------------
        init_future = asyncio.run_coroutine_threadsafe(self.async_init(), self.loop)
        try:
            init_future.result(timeout=self.init_timeout)
        except Exception as e:
            rospy.logerr(f"Failed to initialize Revo2 controller: {e}")
            rospy.signal_shutdown("Revo2 init failed")
            return

        # -------------------------------
        # ROS services
        # -------------------------------
        # Private services -> /<node_name>/<service>, i.e. /g1_hand/pre_grasp
        rospy.Service('~pre_grasp', Trigger, self.pre_grasp_cb)
        rospy.Service('~grasp_5f', Trigger, self.grasp_5f_cb)
        rospy.Service('~release', Trigger, self.release_cb)
        rospy.Service('~point_gesture', Trigger, self.point_gesture_cb)
        rospy.Service('~grasp_berry', Trigger, self.grasp_berry_cb)
        rospy.Service('~pre_grasp_berry', Trigger, self.pre_grasp_berry_cb)

        rospy.loginfo("Revo2 ROS1 node ready")

    # =========================================
    # Async init
    # =========================================
    async def async_init(self):
        try:
            # Stagger the very first probe by a small random delay so when
            # two hand nodes start in parallel (revo2_dual_hand_node.launch)
            # they don't always hit the USB/Modbus stack on the same tick.
            initial_jitter = random.random() * self.open_retry_jitter
            if initial_jitter > 0:
                await asyncio.sleep(initial_jitter)

            node_name = rospy.get_name()
            self.client, self.slave_id = await _robust_open_revo2(
                port_name=self.port,
                retries=self.open_retries,
                interval=self.open_retry_interval,
                jitter=self.open_retry_jitter,
                logger_prefix=f"[{node_name}]",
            )

            self.controller = TactileGraspController(
                self.client, self.slave_id
            )

            # asyncio.Lock must be created on the loop's thread.
            self._client_lock = asyncio.Lock()

            rospy.loginfo("Revo2 controller initialized")

            if self.tactile_publish_hz > 0.0:
                # Schedule the tactile poll as a long-running task on this
                # same loop. Don't await it -- we want it to run forever
                # alongside the service callbacks.
                self.loop.create_task(self._tactile_poll_forever())
                rospy.loginfo(
                    f"tactile poll scheduled at {self.tactile_publish_hz:.1f} Hz"
                )
            else:
                rospy.loginfo("tactile poll disabled (~tactile_publish_hz<=0)")
        except Exception as e:
            rospy.logerr(f"async_init failed: {e}")
            raise

    # =========================================
    # Tactile polling
    # =========================================
    async def _locked_call(self, coro):
        # Serializes Modbus transactions. The asyncio loop is single-threaded
        # but coroutines can yield at internal awaits, allowing another
        # coroutine to interleave a Modbus frame; the lock prevents that.
        async with self._client_lock:
            return await coro

    async def _tactile_poll_forever(self):
        period = 1.0 / max(1e-3, self.tactile_publish_hz)
        consecutive_errors = 0
        while not rospy.is_shutdown():
            try:
                async with self._client_lock:
                    status = await self.client.get_touch_sensor_status(
                        self.slave_id
                    )
                consecutive_errors = 0
                self._publish_tactile(status)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                consecutive_errors += 1
                rospy.logwarn_throttle(
                    5.0,
                    f"tactile poll error #{consecutive_errors}: {e}",
                )
                # Back off briefly on errors so we don't hammer the bus.
                await asyncio.sleep(min(1.0, period * 5.0))
                continue
            await asyncio.sleep(period)

    def _publish_tactile(self, status):
        msg = TactileStatus()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.tactile_frame_id
        # Five fingers, all four channels. If the SDK ever returns fewer
        # entries (firmware regression), pad with NaN so consumers can
        # detect dead channels rather than reading silent zeros.
        normal = []
        tangential = []
        tangential_dir = []
        proximity = []
        for i in range(5):
            try:
                f = status[i]
            except (IndexError, KeyError, TypeError):
                normal.append(float('nan'))
                tangential.append(float('nan'))
                tangential_dir.append(float('nan'))
                proximity.append(float('nan'))
                continue
            normal.append(float(getattr(f, 'normal_force1', float('nan'))))
            tangential.append(float(getattr(f, 'tangential_force1', float('nan'))))
            tangential_dir.append(float(getattr(f, 'tangential_direction1', float('nan'))))
            proximity.append(float(getattr(f, 'self_proximity1', float('nan'))))
        msg.normal_force = normal
        msg.tangential_force = tangential
        msg.tangential_direction = tangential_dir
        msg.self_proximity = proximity
        try:
            self.tactile_pub.publish(msg)
        except Exception as e:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, f"tactile publish failed: {e}")

    def _check_controller_ready(self):
        if self.controller is None:
            return TriggerResponse(
                success=False,
                message="Revo2 controller not initialized",
            )
        return None

    # =========================================
    # Service callbacks
    # =========================================
    def _run_blocking(self, coro, wait_timeout, label):
        """
        Schedule `coro` on the asyncio loop and block (in the ROS service
        thread) until it completes or `wait_timeout` elapses. Returns
        (success, message).
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            # Allow a small margin so the coroutine's own internal timeout
            # has time to expire and report a clean False before we cancel.
            result = future.result(timeout=wait_timeout + 1.0)
        except asyncio.TimeoutError:
            future.cancel()
            return False, f"{label} timed out after {wait_timeout:.2f}s"
        except Exception as e:
            return False, f"{label} raised: {e}"

        if result is False:
            return False, f"{label} did not reach target within timeout"
        return True, label

    def grasp_5f_cb(self, req):
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        ok, msg = self._run_blocking(
            self._locked_call(self.controller.grasp(num_fingers=5)),
            wait_timeout=self.grasp_5f_timeout,
            label="5-finger grasp",
        )
        return TriggerResponse(success=ok, message=msg)

    def pre_grasp_cb(self, req):
        """
        Pre-grasp pose: thumb base (joint 1) closed to 1000, every other
        finger fully open at 0. Blocks until the thumb has reached 1000
        (or motion settles), so callers know the hand is ready before the
        arm starts moving toward the grasp pose.
        """
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        positions = [0, 1000, 0, 0, 0, 0]
        ok, msg = self._run_blocking(
            self._locked_call(
                self.controller.set_positions(positions, wait=True, mask=[1])
            ),
            wait_timeout=self.pre_grasp_timeout,
            label="Pre-grasp",
        )
        return TriggerResponse(success=ok, message=msg)

    def point_gesture_cb(self, req):
        """
        Elevator-button gesture: fold thumb (base + flex) and ring + pinky;
        keep index and middle fully extended. Blocks until index + middle
        have reached the open target (or motion settles), so the orchestrator
        knows the hand is in the pressing pose before the arm starts moving.
        """
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        # Wait on the index + middle fingers (slots 2, 3) since those are the
        # ones the press relies on. Folded fingers may hit a mechanical limit
        # and "settle" early, which the controller already handles.
        ok, msg = self._run_blocking(
            self._locked_call(
                self.controller.set_positions(
                    self.point_gesture_positions, wait=True, mask=[2, 3]
                )
            ),
            wait_timeout=self.point_gesture_timeout,
            label=f"Point gesture to {self.point_gesture_positions}",
        )
        return TriggerResponse(success=ok, message=msg)

    def grasp_berry_cb(self, req):
        """
        Berry pinch: thumb opposes and thumb-flex + index + middle curl to a
        partial close onto a small berry; ring + pinky fold away. Waits on the
        pinching fingers (thumb-flex 1, index 2, middle 3); those are blocked by
        the berry before reaching target, so the wait helper's "motion settled"
        path reports success.
        """
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        ok, msg = self._run_blocking(
            self._locked_call(
                self.controller.set_positions(
                    self.grasp_berry_positions, wait=True, mask=[1, 2, 3]
                )
            ),
            wait_timeout=self.grasp_berry_timeout,
            label=f"Grasp berry to {self.grasp_berry_positions}",
        )
        return TriggerResponse(success=ok, message=msg)

    def pre_grasp_berry_cb(self, req):
        """
        Berry approach pose: thumb + index + middle partially pre-curled (a
        looser version of the grasp pose), ring + pinky folded away. Waits on
        thumb-flex (slot 1) so the caller knows the hand is staged before the
        arm moves toward the berry; the folded ring/pinky may settle early,
        which the wait helper accepts.
        """
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        ok, msg = self._run_blocking(
            self._locked_call(
                self.controller.set_positions(
                    self.pre_grasp_berry_positions, wait=True, mask=[1]
                )
            ),
            wait_timeout=self.pre_grasp_berry_timeout,
            label=f"Pre-grasp berry to {self.pre_grasp_berry_positions}",
        )
        return TriggerResponse(success=ok, message=msg)

    def release_cb(self, req):
        not_ready = self._check_controller_ready()
        if not_ready is not None:
            return not_ready

        ok, msg = self._run_blocking(
            self._locked_call(
                self.controller.release_all(self.release_positions)
            ),
            wait_timeout=self.release_timeout,
            label=f"Release to {self.release_positions}",
        )
        return TriggerResponse(success=ok, message=msg)


# =========================================
# Main
# =========================================
if __name__ == '__main__':
    node = Revo2Node()
    rospy.spin()
