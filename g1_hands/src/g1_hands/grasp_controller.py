import asyncio
import logging
from typing import List


class TactileGraspController:
    """Direct finger control over a Revo2 Modbus client. No feedback algorithm."""

    def __init__(self, client, slave_id: int):
        self.client = client
        self.slave_id = slave_id

        self.logger = logging.getLogger(__name__)

        self.current_currents = [0] * 6
        self.last_positions = [0] * 6

    def update_currents(self, currents: List[int]):
        if len(currents) != 6:
            raise ValueError("currents must be length 6")
        self.current_currents = currents.copy()

    async def set_currents(self, currents: List[int]):
        """Set the 6 joint currents directly."""
        self.update_currents(currents)
        await self.client.set_finger_currents(self.slave_id, currents)
        self.logger.info(f"[Controller] set currents: {currents}")

    async def grasp_with_current(self, finger_mask: List[int]):
        """
        Coarse current-based grasp.

        finger_mask[i] convention:
            > 0  close
            < 0  open
            = 0  hold
        Thumb flex (index 1) is pinned to +500.
        """
        if len(finger_mask) != 6:
            raise ValueError("finger_mask must be length 6")

        currents = []
        for i, val in enumerate(finger_mask):
            if i == 1:
                currents.append(500)
            else:
                if val > 0:
                    currents.append(100)
                elif val < 0:
                    currents.append(-500)
                else:
                    currents.append(0)

        await self.set_currents(currents)

    async def _wait_positions_reached(
        self,
        targets: List[int],
        mask: List[int],
        tolerance: int,
        timeout: float,
        poll_interval: float = 0.05,
        stable_window: int = 4,
        stable_tol: int = 5,
    ) -> bool:
        """
        Wait until motion has stopped on every finger in `mask`. Success when
        EITHER:
          (a) all masked fingers are within `tolerance` of their target, OR
          (b) all masked fingers have been stable (max-change <= stable_tol)
              over the last `stable_window` polls (i.e. the finger has hit a
              mechanical limit or an object and is no longer moving).

        Returns True on success, False on timeout.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        history: List[List[int]] = []
        last_err = None
        current = None
        while True:
            try:
                current = await self.client.get_finger_positions(self.slave_id)
            except Exception as e:
                last_err = e
                current = None

            if current is not None and len(current) == 6:
                current_int = [int(v) for v in current]

                if all(
                    abs(current_int[i] - int(targets[i])) <= tolerance
                    for i in mask
                ):
                    return True

                history.append(current_int)
                if len(history) > stable_window:
                    history.pop(0)
                if len(history) == stable_window and all(
                    max(s[i] for s in history) - min(s[i] for s in history) <= stable_tol
                    for i in mask
                ):
                    self.logger.info(
                        f"[Controller] wait_positions: motion settled "
                        f"target={targets} current={current_int} mask={mask}"
                    )
                    return True

            if loop.time() >= deadline:
                self.logger.warning(
                    f"[Controller] wait_positions timeout: target={targets} "
                    f"mask={mask} current={current} last_err={last_err}"
                )
                return False

            await asyncio.sleep(poll_interval)

    async def set_positions(
        self,
        positions: List[int],
        wait: bool = True,
        mask: List[int] = None,
        tolerance: int = 50,
        timeout: float = 2.0,
    ) -> bool:
        """
        Set the 6 joint positions. With `wait=True`, block until the masked
        fingers reach target (or the wait helper returns False on timeout).
        """
        if len(positions) != 6:
            raise ValueError("positions must be length 6")

        await self.client.set_finger_positions(self.slave_id, positions)
        self.last_positions = positions.copy()
        self.logger.info(f"[Controller] set positions: {positions} (wait={wait})")

        if not wait:
            return True

        if mask is None:
            mask = list(range(6))
        return await self._wait_positions_reached(
            positions, mask, tolerance=tolerance, timeout=timeout,
        )

    async def grasp(
        self,
        num_fingers: int = 5,
        settle_seconds: float = 1.0,
    ) -> bool:
        """
        Position-controlled grasp. Fingers will be blocked by the object
        before reaching the commanded target, so this issues the command
        and waits a fixed `settle_seconds` instead of polling positions.
        """
        if num_fingers < 1 or num_fingers > 5:
            raise ValueError("num_fingers must be 1~5")

        base = [500, 1000, 500, 500, 500, 500]
        positions = base[:num_fingers + 1] + [0] * (5 - num_fingers)

        await self.set_positions(positions, wait=False)
        if settle_seconds > 0:
            await asyncio.sleep(float(settle_seconds))
        return True

    async def release_all(
        self,
        release_positions: List[int] = None,
        timeout: float = 2.0,
    ) -> bool:
        """Release to a relaxed natural pose."""
        if release_positions is None:
            release_positions = [180, 260, 140, 140, 140, 140]
        if len(release_positions) != 6:
            raise ValueError("release_positions must be length 6")
        return await self.set_positions(
            [int(v) for v in release_positions],
            wait=True,
            timeout=timeout,
        )

    async def move_single_finger(
        self,
        finger_index: int,
        position: int,
        tolerance: int = 50,
        timeout: float = 2.0,
    ) -> bool:
        """
        Move a single finger and block until it reaches the target
        (or the wait helper returns False on timeout).

        finger_index:
            0 Thumb base
            1 Thumb flex
            2 Index
            3 Middle
            4 Ring
            5 Pinky
        """
        if finger_index < 0 or finger_index > 5:
            raise ValueError("finger_index must be 0~5")

        positions = self.last_positions.copy()
        positions[finger_index] = position

        return await self.set_positions(
            positions,
            wait=True,
            mask=[finger_index],
            tolerance=tolerance,
            timeout=timeout,
        )

    async def stop(self):
        """Stop all motion by zeroing currents."""
        await self.set_currents([0] * 6)
        self.logger.info("[Controller] stopped")
