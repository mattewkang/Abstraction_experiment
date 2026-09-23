"""
Utility functions for G1 arm control.

Provides velocity clipping, thread-safe data buffer, and other helpers.
"""

import numpy as np
import threading
from typing import Optional, Any


def rad_list_to_deg(q_rad):
    return np.degrees(np.array(q_rad, dtype=float)).tolist()


def deg_list_to_rad(q_deg):
    return np.radians(np.array(q_deg, dtype=float)).tolist()


def merge_close_points_rad(path_rad, joint_thresh_rad=0.03):
    if len(path_rad) <= 1:
        return path_rad

    merged = [np.array(path_rad[0], dtype=float)]
    for q in path_rad[1:]:
        q = np.array(q, dtype=float)
        prev = merged[-1]
        if np.max(np.abs(q - prev)) < joint_thresh_rad:
            continue
        merged.append(q)
    return [q.tolist() for q in merged]


class DataBuffer:
    """Thread-safe data buffer for sharing data between threads."""

    def __init__(self):
        self._data = None
        self._has_data = False
        self._lock = threading.Lock()

    def get(self) -> Optional[Any]:
        """Get current data (thread-safe)."""
        with self._lock:
            return self._data

    def set(self, data: Any) -> None:
        """Set data (thread-safe)."""
        with self._lock:
            self._data = data
            self._has_data = True

    def has_data(self) -> bool:
        """Check if data has been set (thread-safe)."""
        with self._lock:
            return self._has_data

    def clear(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self._lock:
            self._data = None
            self._has_data = False

    # Aliases for compatibility with existing code
    def GetData(self) -> Optional[Any]:
        return self.get()

    def SetData(self, data: Any) -> None:
        self.set(data)


def clip_arm_q_target(
    target_q: np.ndarray,
    current_q: np.ndarray,
    velocity_limit: float = 20.0,
    dt: float = 0.004
) -> np.ndarray:
    """
    Clip target joint angles to respect velocity limits.

    Ensures smooth motion by limiting the maximum change in joint angles
    per control cycle based on velocity constraints.

    Args:
        target_q: Target joint angles (rad)
        current_q: Current joint angles (rad)
        velocity_limit: Maximum joint velocity (rad/s)
        dt: Control loop period (s), default 4ms for 250Hz

    Returns:
        Clipped target joint angles
    """
    delta = target_q - current_q
    max_delta = velocity_limit * dt

    # Scale factor based on maximum delta (always apply max() for numerical consistency)
    motion_scale = np.max(np.abs(delta)) / max_delta
    clipped_target = current_q + delta / max(motion_scale, 1.0)

    return clipped_target


def interpolate_joint_angles(
    start_q: np.ndarray,
    end_q: np.ndarray,
    duration: float,
    frequency: float = 250.0
) -> np.ndarray:
    """
    Generate interpolated joint angle trajectory.

    Args:
        start_q: Starting joint angles
        end_q: Ending joint angles
        duration: Motion duration (s)
        frequency: Control frequency (Hz)

    Returns:
        Array of shape (num_steps, num_joints) with interpolated angles
    """
    num_steps = int(duration * frequency)
    if num_steps < 1:
        num_steps = 1

    # Linear interpolation
    t = np.linspace(0, 1, num_steps)
    trajectory = np.outer(1 - t, start_q) + np.outer(t, end_q)

    return trajectory


def smooth_trajectory(
    trajectory: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Apply moving average smoothing to trajectory.

    Args:
        trajectory: Joint angle trajectory (num_steps, num_joints)
        window_size: Smoothing window size

    Returns:
        Smoothed trajectory
    """
    if window_size < 2 or len(trajectory) < window_size:
        return trajectory

    smoothed = np.zeros_like(trajectory)
    half_window = window_size // 2

    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)
        smoothed[i] = np.mean(trajectory[start:end], axis=0)

    return smoothed


class WeightTransition:
    """
    Manages smooth weight parameter transitions for motion mode.

    Weight parameter controls the blend between motion control and SDK commands:
    - weight = 0: Robot uses motion control only
    - weight = 1: Robot uses SDK commands only
    """

    def __init__(
        self,
        num_steps: int = 101,
        step_interval: float = 0.02
    ):
        """
        Initialize weight transition manager.

        Args:
            num_steps: Number of steps for transition (default: 101)
            step_interval: Time between steps in seconds (default: 0.02s)
        """
        self.num_steps = num_steps
        self.step_interval = step_interval
        self._current_weight = 0.0
        self._target_weight = 0.0
        self._start_weight = 0.0
        self._step_index = 0
        self._transitioning = False
        self._lock = threading.Lock()

    @property
    def current_weight(self) -> float:
        """Get current weight value."""
        with self._lock:
            return self._current_weight

    @property
    def is_transitioning(self) -> bool:
        """Check if transition is in progress."""
        with self._lock:
            return self._transitioning

    def start_transition(self, target_weight: float) -> None:
        """
        Start transition to target weight.

        Args:
            target_weight: Target weight value (0.0 to 1.0)
        """
        with self._lock:
            self._target_weight = np.clip(target_weight, 0.0, 1.0)
            self._step_index = 0
            self._transitioning = True

    def step(self) -> float:
        """
        Advance transition by one step.

        Returns:
            Current weight value after step
        """
        with self._lock:
            if not self._transitioning:
                return self._current_weight

            # Linear interpolation from current weight at step 0
            progress = self._step_index / (self.num_steps - 1)
            if self._step_index == 0:
                self._start_weight = self._current_weight
            self._current_weight = self._start_weight + (self._target_weight - self._start_weight) * progress

            self._step_index += 1
            if self._step_index >= self.num_steps:
                self._current_weight = self._target_weight
                self._transitioning = False

            return self._current_weight

    def set_immediate(self, weight: float) -> None:
        """
        Set weight immediately without transition.

        Args:
            weight: Weight value (0.0 to 1.0)
        """
        with self._lock:
            self._current_weight = np.clip(weight, 0.0, 1.0)
            self._target_weight = self._current_weight
            self._transitioning = False
