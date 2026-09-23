# -*- coding: utf-8 -*-
"""
Elevator-button MLP IK.

Loads a SEPARATE pair of MLP models (left/right) trained for the
elevator-button pressing pose. Must not share state with the grasp IK in
solve_grasp_pose.py — the model weights and the pose convention differ.

Input convention (matches the training config.json shipped with the
models, see g1_arm_abs/models/elevator_button/{left,right}/config.json):

    [tip_x, tip_y, tip_z, ee_rx_rad, ee_rz_rad]

All five values are in METERS / RADIANS. rx and rz are end-effector
orientation angles expressed in radians; callers that have degrees must
convert before invoking solve_elevator_button_ik().

Output: 5 arm joint angles in radians, ordered as
    [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll]
i.e. the same 5-DOF layout used everywhere else in g1_arm_abs.

The yaw (rz) rule mirrors solve_grasp_pose.compute_grasp_and_pregrasp:
    right hand: rz = atan2(target_y + 0.17, target_x)
    left  hand: rz = atan2(target_y - 0.17, target_x)
rx is fixed per arm:
    right hand: rx = -90 deg (-pi/2 rad)
    left  hand: rx = +90 deg (+pi/2 rad)
"""

import math
import os
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# Model files. The directory layout matches what the user shipped:
#   g1_arm_abs/models/elevator_button/left/{best_model.pth, x_scaler.pkl}
#   g1_arm_abs/models/elevator_button/right/{best_model.pth, x_scaler.pkl}
# Kept fully separate from the grasp model directory (models/grasp_bottle/{left,right}/)
# so reloading or retraining one set never touches the other.
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../g1_arm_abs

ELEVATOR_BUTTON_MODELS_ROOT = os.path.join(
    _PKG_ROOT, "models", "elevator_button"
)

_MODEL_FILE = "best_model.pth"

LEFT_MODEL_PATH = os.path.join(
    ELEVATOR_BUTTON_MODELS_ROOT, "left", _MODEL_FILE
)
LEFT_SCALER_PATH = os.path.join(
    ELEVATOR_BUTTON_MODELS_ROOT, "left", "x_scaler.pkl"
)
RIGHT_MODEL_PATH = os.path.join(
    ELEVATOR_BUTTON_MODELS_ROOT, "right", _MODEL_FILE
)
RIGHT_SCALER_PATH = os.path.join(
    ELEVATOR_BUTTON_MODELS_ROOT, "right", "x_scaler.pkl"
)


# Per-arm fixed elevator-button rx (degrees -> radians). Pointing finger
# angled at the button face. The two arms use opposite signs so each
# wrist roll lands in the natural side of its joint range.
RIGHT_RX_FIXED_DEG = -90.0
LEFT_RX_FIXED_DEG = +90.0
RIGHT_RX_FIXED_RAD = math.radians(RIGHT_RX_FIXED_DEG)
LEFT_RX_FIXED_RAD = math.radians(LEFT_RX_FIXED_DEG)


def compute_rx_rad(arm: str) -> float:
    """Return the fixed rx (radians) for the given arm: left=+pi/2, right=-pi/2."""
    arm = (arm or "").strip().lower()
    if arm == "left":
        return LEFT_RX_FIXED_RAD
    if arm == "right":
        return RIGHT_RX_FIXED_RAD
    raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")


def compute_rx_deg(arm: str) -> float:
    """Return the fixed rx (degrees) for the given arm: left=+90, right=-90."""
    return math.degrees(compute_rx_rad(arm))


_HIDDEN_DIMS = [256, 512, 256]


def _select_torch_device() -> torch.device:
    # CPU-only while the GX10 power fault makes GPU load power-cut the
    # machine; delete the next line to restore CUDA auto-detection.
    return torch.device("cpu")
    # torch.cuda.is_available() can return True for a GPU whose compute
    # capability this PyTorch build can't actually run on. Probe with a
    # tiny op and fall back to CPU on failure (same trick as
    # solve_grasp_pose._select_torch_device).
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


_DEVICE = _select_torch_device()


class _MLPRegressor(nn.Module):
    """Mirror of the training-side MLPRegressor (input_dim=5, output_dim=5)."""

    def __init__(self, input_dim=5, output_dim=5, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = list(_HIDDEN_DIMS)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Lazy global cache — models load on first solve_elevator_button_ik() call
# instead of at import time, so just importing this module from a test
# script doesn't pay the torch.load cost. Set both entries to None until
# the first hit.
_MODELS = {"left": None, "right": None}
_SCALERS = {"left": None, "right": None}


def _resolve_paths(arm: str) -> Tuple[str, str]:
    arm = (arm or "").strip().lower()
    if arm == "left":
        return LEFT_MODEL_PATH, LEFT_SCALER_PATH
    if arm == "right":
        return RIGHT_MODEL_PATH, RIGHT_SCALER_PATH
    raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")


def _load_arm(arm: str) -> Tuple[object, object]:
    arm = arm.strip().lower()
    if _MODELS[arm] is not None and _SCALERS[arm] is not None:
        return _MODELS[arm], _SCALERS[arm]

    model_path, scaler_path = _resolve_paths(arm)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"elevator-button MLP model not found for {arm}: {model_path}"
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"elevator-button MLP scaler not found for {arm}: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    model = _MLPRegressor(
        input_dim=5, output_dim=5, hidden_dims=_HIDDEN_DIMS, dropout=0.0,
    ).to(_DEVICE)
    state = torch.load(model_path, map_location=_DEVICE)
    state_dict = (
        state["model_state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state
        else state
    )
    model.load_state_dict(state_dict)
    model.eval()

    _MODELS[arm] = model
    _SCALERS[arm] = scaler
    return model, scaler


def get_model_paths(arm: str) -> Tuple[str, str]:
    """Return (model_path, scaler_path) for `arm` — handy for logging."""
    return _resolve_paths(arm)


def compute_rz_rad(arm: str, target_xyz: Sequence[float]) -> float:
    """
    Compute the yaw (rz) angle for the elevator-button pose using the SAME
    rule as the grasp pipeline (solve_grasp_pose.compute_grasp_and_pregrasp):
        right hand: rz = atan2(target_y + 0.17, target_x)
        left  hand: rz = atan2(target_y - 0.17, target_x)
    Returns radians.
    """
    arm = arm.strip().lower()
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")
    x = float(target_xyz[0])
    y = float(target_xyz[1])
    if arm == "right":
        return math.atan2(y + 0.17, x)
    return math.atan2(y - 0.17, x)


def solve_elevator_button_ik(
    arm: str,
    target_xyz: Sequence[float],
    rx_rad: Optional[float] = None,
) -> np.ndarray:
    """
    Predict 5 joint angles (radians) for the elevator-button pressing pose.

    Args:
        arm:       'left' or 'right'.
        target_xyz: (x, y, z) of the button face, in METERS, in the same
                   base frame used by the grasp pipeline (torso_link).
        rx_rad:    Optional rx override in RADIANS. None -> use the arm's
                   fixed default (compute_rx_rad(arm): +pi/2 for left,
                   -pi/2 for right). Callers should normally leave this
                   as None.

    Returns:
        np.ndarray shape (5,), joint angles in RADIANS, in the order
            [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll]
        of the active arm.
    """
    if target_xyz is None or len(target_xyz) < 3:
        raise ValueError(f"target_xyz must have 3 elements, got {target_xyz!r}")

    arm = arm.strip().lower()
    model, scaler = _load_arm(arm)

    x = float(target_xyz[0])
    y = float(target_xyz[1])
    z = float(target_xyz[2])
    rx = float(rx_rad) if rx_rad is not None else compute_rx_rad(arm)
    rz = compute_rz_rad(arm, (x, y, z))

    pose5 = np.array([[x, y, z, rx, rz]], dtype=np.float32)

    with torch.no_grad():
        pose_scaled = scaler.transform(pose5)
        tensor = torch.tensor(pose_scaled, dtype=torch.float32, device=_DEVICE)
        joints = model(tensor).cpu().numpy()[0]

    return np.asarray(joints, dtype=float).reshape(-1)


__all__ = [
    "ELEVATOR_BUTTON_MODELS_ROOT",
    "LEFT_MODEL_PATH",
    "LEFT_SCALER_PATH",
    "RIGHT_MODEL_PATH",
    "RIGHT_SCALER_PATH",
    "LEFT_RX_FIXED_DEG",
    "LEFT_RX_FIXED_RAD",
    "RIGHT_RX_FIXED_DEG",
    "RIGHT_RX_FIXED_RAD",
    "compute_rx_deg",
    "compute_rx_rad",
    "compute_rz_rad",
    "get_model_paths",
    "solve_elevator_button_ik",
]
