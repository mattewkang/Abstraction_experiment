# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/nn_backend.py

PyTorch NN IK plumbing for the grasp MLPs, extracted from ``arm_node_abs.py``.

This is the single home for the node's grasp IK backend. It deliberately keeps
the node's exact policies (which differ from ``solve_grasp_pose.py``):

- ``MLPRegressor`` includes ``Dropout`` layers when ``dropout > 0`` (the
  ``solve_grasp_pose`` copy omits them).
- ``make_device`` selects the device from an explicit ``use_cuda`` flag and
  ``torch.cuda.is_available()`` only — it does NOT run the tiny-op CUDA probe
  that ``solve_grasp_pose._select_torch_device`` uses. Do not collapse the two.
- ``load_ai_tools`` raises ``FileNotFoundError`` on a missing weight/scaler when
  ``raise_on_missing`` is true (the node relies on the raise); set it false to
  get the print-and-return-``(None, None)`` behaviour instead.

The package is torch-only: the runtime always has PyTorch, so there is no
ONNX / Jetson backend here. Do not reintroduce one.
"""

import os
from typing import Tuple

import joblib
import numpy as np

import torch
import torch.nn as nn

from g1_arm_abs.pose_compensation import PoseCompensationConfig, compensate_pose

__all__ = [
    "make_device",
    "load_ai_tools",
    "infer_joints",
    "solve_ai_ik",
]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=5, output_dim=5, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
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


def make_device(use_cuda: bool):
    """Build the torch device from the ``~use_cuda`` flag.

    Uses the flag + ``torch.cuda.is_available()`` only — no tiny-op CUDA probe.
    """
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def load_ai_tools(
    model_path: str,
    scaler_path: str,
    device,
    hidden_dims=None,
    dropout: float = 0.0,
    raise_on_missing: bool = True,
) -> Tuple[object, object]:
    """Load an MLP model + its scaler.

    With ``raise_on_missing`` true (the node default), a missing model or
    scaler raises ``FileNotFoundError``; otherwise it prints a warning and
    returns ``(None, None)``.
    """
    if not os.path.exists(model_path):
        if raise_on_missing:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"警告: 找不到模型或权重文件: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        if raise_on_missing:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        print(f"警告: 找不到模型或权重文件: {scaler_path}")
        return None, None

    scaler = joblib.load(scaler_path)

    model = MLPRegressor(
        input_dim=5, output_dim=5,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    state_dict = (
        state["model_state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state
        else state
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


def infer_joints(model, scaler, pose5, device) -> np.ndarray:
    """Run one MLP forward pass on an already-compensated 5-D pose.

    ``pose5`` is ``[x, y, z, rx, rz]`` (compensation, if any, is applied by the
    caller). Returns the predicted 5-DOF joint vector.
    """
    x, y, z, rx, rz = pose5
    pose = np.array([x, y, z, rx, rz], dtype=np.float32).reshape(1, -1)

    with torch.no_grad():
        pose_scaled = scaler.transform(pose)
        input_tensor = torch.tensor(pose_scaled, dtype=torch.float32, device=device)
        joints = model(input_tensor).cpu().numpy()[0]

    return joints


def solve_ai_ik(
    pose5,
    hand: str,
    models: dict,
    scalers: dict,
    comp_cfg: PoseCompensationConfig,
    device,
) -> np.ndarray:
    """Per-arm grasp IK: optional pose compensation then one MLP forward pass.

    ``models`` / ``scalers`` are ``{"left": ..., "right": ...}`` dicts. ``hand``
    selects ``"right"`` only when exactly ``"right"``, else ``"left"`` (matches
    the original node behaviour). Raises ``RuntimeError`` if the selected model
    or scaler is not loaded.
    """
    x, y, z, rx, rz = pose5

    if comp_cfg.use_compensation:
        x, y, z = compensate_pose(x, y, z, hand, comp_cfg)

    if hand == "right":
        model, scaler = models["right"], scalers["right"]
    else:
        model, scaler = models["left"], scalers["left"]

    if model is None or scaler is None:
        raise RuntimeError(f"NN model for {hand} arm is not loaded")

    return infer_joints(model, scaler, (x, y, z, rx, rz), device)
