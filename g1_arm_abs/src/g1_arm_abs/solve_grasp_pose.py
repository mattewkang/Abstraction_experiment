import math
import os
import numpy as np
import joblib

import torch
import torch.nn as nn

# =========================================================
# 🔥 路径配置（必须和主程序一致）
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # .../g1_arm_abs
GRASP_BOTTLE_MODELS_ROOT = os.path.join(_PKG_ROOT, "models", "grasp_bottle")

_MODEL_FILE = "best_model.pth"

RIGHT_MODEL_PATH  = os.path.join(GRASP_BOTTLE_MODELS_ROOT, "right", _MODEL_FILE)
RIGHT_SCALER_PATH = os.path.join(GRASP_BOTTLE_MODELS_ROOT, "right", "x_scaler.pkl")

LEFT_MODEL_PATH   = os.path.join(GRASP_BOTTLE_MODELS_ROOT, "left", _MODEL_FILE)
LEFT_SCALER_PATH  = os.path.join(GRASP_BOTTLE_MODELS_ROOT, "left", "x_scaler.pkl")


# =========================================================
# 🔥 PyTorch backend helpers
# =========================================================
def _select_torch_device():
    # CPU-only while the GX10 power fault makes GPU load power-cut the
    # machine; delete the next line to restore CUDA auto-detection.
    return torch.device("cpu")
    # torch.cuda.is_available() returns True even when this PyTorch build has
    # no kernel for the GPU's compute capability (e.g. RTX 5070 sm_120 on a
    # cu124 build). Probe with a tiny op and fall back to CPU on failure.
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


DEVICE = _select_torch_device()

RIGHT_HIDDEN_DIMS = [256, 512, 256]
LEFT_HIDDEN_DIMS  = [256, 512, 256]


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
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# 🔥 加载模型
# =========================================================
def load_ai_tools(model_path, scaler_path, hidden_dims=None, dropout=0.0):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"警告: 找不到模型或权重文件: {model_path}")
        return None, None

    scaler = joblib.load(scaler_path)

    model = MLPRegressor(
        input_dim=5, output_dim=5,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    state_dict = (
        state["model_state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state
        else state
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


# =========================================================
# 🔥 全局模型（只加载一次）
# =========================================================
r_model, r_scaler = load_ai_tools(
    RIGHT_MODEL_PATH, RIGHT_SCALER_PATH, RIGHT_HIDDEN_DIMS
)
l_model, l_scaler = load_ai_tools(
    LEFT_MODEL_PATH, LEFT_SCALER_PATH, LEFT_HIDDEN_DIMS
)

# =========================================================
# 🔥 坐标补偿（核心）
# =========================================================
def compensate_pose(x, y, z, tilt_deg=2.0, z_offset=0.0541):
    """
    补偿：
    - base前倾2°
    - base高度偏移
    """

    # 平移
    z = z - z_offset

    # 旋转（绕y轴）
    theta = math.radians(tilt_deg)

    x_new = x * math.cos(theta) + z * math.sin(theta)
    y_new = y
    z_new = -x * math.sin(theta) + z * math.cos(theta)

    return x_new, y_new, z_new


# =========================================================
# 🔥 神经网络 IK
# =========================================================
def solve_ik(grasp_pose, hand="right"):
    x, y, z, rx, rz = grasp_pose

    # ⭐ 坐标补偿（非常关键）
    # x, y, z = compensate_pose(x, y, z)

    pose = np.array([x, y, z, rx, rz], dtype=np.float32).reshape(1, -1)

    # 选择模型
    if hand == "right":
        model, scaler = r_model, r_scaler
    else:
        model, scaler = l_model, l_scaler

    if model is None:
        raise RuntimeError("模型未加载成功")

    with torch.no_grad():
        pose_scaled = scaler.transform(pose)
        input_tensor = torch.tensor(pose_scaled, dtype=torch.float32, device=DEVICE)
        joints = model(input_tensor).cpu().numpy()[0]

    return joints


# =========================================================
# 🔥 抓取点 & 预抓取点
# =========================================================
def compute_grasp_and_pregrasp(
    target,
    hand="right",
    delta_rz_deg=40,
    pre_dist=0.12,
    decimals=3
):
    target_x, target_y, target_z = target

    # -------------------------
    # 左右手不同策略
    # grasp_pose 的 rz 仍按 ±0.17 计算（保持原行为）。
    # pregrasp 单独使用一个基于 ±0.07 的 pre_rz_base，
    # 用于：(a) pre_rz 方向偏置，(b) pregrasp_pose 的姿态字段。
    # -------------------------
    if hand == "right":
        rz = math.atan2(target_y + 0.17, target_x)
        pre_rz_base = math.atan2(target_y + 0.07, target_x)
    else:
        rz = math.atan2(target_y - 0.17, target_x)
        pre_rz_base = math.atan2(target_y - 0.07, target_x)
        delta_rz_deg = -delta_rz_deg

    # -------------------------
    # 预抓取角（基于 pre_rz_base，与 grasp 的 rz 解耦）
    # -------------------------
    pre_rz = pre_rz_base + math.radians(delta_rz_deg)

    # -------------------------
    # 预抓取位置
    # -------------------------
    pre_x = target_x - pre_dist * math.cos(pre_rz)
    pre_y = target_y - pre_dist * math.sin(pre_rz)
    pre_z = target_z

    def r(v): return round(v, decimals)

    grasp_pose = (r(target_x), r(target_y), r(target_z), 0.0, r(rz))
    pregrasp_pose = (r(pre_x), r(pre_y), r(pre_z), 0.0, r(pre_rz_base))

    return grasp_pose, pregrasp_pose


# =========================================================
# 🔥 轨迹生成
# =========================================================
def generate_linear_trajectory(pregrasp_pose, grasp_pose, num_points=20):
    x0, y0, z0, rx0, rz0 = pregrasp_pose
    x1, y1, z1, rx1, rz1 = grasp_pose

    traj = []

    for i in range(num_points):
        t = i / (num_points - 1)

        x = x0 * (1 - t) + x1 * t
        y = y0 * (1 - t) + y1 * t
        z = z0 * (1 - t) + z1 * t
        rx = rx0 * (1 - t) + rx1 * t
        rz = rz0 * (1 - t) + rz1 * t

        traj.append((x, y, z, rx, rz))

    return traj


# =========================================================
# 🔥 轨迹 → joints
# =========================================================
def pose5_to_joints_trajactory(traj, hand="right"):
    joints_traj = []

    for pose in traj:
        joints = solve_ik(pose, hand)
        joints_traj.append(joints)

    return joints_traj


def target_to_joints_trajectory(
    target,
    hand="right",
    num_points=10
):
    """
    🔥 一步完成：
    target → grasp → trajectory → joints trajectory

    输入：
        target: [x, y, z]
        hand: "left" or "right"
        num_points: 轨迹点数

    输出：
        joints_traj: list of (5,)
    """

    # 1️⃣ 计算抓取点 & 预抓取点
    grasp_pose, pregrasp_pose = compute_grasp_and_pregrasp(
        target,
        hand=hand
    )

    # 2️⃣ 生成笛卡尔轨迹
    traj = generate_linear_trajectory(
        pregrasp_pose,
        grasp_pose,
        num_points=num_points
    )

    # 3️⃣ 转换为关节轨迹
    joints_traj = pose5_to_joints_trajactory(
        traj,
        hand=hand
    )

    return joints_traj


# =========================================================
# 🔥 测试
# =========================================================
if __name__ == "__main__":
    grasp, pregrasp = compute_grasp_and_pregrasp(
        [0.28, -0.158, 0.04],
        hand="right"
    )

    traj = generate_linear_trajectory(pregrasp, grasp, num_points=10)

    joints_traj = pose5_to_joints_trajactory(traj, hand="right")

    print("\n轨迹 joints：")
    for j in joints_traj:
        print(j)

    joints_traj = target_to_joints_trajectory([0.28, -0.158, 0.04], hand="right", num_points=10)

    print("\n轨迹 joints：")
    for j in joints_traj:
        print(j)