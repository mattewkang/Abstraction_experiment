"""
Joint index definitions for Unitree G1 robot (23-DOF variant).

G1_23 has 5 DOF per arm (10 DOF total for both arms):
- ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll

Joint indices follow the Unitree SDK2 convention.
"""

from enum import IntEnum


class G1_23_JointArmIndex(IntEnum):
    """Joint indices for G1_23 arm motors (10 DOF total)."""

    # Left arm (5 joints)
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm (5 joints)
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26


class G1_23_JointIndex(IntEnum):
    """Joint indices for all G1_23 motors."""

    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    # Waist
    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristYawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28

    # Not used (weight control is at index 29)
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


# Total number of motors in G1_23
G1_23_NUM_MOTORS = 35

# DOF per arm
ARM_DOF = 5

# Total arm DOF (both arms)
TOTAL_ARM_DOF = 10

# Joint index for weight parameter in motion mode
# Actual command = motion_control * (1 - weight) + arm_sdk_cmd * weight
WEIGHT_INDEX = G1_23_JointIndex.kNotUsedJoint0

# Left arm joint indices (ordered)
LEFT_ARM_INDICES = [
    G1_23_JointArmIndex.kLeftShoulderPitch,
    G1_23_JointArmIndex.kLeftShoulderRoll,
    G1_23_JointArmIndex.kLeftShoulderYaw,
    G1_23_JointArmIndex.kLeftElbow,
    G1_23_JointArmIndex.kLeftWristRoll,
]

# Right arm joint indices (ordered)
RIGHT_ARM_INDICES = [
    G1_23_JointArmIndex.kRightShoulderPitch,
    G1_23_JointArmIndex.kRightShoulderRoll,
    G1_23_JointArmIndex.kRightShoulderYaw,
    G1_23_JointArmIndex.kRightElbow,
    G1_23_JointArmIndex.kRightWristRoll,
]

# Joint names for ROS interface (matches URDF naming)
JOINT_NAMES = {
    # Left arm
    G1_23_JointArmIndex.kLeftShoulderPitch: 'left_shoulder_pitch_joint',
    G1_23_JointArmIndex.kLeftShoulderRoll: 'left_shoulder_roll_joint',
    G1_23_JointArmIndex.kLeftShoulderYaw: 'left_shoulder_yaw_joint',
    G1_23_JointArmIndex.kLeftElbow: 'left_elbow_joint',
    G1_23_JointArmIndex.kLeftWristRoll: 'left_wrist_roll_joint',
    # Right arm
    G1_23_JointArmIndex.kRightShoulderPitch: 'right_shoulder_pitch_joint',
    G1_23_JointArmIndex.kRightShoulderRoll: 'right_shoulder_roll_joint',
    G1_23_JointArmIndex.kRightShoulderYaw: 'right_shoulder_yaw_joint',
    G1_23_JointArmIndex.kRightElbow: 'right_elbow_joint',
    G1_23_JointArmIndex.kRightWristRoll: 'right_wrist_roll_joint',
}

# Reverse mapping: joint name to index
JOINT_NAME_TO_INDEX = {v: k for k, v in JOINT_NAMES.items()}

# Short names for services (e.g., "elbow" -> full joint name)
SHORT_NAMES = {
    'shoulder_pitch': 'shoulder_pitch_joint',
    'shoulder_roll': 'shoulder_roll_joint',
    'shoulder_yaw': 'shoulder_yaw_joint',
    'elbow': 'elbow_joint',
    'wrist_roll': 'wrist_roll_joint',
}

# Joint limits from URDF (lower, upper) in radians
# Order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
ARM_JOINT_LIMITS = {
    'left': [
        (-3.0892, 2.6704),   # shoulder_pitch
        (-1.5882, 2.2515),   # shoulder_roll
        (-2.618, 2.618),     # shoulder_yaw
        (-1.0472, 2.0944),   # elbow
        (-1.9722, 1.9722),   # wrist_roll
    ],
    'right': [
        (-3.0892, 2.6704),   # shoulder_pitch
        (-2.2515, 1.5882),   # shoulder_roll (note: asymmetric)
        (-2.618, 2.618),     # shoulder_yaw
        (-1.0472, 2.0944),   # elbow
        (-1.9722, 1.9722),   # wrist_roll
    ],
}


def check_joint_limit(arm: str, joint_idx: int, angle: float) -> tuple:
    """
    Check if angle is within joint limits.

    Args:
        arm: 'left' or 'right'
        joint_idx: Joint index within arm (0-4)
        angle: Angle to check in radians

    Returns:
        Tuple of (is_valid, lower_limit, upper_limit)
    """
    arm = arm.lower()
    if arm not in ARM_JOINT_LIMITS:
        raise ValueError(f"Invalid arm: {arm}")
    if joint_idx < 0 or joint_idx >= len(ARM_JOINT_LIMITS[arm]):
        raise ValueError(f"Invalid joint index: {joint_idx}")

    lower, upper = ARM_JOINT_LIMITS[arm][joint_idx]
    is_valid = lower <= angle <= upper
    return is_valid, lower, upper


def get_joint_index(arm: str, joint_name: str) -> int:
    """
    Get joint index from arm name and joint name.

    Args:
        arm: 'left' or 'right'
        joint_name: Joint name (e.g., 'elbow', 'shoulder_pitch')

    Returns:
        Joint index

    Raises:
        ValueError: If arm or joint_name is invalid
    """
    arm = arm.lower()
    if arm not in ('left', 'right'):
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'")

    # Handle short names
    if joint_name in SHORT_NAMES:
        joint_name = SHORT_NAMES[joint_name]

    # Build full joint name
    if not joint_name.startswith(arm):
        full_name = f"{arm}_{joint_name}"
    else:
        full_name = joint_name

    if full_name not in JOINT_NAME_TO_INDEX:
        raise ValueError(f"Invalid joint name: {joint_name}")

    return JOINT_NAME_TO_INDEX[full_name]


def get_arm_indices(arm: str) -> list:
    """
    Get all joint indices for specified arm.

    Args:
        arm: 'left' or 'right'

    Returns:
        List of joint indices
    """
    arm = arm.lower()
    if arm == 'left':
        return list(LEFT_ARM_INDICES)
    elif arm == 'right':
        return list(RIGHT_ARM_INDICES)
    else:
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'")
