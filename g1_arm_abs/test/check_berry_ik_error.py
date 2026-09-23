#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug tool: measure the berry solve-IK reach error.

Replicates the node's berry IK exactly (berry rx convention + per-arm pose
compensation + grasp_berry MLP), then uses pinocchio forward kinematics on the
G1 URDF to report where the predicted joints actually place the hand tip, and
(optionally) drives the real arm to those joints to measure the on-robot reach.

Pipeline (per the node's _compute_grasp_and_pregrasp_berry -> _solve_ai_ik_berry):

    target (berry xyz, base_link == torso_link)
      + berry grasp offset (~pick_berry_grasp_{x,y,z}_offset, default +0.025 x)
      -> grasp_pose = [gx, gy, gz, rx, rz]
            rz = atan2(gy +/- 0.17, gx)   (right +0.17, left -0.17)
            rx = +pi/2 right / -pi/2 left  (berry convention)
      X  = compensate_pose(grasp_pose_xyz)  (per-arm offsets + tilt)  <- MLP input
      q  = grasp_berry MLP(X)               (5 joints, [sp, sr, sy, el, wr])
      tip_fk = FK(q) . tip_offset_local in torso_link

Errors reported:
    model inversion   = tip_fk - X        (how well the MLP inverts FK; ~0 = good)
    predicted vs berry= tip_fk - target   (where the hand is commanded vs the berry)
  with --execute (robot + g1_arm_simple running):
    tracking          = tip_actual - tip_fk     (controller error)
    TOTAL reach error = tip_actual - target      (hand tip vs the real berry)

Offline (no --execute) needs only pinocchio + torch + the weights. --execute
additionally needs roscore + the running g1_arm_simple node (it calls
/g1_arm_simple/move_arm_joints, which PHYSICALLY MOVES THE ARM).

Examples:
    python check_berry_ik_error.py 0.336 -0.179 -0.050
    python check_berry_ik_error.py 0.336 -0.179 -0.050 --arm right --execute
    python check_berry_ik_error.py 0.30 0.12 -0.05 --no-grasp-offset
"""

import argparse
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)                 # .../g1_arm_abs
_REPO_ROOT = os.path.dirname(_PKG_ROOT)                # .../g1_intellect
sys.path.insert(0, os.path.join(_PKG_ROOT, "src"))

from g1_arm_abs.nn_backend import load_ai_tools, make_device, solve_ai_ik
from g1_arm_abs.pose_compensation import PoseCompensationConfig, compensate_pose

# Arm joints in the MLP's target_cols order (== MoveArmJoints joint_angles order).
ARM_JOINTS = {
    "right": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    ],
    "left": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    ],
}
EE_LINK = {"right": "right_wrist_roll_rubber_hand",
           "left": "left_wrist_roll_rubber_hand"}

# Berry pose-compensation defaults (g1_arm_abs/src/g1_arm_abs/params.py:
# ~pick_berry_{r,l}_{x,y,z}_offset / ~pick_berry_tilt_deg). Default identity --
# the clean berry pipeline does NOT reuse the bottle ~r_/~l_ offsets. Override
# with --comp r_x r_y r_z l_x l_y l_z to explore a residual bias.
DEF_COMP = dict(tilt_deg=0.0,
                r_x=0.0, r_y=0.0, r_z=0.0,
                l_x=0.0, l_y=0.0, l_z=0.0)


class TipFK:
    """pinocchio FK of an arm tip (EE link + local offset) in torso_link."""

    def __init__(self, urdf_path, tip_offset):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.tip_offset = np.asarray(tip_offset, dtype=float)
        self.torso_fid = self.model.getFrameId("torso_link")

    def tip_in_torso(self, arm, joints5):
        pin = self.pin
        q = pin.neutral(self.model)
        for name, val in zip(ARM_JOINTS[arm], joints5):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = float(val)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMt = self.data.oMf[self.torso_fid]
        oMe = self.data.oMf[self.model.getFrameId(EE_LINK[arm])]
        tMe = oMt.inverse() * oMe
        return tMe.translation + tMe.rotation @ self.tip_offset


def _fmt(v):
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def _report(name, a, b):
    d = np.asarray(a) - np.asarray(b)
    print(f"  {name:22s} d={_fmt(d)}  |d|={np.linalg.norm(d)*100:6.2f} cm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=float, help="berry target x (base_link, m)")
    ap.add_argument("y", type=float, help="berry target y (m)")
    ap.add_argument("z", type=float, help="berry target z (m)")
    ap.add_argument("--arm", choices=["auto", "left", "right"], default="auto",
                    help="auto: y>=arm_y_split -> left, else right")
    ap.add_argument("--arm-y-split", type=float, default=0.0)
    ap.add_argument("--urdf", default=os.path.join(
        _REPO_ROOT, "g1_slam", "assets", "g1_23dof_mode_10.urdf"))
    ap.add_argument("--tip-offset", type=float, nargs=3,
                    default=[0.215, 0.06, 0.025],
                    help="tip offset in EE-link frame (m); berry default")
    ap.add_argument("--model-root", default=os.path.join(
        _PKG_ROOT, "models", "grasp_berry"))
    ap.add_argument("--grasp-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="extra xyz ADDED to target before IK (what-if; node adds none)")
    ap.add_argument("--no-grasp-offset", action="store_true")
    ap.add_argument("--rx-deg", type=float, default=None,
                    help="override rx (deg); default berry right=+90 left=-90")
    ap.add_argument("--pregrasp-dz", type=float, default=0.05,
                    help="berry pregrasp = straight below the berry by this dz (m)")
    ap.add_argument("--no-compensation", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--execute", action="store_true",
                    help="drive the real arm via /g1_arm_simple/move_arm_joints")
    ap.add_argument("--service", default="/g1_arm_simple/move_arm_joints")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    arm = args.arm
    if arm == "auto":
        arm = "left" if args.y > args.arm_y_split else "right"

    target = np.array([args.x, args.y, args.z], dtype=float)
    g_off = np.zeros(3) if args.no_grasp_offset else np.asarray(args.grasp_offset)
    gtgt = target + g_off

    rz = (math.atan2(gtgt[1] + 0.17, gtgt[0]) if arm == "right"
          else math.atan2(gtgt[1] - 0.17, gtgt[0]))
    if args.rx_deg is not None:
        rx = math.radians(args.rx_deg)
    else:
        rx = math.pi / 2.0 if arm == "right" else -math.pi / 2.0

    cfg = PoseCompensationConfig(
        use_compensation=not args.no_compensation, tilt_deg=DEF_COMP["tilt_deg"],
        r_x_offset=DEF_COMP["r_x"], r_y_offset=DEF_COMP["r_y"], r_z_offset=DEF_COMP["r_z"],
        l_x_offset=DEF_COMP["l_x"], l_y_offset=DEF_COMP["l_y"], l_z_offset=DEF_COMP["l_z"],
    )

    device = make_device(use_cuda=not args.cpu)
    model, scaler = load_ai_tools(
        os.path.join(args.model_root, arm, "best_model.pth"),
        os.path.join(args.model_root, arm, "x_scaler.pkl"),
        device,
    )
    models = {arm: model}
    scalers = {arm: scaler}

    # GRASP pose: the berry itself, rz = atan2(y +/- 0.17, x), rx = +/-90.
    grasp_pose = (gtgt[0], gtgt[1], gtgt[2], rx, rz)
    q_grasp = solve_ai_ik(grasp_pose, arm, models, scalers, cfg, device)

    # PREGRASP pose: straight below the berry by pregrasp_dz, SAME rx/rz as the
    # grasp -- mirrors _compute_grasp_and_pregrasp_berry (pregrasp shares the
    # grasp orientation; it does NOT use the bottle's fanned pre_rz_base).
    pregrasp_pose = (gtgt[0], gtgt[1], gtgt[2] - args.pregrasp_dz, rx, rz)
    q_pre = solve_ai_ik(pregrasp_pose, arm, models, scalers, cfg, device)

    fk = TipFK(args.urdf, args.tip_offset)
    tip_grasp = fk.tip_in_torso(arm, q_grasp)
    tip_pre = fk.tip_in_torso(arm, q_pre)

    def mlp_input(pose5):
        """The exact 5-D vector the MLP sees: xyz after pose compensation,
        plus rx, rz unchanged (matches nn_backend.solve_ai_ik)."""
        x, y, z, prx, prz = pose5
        if cfg.use_compensation:
            x, y, z = compensate_pose(x, y, z, arm, cfg)
        return np.array([x, y, z, prx, prz], dtype=float)

    def _show(tag, pose5, joints, tip):
        inp = mlp_input(pose5)
        print(f"\n=== {tag} ===")
        print(f"MLP input  xyz       {_fmt(inp[:3])}")
        print(f"           rx,rz     {inp[3]:+.4f} {inp[4]:+.4f} rad "
              f"({math.degrees(inp[3]):+.1f} {math.degrees(inp[4]):+.1f} deg)")
        print(f"-> joints (deg)      {_fmt(np.degrees(joints))}")
        print(f"   FK tip            {_fmt(tip)}")

    print(f"\narm={arm}  tip_offset={_fmt(args.tip_offset)}  "
          f"compensation={'on' if cfg.use_compensation else 'off'}")
    print(f"berry target          {_fmt(target)}")
    if np.any(g_off):
        print(f"+ grasp offset        {_fmt(gtgt)}  (g_off={_fmt(g_off)})")
    _show("GRASP", grasp_pose, q_grasp, tip_grasp)
    _show(f"PREGRASP (straight below, dz={args.pregrasp_dz:.3f})",
          pregrasp_pose, q_pre, tip_pre)
    print("\nerrors (grasp):")
    _report("model inversion", tip_grasp, mlp_input(grasp_pose))
    _report("predicted vs berry", tip_grasp, target)

    if not args.execute:
        print("\n(offline) re-run with --execute to drive the arm and measure the "
              "on-robot reach error.\n")
        return

    import rospy
    from g1_arm_abs.srv import MoveArmJoints
    rospy.init_node("check_berry_ik_error", anonymous=True, disable_signals=True)
    print(f"\nwaiting for {args.service} ...")
    rospy.wait_for_service(args.service, timeout=args.timeout)
    call = rospy.ServiceProxy(args.service, MoveArmJoints)
    print(f"commanding {arm} arm to GRASP joints (PHYSICAL MOTION) ...")
    resp = call(arm=arm, joint_angles=list(map(float, q_grasp)),
                wait=True, timeout=args.timeout)
    if not resp.success:
        print(f"move_arm_joints failed: {resp.message}")
        return
    q_act = np.asarray(resp.current_joint_angles, dtype=float)
    tip_act = fk.tip_in_torso(arm, q_act)
    print(f"q_actual (deg)      {_fmt(np.degrees(q_act))}")
    print(f"FK(q_actual) tip    {_fmt(tip_act)}")
    print("errors:")
    _report("joint tracking(deg)", np.degrees(q_act), np.degrees(q_grasp))
    _report("tracking (tip)", tip_act, tip_grasp)
    _report("TOTAL reach error", tip_act, target)
    print()


if __name__ == "__main__":
    main()
