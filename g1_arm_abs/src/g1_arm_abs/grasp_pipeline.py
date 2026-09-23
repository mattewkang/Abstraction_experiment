# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/grasp_pipeline.py

The grasp planning + execution orchestrators, extracted from
``arm_node_abs.py``.

``plan_grasp`` is the shared planning engine (NN IK -> A* pre-path -> Cartesian
pregrasp->grasp -> lift -> retract -> home joint targets) and runs no motion.
``execute_grasp`` runs the full pipeline: plan, move to pregrasp, pre_grasp
hand, move to grasp, grasp_5f hand, lift.

These call back into many node capabilities and read a lot of config, so the
dependencies are bundled into :class:`GraspContext` (built by the node). The
module is ROS-light: it uses ``rospy`` only for the pre-execution service-wait
probes and the lift-call exception types; the lift MoveArmJoints call itself is
injected as ``ctx.lift_move`` so the generated srv types stay in the node.
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import rospy

from g1_arm_abs.utils import merge_close_points_rad, rad_list_to_deg
from g1_arm_abs.path_ops import flatten_path
from g1_arm_abs.grasp_plan_math import (
    infer_grasp_joints,
    make_lift_pose,
    make_retract_pose,
)

__all__ = [
    "GraspContext",
    "plan_grasp",
    "execute_grasp",
]


@dataclass
class GraspContext:
    """Dependencies the grasp planning + execution pipeline needs."""

    logger: object
    # config
    clip_inferred_joints_to_limits: bool
    use_astar_for_grasp: bool
    astar_merge_joint_thresh_rad: float
    lift_after_grasp: bool
    lift_dz: float
    retract_after_grasp: bool
    retract_x_target: float
    post_grasp_home_roll_offset_deg: dict
    hand_service_timeout: float
    move_arm_service_name: str
    enable_pregrasp_breakpoint: bool
    pregrasp_breakpoint_seconds: float
    hand_settle_time: float
    grasp_5f_retries: int
    grasp_retry_interval: float
    post_grasp_hold_time: float
    default_timeout: float
    # callables (bound node methods)
    check_target_workspace: Callable
    compute_grasp_and_pregrasp: Callable
    solve_ai_ik_robust: Callable
    check_joint_limits: Callable
    clip_joints_to_limits: Callable
    build_horizontal_pre_to_target_waypoints: Callable
    get_start_joint_for_astar: Callable
    plan_astar_segments: Callable
    get_home_joints_rad: Callable
    execute_waypoint_path: Callable
    execute_joint_target: Callable
    get_current_arm_q: Callable
    call_trigger_service: Callable
    call_trigger_service_with_retry: Callable
    pre_grasp_service_for: Callable
    pre_grasp_client_for: Callable
    grasp_5f_service_for: Callable
    grasp_5f_client_for: Callable
    lift_move: Callable
    # When true, verify the arm actually reached the grasp pose (max per-joint
    # error vs grasp_joints <= grasp_reach_tol_rad) BEFORE triggering the grasp
    # hand; abort without closing the hand if it did not. Default off so the
    # bottle path is unchanged; the berry path turns it on.
    require_grasp_reach: bool = False
    grasp_reach_tol_rad: float = 0.1
    # Settle pause (s) AFTER the arm reaches the grasp pose and BEFORE the grasp
    # hand is triggered, so the arm fully stops first. Default 0 (bottle); berry
    # sets it.
    grasp_hand_settle_time: float = 0.0


def plan_grasp(
    ctx: GraspContext,
    arm,
    target,
    force_astar=None,
    pre_to_target_waypoint_count=None,
    log_tag="plan",
) -> dict:
    """Compute the full grasp plan (NN IK, A* pre-path, Cartesian target path,
    optional lift / retract / home joints) WITHOUT executing any motion.

    Returns a dict with either ``success=False`` + error message, or a complete
    plan (pregrasp/grasp poses + joints, pre/target/full paths, lift, retract,
    home, use_astar, clip_messages, astar_plan).
    """
    plan = {
        "success": False,
        "message": "",
        "pregrasp_pose": None,
        "grasp_pose": None,
        "pregrasp_joints": None,
        "grasp_joints": None,
        "pre_path_rad": [],
        "target_path_rad": [],
        "full_path_rad": [],
        "lift_joints": None,
        "retract_joints": None,
        "home_joints": None,
        "use_astar": False,
        "clip_messages": [],
        "astar_plan": None,
    }

    in_workspace, workspace_msg = ctx.check_target_workspace(target)
    if not in_workspace:
        ctx.logger.warning(f"[{log_tag}] arm={arm} | {workspace_msg}")
        plan["message"] = workspace_msg
        return plan

    # Shared NN-IK core (compute poses, solve both IK, clip + recheck) — the
    # same mechanic the infer_grasp_from_point handler uses.
    res = infer_grasp_joints(
        arm, target,
        compute_grasp_and_pregrasp=ctx.compute_grasp_and_pregrasp,
        solve_ik=ctx.solve_ai_ik_robust,
        check_limits=ctx.check_joint_limits,
        clip_limits=ctx.clip_joints_to_limits,
        clip_enabled=ctx.clip_inferred_joints_to_limits,
    )
    pregrasp_pose = res.pregrasp_pose
    grasp_pose = res.grasp_pose
    pregrasp_joints = res.pregrasp_joints
    grasp_joints = res.grasp_joints
    clip_messages = res.clip_messages

    # Debug: print Cartesian pregrasp/grasp pose and the corresponding (pre-clip)
    # IK joints (deg) so the operator can compare against:
    #   - the viewer's pregrasp_cell snapshot
    #   - the planner's final waypoint logged by the voxel adapter
    #   - the actual joint state logged after pregrasp execution.
    # Discrepancy between any of these explains sim-vs-real mismatches.
    ctx.logger.info(
        f"[{log_tag}] arm={arm} | "
        f"pregrasp_pose (x,y,z,rx,rz) = "
        f"{[round(float(v), 4) for v in pregrasp_pose]} | "
        f"IK pregrasp joints (deg) = "
        f"{[round(v, 2) for v in rad_list_to_deg(res.pregrasp_joints_raw)]}"
    )
    ctx.logger.info(
        f"[{log_tag}] arm={arm} | "
        f"grasp_pose    (x,y,z,rx,rz) = "
        f"{[round(float(v), 4) for v in grasp_pose]} | "
        f"IK grasp    joints (deg) = "
        f"{[round(v, 2) for v in rad_list_to_deg(res.grasp_joints_raw)]}"
    )

    plan["pregrasp_pose"] = pregrasp_pose
    plan["grasp_pose"] = grasp_pose
    plan["pregrasp_joints"] = pregrasp_joints
    plan["grasp_joints"] = grasp_joints
    plan["clip_messages"] = clip_messages

    if res.clip_errors:
        plan["message"] = "failed to clip inferred joints | " + " | ".join(
            res.clip_errors
        )
        return plan
    if clip_messages:
        ctx.logger.warning(f"[{log_tag}] " + " | ".join(clip_messages))

    if not res.pre_ok or not res.grasp_ok:
        details = []
        if not res.pre_ok:
            details.append("pregrasp: " + "; ".join(res.pre_violations))
        if not res.grasp_ok:
            details.append("grasp: " + "; ".join(res.grasp_violations))
        plan["message"] = "inferred joints out-of-limit | " + " | ".join(details)
        return plan

    ctx.logger.info(f"[{log_tag}] arm={arm}")
    ctx.logger.info(f"[{log_tag}] target={np.asarray(target, dtype=float).tolist()}")
    ctx.logger.info(f"[{log_tag}] pregrasp_pose={pregrasp_pose.tolist()}")
    ctx.logger.info(f"[{log_tag}] grasp_pose={grasp_pose.tolist()}")
    ctx.logger.info(f"[{log_tag}] pregrasp_joints={pregrasp_joints.tolist()}")
    ctx.logger.info(f"[{log_tag}] grasp_joints={grasp_joints.tolist()}")

    # Cartesian horizontal interpolation from pregrasp to grasp (always a list,
    # last entry is grasp_joints).
    target_path_rad = ctx.build_horizontal_pre_to_target_waypoints(
        arm=arm,
        pregrasp_pose=pregrasp_pose,
        grasp_pose=grasp_pose,
        grasp_joints=grasp_joints,
        waypoint_count=pre_to_target_waypoint_count,
    )
    ctx.logger.info(
        f"[{log_tag}] pre->target waypoint count={len(target_path_rad)} "
        f"(intermediate={max(0, len(target_path_rad) - 1)})"
    )

    use_astar = ctx.use_astar_for_grasp if force_astar is None else bool(force_astar)
    plan["use_astar"] = use_astar
    astar_plan = None
    if use_astar:
        try:
            q_start = ctx.get_start_joint_for_astar(arm)
            astar_plan = ctx.plan_astar_segments(
                arm=arm,
                q_start_rad=q_start,
                q_pregrasp_rad=pregrasp_joints,
                q_target_rad=grasp_joints,
            )
            astar_plan["target_path_rad"] = target_path_rad
            astar_plan["full_path_rad"] = merge_close_points_rad(
                astar_plan["pre_path_rad"] + astar_plan["target_path_rad"],
                joint_thresh_rad=ctx.astar_merge_joint_thresh_rad,
            )
            ctx.logger.info(
                f"[{log_tag}] A* waypoints: full={len(astar_plan['full_path_rad'])}, "
                f"pre={len(astar_plan['pre_path_rad'])}, target={len(astar_plan['target_path_rad'])}"
            )
        except Exception as e:
            plan["message"] = f"A* planning failed: {e}"
            return plan

    # Expose pre/target/full paths uniformly. When A* is off, pre_path is a
    # single waypoint equal to the pregrasp joints (single-target motion).
    if astar_plan is not None:
        plan["pre_path_rad"] = [list(q) for q in astar_plan["pre_path_rad"]]
        plan["full_path_rad"] = [list(q) for q in astar_plan["full_path_rad"]]
    else:
        plan["pre_path_rad"] = [pregrasp_joints.tolist()]
        plan["full_path_rad"] = merge_close_points_rad(
            plan["pre_path_rad"] + list(target_path_rad),
            joint_thresh_rad=ctx.astar_merge_joint_thresh_rad,
        )
        plan["full_path_rad"] = [list(q) for q in plan["full_path_rad"]]
    plan["target_path_rad"] = [list(q) for q in target_path_rad]
    plan["astar_plan"] = astar_plan

    # Lift plan: PURE-VERTICAL lift from the grasp pose.
    if ctx.lift_after_grasp:
        lift_pose = make_lift_pose(grasp_pose, ctx.lift_dz)
        lift_joints = ctx.solve_ai_ik_robust(lift_pose, hand=arm)
        lift_ok, lift_violations = ctx.check_joint_limits(arm, lift_joints)
        if not lift_ok and ctx.clip_inferred_joints_to_limits:
            lift_joints, lift_adjustments, lift_clip_errors = ctx.clip_joints_to_limits(
                arm, lift_joints
            )
            if lift_clip_errors:
                plan["message"] = "failed to clip lift joints | " + " | ".join(lift_clip_errors)
                return plan
            if lift_adjustments:
                ctx.logger.warning(
                    f"[{log_tag}] lift clipped: " + "; ".join(lift_adjustments)
                )
            lift_ok, lift_violations = ctx.check_joint_limits(arm, lift_joints)
        if not lift_ok:
            plan["message"] = "lift joints out-of-limit | " + "; ".join(lift_violations)
            return plan
        plan["lift_joints"] = lift_joints
        ctx.logger.info(
            f"[{log_tag}] lift_pose={lift_pose.tolist()} (dz={ctx.lift_dz:+.4f})"
        )
        ctx.logger.info(f"[{log_tag}] lift_joints={lift_joints.tolist()}")

    # Retract plan: keep yaw + roll, drop z back to the original target point's
    # z, slide in the opposite-yaw direction until x reaches retract_x_target.
    # Skipped (retract_joints stays None) when retract is off, lift was
    # disabled/unsolvable, |cos(rz)| < 1e-3, or lift_x is already past target.
    if ctx.retract_after_grasp and plan["lift_joints"] is not None:
        target_z = float(target[2])
        retract_pose, retract_meta = make_retract_pose(
            grasp_pose, lift_pose, target_z, ctx.retract_x_target
        )

        if retract_meta["reason"] == "perpendicular":
            ctx.logger.warning(
                f"[{log_tag}] retract skipped: |cos(rz)|<1e-3 "
                f"(rz={retract_meta['rz']:+.4f} rad); retract direction is perpendicular "
                f"to x-axis"
            )
        elif retract_meta["reason"] == "past_target":
            ctx.logger.warning(
                f"[{log_tag}] retract skipped: lift_x={retract_meta['lift_x']:.3f} m "
                f"already past target x={ctx.retract_x_target:.3f} m "
                f"along yaw (s={retract_meta['s']:+.4f} m, cos(rz)={retract_meta['cos_rz']:+.4f})"
            )
        else:
            s = retract_meta["s"]
            retract_joints = ctx.solve_ai_ik_robust(retract_pose, hand=arm)
            r_ok, r_violations = ctx.check_joint_limits(arm, retract_joints)
            if not r_ok and ctx.clip_inferred_joints_to_limits:
                retract_joints, r_adjustments, r_clip_errors = (
                    ctx.clip_joints_to_limits(arm, retract_joints)
                )
                if r_clip_errors:
                    plan["message"] = (
                        "failed to clip retract joints | "
                        + " | ".join(r_clip_errors)
                    )
                    return plan
                if r_adjustments:
                    ctx.logger.warning(
                        f"[{log_tag}] retract clipped: "
                        + "; ".join(r_adjustments)
                    )
                r_ok, r_violations = ctx.check_joint_limits(arm, retract_joints)
            if not r_ok:
                plan["message"] = (
                    "retract joints out-of-limit | " + "; ".join(r_violations)
                )
                return plan
            plan["retract_joints"] = retract_joints
            ctx.logger.info(
                f"[{log_tag}] retract_pose={retract_pose.tolist()} "
                f"(s={s:.4f} m, x_target={ctx.retract_x_target:.3f} m)"
            )
            ctx.logger.info(
                f"[{log_tag}] retract_joints={retract_joints.tolist()}"
            )

    # Home joints (from ~{arm}_home_joints_deg). No IK -- a fixed joint config.
    # Per-arm shoulder_roll offset is applied so the post-grasp home leg differs
    # from the cold-start /GoHome pose (arms open slightly sideways).
    try:
        home_rad = ctx.get_home_joints_rad(arm)
        offset_deg = ctx.post_grasp_home_roll_offset_deg.get(arm, 0.0)
        if abs(offset_deg) > 1e-9:
            home_rad[1] += math.radians(offset_deg)
            ctx.logger.info(
                f"[{log_tag}] home_joints[shoulder_roll] += "
                f"{offset_deg:+.2f} deg -> "
                f"{math.degrees(home_rad[1]):+.2f} deg (post-grasp open)"
            )
        plan["home_joints"] = home_rad
    except Exception as exc:
        ctx.logger.warning(f"[{log_tag}] home_joints unavailable: {exc}")

    plan["success"] = True
    plan["message"] = (
        "plan success"
        if not clip_messages
        else "plan success (with joint clipping)"
    )
    return plan


def execute_grasp(
    ctx: GraspContext,
    arm,
    target,
    wait=True,
    timeout=None,
    force_astar=None,
) -> dict:
    """Execute the full grasp pipeline: plan, move to pregrasp, pre_grasp hand,
    move to grasp, grasp_5f hand, lift. Returns a result dict.
    """
    result = {
        "success": False,
        "message": "",
        "pregrasp_pose": [],
        "grasp_pose": [],
        "pregrasp_joint_angles": [],
        "grasp_joint_angles": [],
        "path_joint_angles_flat": [],
        "waypoint_count": 0,
        "final_joint_angles": [],
    }

    plan = plan_grasp(
        ctx, arm=arm, target=target, force_astar=force_astar, log_tag="execute",
    )
    # Populate debug fields in the response regardless of plan success.
    if plan["pregrasp_pose"] is not None:
        result["pregrasp_pose"] = plan["pregrasp_pose"].tolist()
    if plan["grasp_pose"] is not None:
        result["grasp_pose"] = plan["grasp_pose"].tolist()
    if plan["pregrasp_joints"] is not None:
        result["pregrasp_joint_angles"] = plan["pregrasp_joints"].tolist()
    if plan["grasp_joints"] is not None:
        result["grasp_joint_angles"] = plan["grasp_joints"].tolist()
    if plan["use_astar"] and plan["full_path_rad"]:
        result["path_joint_angles_flat"] = flatten_path(plan["full_path_rad"])
        result["waypoint_count"] = len(plan["full_path_rad"])

    if not plan["success"]:
        result["message"] = plan["message"]
        return result

    pregrasp_joints = plan["pregrasp_joints"]
    grasp_joints = plan["grasp_joints"]
    pregrasp_pose = plan["pregrasp_pose"]
    target_path_rad = plan["target_path_rad"]
    astar_plan = plan["astar_plan"]
    clip_messages = plan["clip_messages"]

    lift_plan = None
    if plan["lift_joints"] is not None:
        lift_plan = {"joints": plan["lift_joints"]}

    # Pre-check required services before starting any motion sequence.
    required_services = [
        ctx.pre_grasp_service_for(arm),
        ctx.grasp_5f_service_for(arm),
    ]
    if ctx.lift_after_grasp:
        required_services.append(ctx.move_arm_service_name)
    for srv_name in required_services:
        try:
            rospy.wait_for_service(srv_name, timeout=ctx.hand_service_timeout)
        except rospy.ROSException as e:
            result["message"] = f"required service {srv_name} not available before execution: {e}"
            return result

    # Step 1: move to pregrasp (or pregrasp path)
    if astar_plan is not None:
        success, q_current, msg = ctx.execute_waypoint_path(
            arm=arm,
            waypoints_rad=astar_plan["pre_path_rad"],
            wait=wait,
            timeout=timeout,
        )
    else:
        success, q_current, msg = ctx.execute_joint_target(
            arm=arm,
            q_target=pregrasp_joints,
            wait=wait,
            timeout=timeout,
        )
    if not success:
        result["message"] = f"failed at pregrasp motion: {msg}"
        result["final_joint_angles"] = q_current.tolist()
        return result

    # Debug: log the actual joint state after pregrasp motion completed.
    actual_q = ctx.get_current_arm_q(arm)
    actual_q_list = (
        [round(v, 2) for v in rad_list_to_deg(actual_q)]
        if actual_q is not None else []
    )
    planned_final_q = (
        [round(v, 2) for v in rad_list_to_deg(pregrasp_joints)]
    )
    ctx.logger.info(
        f"[execute_grasp] arm={arm} pregrasp reached | "
        f"IK pregrasp joints (deg) = {planned_final_q} | "
        f"actual joints (deg) = {actual_q_list}"
    )

    if ctx.enable_pregrasp_breakpoint and ctx.pregrasp_breakpoint_seconds > 0:
        q_now = ctx.get_current_arm_q(arm)
        q_now_list = [] if q_now is None else np.round(q_now, 4).tolist()
        ctx.logger.info(
            f"[breakpoint] pregrasp reached | arm={arm} | "
            f"pause={ctx.pregrasp_breakpoint_seconds:.2f}s | "
            f"pregrasp_pose={np.round(pregrasp_pose, 4).tolist()} | "
            f"joint_now={q_now_list}"
        )
        time.sleep(ctx.pregrasp_breakpoint_seconds)

    # Step 2: call pre_grasp on the hand that matches `arm`.
    hand_ok, hand_msg = ctx.call_trigger_service(
        ctx.pre_grasp_service_for(arm),
        ctx.pre_grasp_client_for(arm),
        wait_available=False,
    )
    if not hand_ok:
        q_now = ctx.get_current_arm_q(arm)
        result["message"] = f"failed at hand pre_grasp: {hand_msg}"
        result["final_joint_angles"] = [] if q_now is None else q_now.tolist()
        return result

    if ctx.hand_settle_time > 0:
        time.sleep(ctx.hand_settle_time)

    # Step 3: move to grasp (or target path)
    if astar_plan is not None:
        success, q_current, msg = ctx.execute_waypoint_path(
            arm=arm,
            waypoints_rad=astar_plan["target_path_rad"],
            wait=wait,
            timeout=timeout,
        )
    else:
        success, q_current, msg = ctx.execute_waypoint_path(
            arm=arm,
            waypoints_rad=target_path_rad,
            wait=wait,
            timeout=timeout,
        )
    if not success:
        result["message"] = f"failed at grasp motion: {msg}"
        result["final_joint_angles"] = q_current.tolist()
        return result

    # Step 3.5: only trigger the grasp hand once the arm has actually REACHED the
    # grasp pose. Verify the current joints are within grasp_reach_tol_rad of the
    # planned grasp_joints; abort (without closing the hand) otherwise.
    if ctx.require_grasp_reach:
        q_now = ctx.get_current_arm_q(arm)
        if q_now is None:
            result["message"] = "grasp reach check: no joint state available"
            return result
        max_err = float(np.max(np.abs(
            np.asarray(q_now, dtype=float).reshape(-1)
            - np.asarray(grasp_joints, dtype=float).reshape(-1))))
        if max_err > ctx.grasp_reach_tol_rad:
            result["final_joint_angles"] = np.asarray(q_now, dtype=float).tolist()
            result["message"] = (
                f"arm did not reach grasp pose (max_err={max_err:.4f} rad > "
                f"tol={ctx.grasp_reach_tol_rad:.4f}); grasp hand NOT triggered")
            return result
        ctx.logger.info(
            f"[execute_grasp] grasp pose reached (max_err={max_err:.4f} rad "
            f"<= {ctx.grasp_reach_tol_rad:.4f}); triggering grasp hand")

    # Settle pause: let the arm fully stop at the grasp pose before closing the
    # hand (guards against the hand firing while the arm is still settling).
    if ctx.grasp_hand_settle_time > 0:
        time.sleep(ctx.grasp_hand_settle_time)

    # Step 4: call grasp_5f on the matching hand (with retry).
    hand_ok, hand_msg = ctx.call_trigger_service_with_retry(
        ctx.grasp_5f_service_for(arm),
        ctx.grasp_5f_client_for(arm),
        retries=ctx.grasp_5f_retries,
        interval=ctx.grasp_retry_interval,
        wait_available=False,
    )
    if not hand_ok:
        q_now = ctx.get_current_arm_q(arm)
        result["message"] = f"failed at hand grasp_5f: {hand_msg}"
        result["final_joint_angles"] = [] if q_now is None else q_now.tolist()
        return result

    if ctx.post_grasp_hold_time > 0:
        time.sleep(ctx.post_grasp_hold_time)

    # Step 5: lift after grasp
    if not ctx.lift_after_grasp:
        q_now = ctx.get_current_arm_q(arm)
        result["final_joint_angles"] = [] if q_now is None else q_now.tolist()
        result["success"] = True
        result["message"] = "execute_grasp success (lift disabled)"
        return result

    if lift_plan is None:
        result["message"] = "internal error: lift plan not prepared"
        return result
    lift_timeout = timeout if (timeout is not None and timeout > 0) else ctx.default_timeout
    try:
        lift_resp = ctx.lift_move(arm, lift_plan["joints"].tolist(), wait, lift_timeout)
    except rospy.ROSException as e:
        result["message"] = f"failed at lift motion: service {ctx.move_arm_service_name} not available: {e}"
        q_now = ctx.get_current_arm_q(arm)
        result["final_joint_angles"] = [] if q_now is None else q_now.tolist()
        return result
    except rospy.ServiceException as e:
        result["message"] = f"failed at lift motion: call {ctx.move_arm_service_name} failed: {e}"
        q_now = ctx.get_current_arm_q(arm)
        result["final_joint_angles"] = [] if q_now is None else q_now.tolist()
        return result

    result["final_joint_angles"] = lift_resp.current_joint_angles
    if not lift_resp.success:
        result["message"] = f"failed at lift motion: {lift_resp.message}"
        return result

    result["success"] = True
    result["message"] = (
        "execute_grasp success"
        if not clip_messages
        else "execute_grasp success (with joint clipping)"
    )
    return result
