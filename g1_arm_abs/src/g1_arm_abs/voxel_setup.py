# -*- coding: utf-8 -*-
"""
g1_arm_abs/src/g1_arm_abs/voxel_setup.py

Factory for the obstacle-aware voxel planner adapter, extracted from
``arm_node_abs.py``.

ROS-light: constructs a :class:`VoxelPlannerAdapter` (which owns the depth /
CameraInfo subscriptions and latched publishers). A one-time bitmap-existence
check surfaces a missing ``decompress_once.py`` step at startup with a clear
message. ``pkg_root`` is passed in by the node because it must be the package
root derived from the SCRIPT's ``__file__`` (not this module's).
"""

import os

from g1_arm_abs.voxel_planner_adapter import VoxelPlannerAdapter

__all__ = ["build_voxel_adapter"]


def build_voxel_adapter(
    pkg_root,
    logger,
    *,
    voxel_output_dir,
    depth_topic,
    camera_info_topic,
    depth_scale,
    min_depth,
    max_depth,
    subsample_stride,
    camera_translation,
    camera_rotation_ypr,
    obstacle_frame_offset,
    defer_joint,
    defer_cost_mult,
    h_weight,
    shortcut,
    publish_obstacles,
    obstacles_topic,
    obstacles_frame,
    publish_first_only,
    inflation_cells,
    live_publish_rate,
    live_publish_arm,
    history_frames,
    history_min_obs,
    depth_median_filter,
    fresh_depth_timeout_sec,
    warmup_on_start,
):
    """Construct the voxel planner adapter (lazy A* load), after verifying the
    decompressed per-arm bitmap files exist. Returns the adapter; raises
    ``RuntimeError`` on missing bitmaps and re-raises any construction error.
    """
    default_output_dir = os.path.join(pkg_root, "models", "abstraction_map")
    output_dir = voxel_output_dir or default_output_dir
    required = [
        os.path.join(output_dir, "voxel_config_bitmap_left.uint8"),
        os.path.join(output_dir, "voxel_config_bitmap_right.uint8"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "voxel A* backend requested but bitmap files are missing "
            f"(e.g. {missing[0]}); run "
            f"'python3 {os.path.join(pkg_root, 'src', 'g1_arm_abs', 'voxel_planner_deploy', 'decompress_once.py')}' "
            "once to enable it."
        )

    try:
        adapter = VoxelPlannerAdapter(
            output_dir=voxel_output_dir or None,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            subsample_stride=subsample_stride,
            camera_translation=camera_translation,
            camera_rotation_ypr=camera_rotation_ypr,
            obstacle_frame_offset=obstacle_frame_offset,
            defer_joint=defer_joint,
            defer_cost_mult=defer_cost_mult,
            h_weight=h_weight,
            shortcut=shortcut,
            publish_obstacles=publish_obstacles,
            obstacles_topic=obstacles_topic,
            obstacles_frame=obstacles_frame,
            publish_first_only=publish_first_only,
            inflation_cells=inflation_cells,
            live_publish_rate=live_publish_rate,
            live_publish_arm=live_publish_arm,
            history_frames=history_frames,
            history_min_obs=history_min_obs,
            depth_median_filter=depth_median_filter,
            fresh_depth_timeout_sec=fresh_depth_timeout_sec,
        )
        logger.info(
            f"[A* backend=voxel] adapter ready: depth='{depth_topic}' "
            f"info='{camera_info_topic}' "
            f"stride={subsample_stride} "
            f"depth_range=[{min_depth}, {max_depth}]"
        )
        if warmup_on_start:
            adapter.warmup(("left", "right"))
    except Exception as e:  # noqa: BLE001
        logger.error(f"voxel A* backend init failed: {e}")
        raise

    return adapter
