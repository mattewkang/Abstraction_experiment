#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
g1_arm_abs/src/g1_arm_abs/voxel_planner_adapter.py

Bridges the voxel-based obstacle-aware A* planner
(src/g1_arm_abs/voxel_planner_deploy/) into arm_node_abs.py without
touching the existing static-reachable-map planner.

What this module provides
-------------------------
VoxelPlannerAdapter
  * Lazy-loads a VoxelObstacleAStar instance per arm ("left"/"right").
  * Subscribes to the robot head camera's aligned depth + camera_info
    topics (same topics g1_camera/object_detector_node.py uses). The
    latest depth frame is back-projected on demand into the obstacle
    frame and voxelized into the planner's grid.
  * Returns joint-space waypoint paths in the same schema produced by
    arm_node_abs._plan_astar_segments, so the caller does not need to
    learn a new return contract.

Why depth image + intrinsics instead of a PointCloud2 topic
-----------------------------------------------------------
g1_camera/object_detector_node.py does NOT publish a PointCloud2 — it
consumes /camera/aligned_depth_to_color/image_raw + /camera/color/camera_info
and deprojects per-detection pixels itself. To stay consistent with that
pattern (and to avoid adding a new dependency on realsense's pointcloud
filter) we reuse the same inputs here: subscribe to the aligned depth
image and the camera_info, deproject every valid pixel (with subsampling)
through the intrinsics K, and transform the resulting cloud into the
planner's obstacle frame using the same static extrinsic defaults
g1_camera uses. See _pixel_to_camera / _camera_to_obstacle_frame below.

Coordinate frames
-----------------
The voxel planner expects obstacles in the **torso_link** frame
(x forward, y left, z up), with the planner's own grid origin / voxel
size baked into voxel_phase2_<arm>.json. We expose two configurable
knobs so the adapter works regardless of the robot's tf tree:

  * camera_translation / camera_rotation_ypr:
        static transform from camera_link to the robot's root link
        (same defaults as g1_camera/object_detector_node.py).
  * obstacle_frame_offset:
        translation applied AFTER the camera→root transform, to shift
        points from the root frame into the planner's frame (torso_link).
        Leave at [0,0,0] if the root frame IS the torso_link; otherwise
        measure the torso_link origin in root-link coordinates and plug
        it in here.

Failure modes
-------------
The VoxelObstacleAStar loader will raise FileNotFoundError if the big
.uint8 bitmaps haven't been produced yet. In that case the operator has
to run `python decompress_once.py` in the voxel_planner_deploy/ directory
once (see voxel_planner_deploy/README.md). This adapter surfaces that error
at construction time rather than hiding it — the caller in arm_node_abs
logs the failure and the node refuses to start in voxel mode, so you
know immediately to run the decompress step.
"""

import math
import os
import sys
import threading
from collections import deque

import numpy as np
import rospy

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header

# Optional 3x3 median pre-filter on raw depth — kills single-pixel speckle
# and the worst color-aligned-resampling edge noise before deprojection.
# scipy is already pulled in transitively by torch+numba so this rarely
# fails, but keep it optional so a stripped install still loads the module.
try:
    from scipy.ndimage import median_filter as _depth_median_filter
    _HAS_DEPTH_MEDIAN = True
except Exception:  # noqa: BLE001
    _depth_median_filter = None
    _HAS_DEPTH_MEDIAN = False


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VOXEL_DEPLOY_DIR = os.path.join(_THIS_DIR, "voxel_planner_deploy")
# Decompressed voxel bitmaps live under the package's models/ tree:
# src/g1_arm_abs/ -> src/ -> g1_arm_abs/ (package root) / models / abstraction_map.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_VOXEL_OUTPUT_DIR = os.path.join(_PACKAGE_ROOT, "models", "abstraction_map")


def _import_voxel_planner():
    """Import VoxelObstacleAStar from the voxel_planner_deploy/ directory.

    The backend scripts use sibling-style imports (``from voxel_phase3_query
    import ...``) that only resolve when voxel_planner_deploy/ is on sys.path.
    We inject it lazily so merely importing this adapter doesn't pay the cost
    when the static backend is selected.
    """
    if _VOXEL_DEPLOY_DIR not in sys.path:
        sys.path.insert(0, _VOXEL_DEPLOY_DIR)
    from voxel_phase4_astar import VoxelObstacleAStar  # noqa: E402
    return VoxelObstacleAStar


# ----------------------------------------------------------------------
# Static transforms — defaults copied from g1_camera/object_detector_node.py
# ----------------------------------------------------------------------

def _euler_ypr_to_matrix(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


# Fixed rotation: camera_color_optical_frame -> camera_link (same as g1_camera).
_R_OPTICAL_TO_LINK = _euler_ypr_to_matrix(-1.5708, 0.0, -1.5708)


def _imgmsg_to_numpy_depth(msg):
    """Minimal cv_bridge-free converter for 16UC1 / 32FC1 depth images.

    We only need depth; pulling in cv_bridge is an avoidable dep here.
    """
    if msg.encoding == "16UC1":
        dtype = np.uint16
        channels = 1
    elif msg.encoding == "32FC1":
        dtype = np.float32
        channels = 1
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=dtype)
    arr = arr.reshape(msg.height, msg.width * channels)
    if channels > 1:
        arr = arr[:, ::channels]
    return arr


class VoxelPlannerAdapter:
    """Thin wrapper coupling a live depth subscription with VoxelObstacleAStar."""

    def __init__(self,
                 output_dir=None,
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic="/camera/color/camera_info",
                 depth_scale=0.001,
                 min_depth=0.15,
                 max_depth=2.5,
                 subsample_stride=4,
                 camera_translation=(0.04765, 0.0, 0.46268),
                 camera_rotation_ypr=(0.0, 0.8378, 0.0),
                 obstacle_frame_offset=(0.0, 0.0, 0.0),
                 defer_joint=2,
                 defer_cost_mult=2,
                 h_weight=1.0,
                 shortcut=True,
                 preload=False,
                 use_numba=True,
                 publish_obstacles=True,
                 obstacles_topic="/g1_arm_simple/voxel_obstacles",
                 obstacles_frame="torso_link",
                 publish_first_only=False,
                 inflation_cells=0,
                 live_publish_rate=0.0,
                 live_publish_arm="right",
                 history_frames=1,
                 history_min_obs=1,
                 depth_median_filter=True,
                 fresh_depth_timeout_sec=1.0):
        self._output_dir = output_dir or _VOXEL_OUTPUT_DIR
        self._preload = bool(preload)
        self._use_numba = bool(use_numba)
        self._VoxelObstacleAStar = _import_voxel_planner()

        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.subsample_stride = max(1, int(subsample_stride))

        yaw, pitch, roll = (float(x) for x in camera_rotation_ypr)
        self._R_root_cam = _euler_ypr_to_matrix(yaw, pitch, roll)
        self._t_root_cam = np.asarray(camera_translation, dtype=np.float64).reshape(3)
        # Combined optical -> root-link.
        self._R_root_opt = self._R_root_cam @ _R_OPTICAL_TO_LINK
        # Camera origin coincides with camera_link origin in the optical frame,
        # mirroring g1_camera's convention.
        self._t_root_opt = self._t_root_cam.copy()
        self._t_root_to_obstacle = np.asarray(
            obstacle_frame_offset, dtype=np.float64,
        ).reshape(3)

        self.defer_joint = int(defer_joint)
        self.defer_cost_mult = float(defer_cost_mult)
        self.h_weight = float(h_weight)
        self.shortcut = bool(shortcut)
        # Inflate obstacle voxels by N cells in each axis to give the arm a
        # collision margin. Voxel size is 2 cm, so N=1 adds ~2 cm clearance,
        # N=2 ~4 cm. Use 6-connected (face-neighbor) expansion to keep the
        # set from exploding; 26-connected would cube the count.
        self.inflation_cells = max(0, int(inflation_cells))

        # Temporal accumulation: keep per-frame voxel index sets for the
        # last N depth frames and apply a K-of-N persistence filter before
        # passing the result to the planner. A voxel becomes a real
        # obstacle only when at least `history_min_obs` of the last
        # `history_frames` frames observe it — frame-to-frame depth noise
        # rarely lands twice in the same 0.01 m cell, so this turns the
        # 2-3 voxel-thick "shell" produced by raw depth jitter into a
        # single surface layer. N=1, K=1 reproduces the legacy "any
        # observation" union behavior. During warm-up (history not full
        # yet) the effective K is clamped to the current count so we
        # don't lose obstacles in the first few frames.
        self._history_size = max(1, int(history_frames))
        self.history_min_obs = max(1, int(history_min_obs))
        # Per-arm rolling buffers of per-frame voxel-index sets. Voxel
        # indices are valid ONLY against their own planner grid (left and
        # right grids have different origins), so a shared deque would
        # cross-contaminate when the live-publish timer is feeding one arm
        # while a plan call is reading the other.
        self._voxel_history = {
            "left":  deque(maxlen=self._history_size),
            "right": deque(maxlen=self._history_size),
        }
        self._history_lock = threading.Lock()
        # Last single-frame deprojected cloud, kept ONLY for the raw
        # diagnostic publisher so RViz shows what the camera fed this
        # frame (un-thickened by history union).
        self._latest_raw_pts = np.empty((0, 3), dtype=np.float64)
        # Apply 3x3 median filter to raw depth before deprojection.
        self._depth_median_filter = bool(depth_median_filter) and _HAS_DEPTH_MEDIAN
        if depth_median_filter and not _HAS_DEPTH_MEDIAN:
            rospy.logwarn(
                "[voxel_adapter] depth_median_filter requested but "
                "scipy.ndimage not importable — falling back to raw depth."
            )

        # Lazy per-arm planner cache — each is ~67 MB RSS, so only
        # materialize the arm(s) actually requested.
        self._planners = {}

        # Depth-frame state.
        self._K = None
        self._depth = None
        self._depth_stamp = None
        self._info_lock = threading.Lock()
        self._depth_lock = threading.Lock()

        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic
        # Per-plan-call wait_for_message timeout (seconds) for fetching one
        # fresh depth frame inside latest_voxel_obstacles(..., fresh=True);
        # falls back to whatever is currently cached on timeout. Must be
        # >= worst-case inter-frame interval on the depth topic. The live
        # timer path keeps using the background _depth_cb cache to avoid
        # blocking the rospy.Timer thread on slow publishers.
        self._fresh_depth_timeout = float(fresh_depth_timeout_sec)
        rospy.Subscriber(camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.Subscriber(depth_topic, Image, self._depth_cb, queue_size=1)

        # RViz-visible obstacle-voxel publisher. Latched so RViz picks up
        # the last cloud even if it's opened after the grasp completes;
        # use Style=Boxes / Size=voxel_size in the PointCloud2 display to
        # see exactly the voxels A* treats as occupied.
        self._obs_frame = str(obstacles_frame)
        self._obs_pub = (
            rospy.Publisher(obstacles_topic, PointCloud2, queue_size=1, latch=True)
            if publish_obstacles else None
        )
        # Raw deprojected cloud (in obstacle frame, BEFORE grid clipping).
        # Useful when _obs_pub is empty because the cloud doesn't intersect
        # the planner grid — RViz can still show what the camera is feeding
        # the adapter so you can diagnose extrinsics / range / aim.
        self._raw_pub = (
            rospy.Publisher(
                obstacles_topic + "_raw", PointCloud2, queue_size=1, latch=True,
            ) if publish_obstacles else None
        )
        # When True, only the first non-empty publish actually hits the wire;
        # subsequent A* plans skip publishing so the RViz view is frozen on
        # the initial obstacle snapshot. Latched queue keeps it visible
        # indefinitely for late RViz subscribers.
        self._publish_first_only = bool(publish_first_only)
        self._obs_published = False

        # Live-refresh timer: regardless of A* planning, re-voxelize the
        # latest depth frame for `live_publish_arm` at `live_publish_rate` Hz
        # and republish. Gives RViz a continuously-updating view of what the
        # planner sees, useful for debugging obstacle avoidance. Set rate=0
        # to disable (publisher then only fires during plans).
        self._live_rate = float(live_publish_rate)
        self._live_arm = str(live_publish_arm).lower().strip()
        if self._live_arm not in ("left", "right"):
            self._live_arm = "right"
        self._live_timer = None
        if (self._obs_pub is not None
                and self._live_rate > 0.0
                and not self._publish_first_only):
            try:
                # Pre-materialize the live-arm planner so the first timer tick
                # doesn't pay the ~1 s planner-load cost inside the callback.
                self._get_planner(self._live_arm)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn(
                    f"[voxel_adapter] live warmup failed for "
                    f"arm='{self._live_arm}': {exc}"
                )
            period = rospy.Duration(1.0 / self._live_rate)
            self._live_timer = rospy.Timer(period, self._live_publish_cb)
            rospy.loginfo(
                f"[voxel_adapter] live refresh enabled: "
                f"{self._live_rate:.2f} Hz arm='{self._live_arm}'"
            )

    def _live_publish_cb(self, _event):
        """Timer tick: warm BOTH arms' rolling histories from the latest
        depth frame, but publish only the configured `_live_arm`'s voxel
        set to RViz. Warming both keeps the K-of-N window populated for
        whichever arm the next plan call uses (auto-arm switching by
        main_process). Publishing only one arm prevents RViz from
        flickering between two per-arm grids on the same topic."""
        other = "left" if self._live_arm == "right" else "right"
        # Update non-published arm's history first (silent), then the
        # publish arm so the visible publish reflects the freshest frame.
        for arm, do_publish in ((other, False), (self._live_arm, True)):
            try:
                self.latest_voxel_obstacles(arm, publish=do_publish)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(
                    5.0,
                    f"[voxel_adapter] live tick failed for arm='{arm}': {exc}",
                )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _info_cb(self, msg):
        with self._info_lock:
            self._K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    def _depth_cb(self, msg):
        try:
            depth = _imgmsg_to_numpy_depth(msg)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0,
                f"[voxel_adapter] depth decode failed: {exc}",
            )
            return
        with self._depth_lock:
            self._depth = depth
            self._depth_stamp = msg.header.stamp

    def _refresh_depth_now(self):
        """Block up to `_fresh_depth_timeout` seconds waiting for one new
        depth frame and overwrite the cache. Returns True on fresh fetch,
        False on timeout / decode failure (cache untouched in that case so
        the caller falls back to whatever the background _depth_cb last
        wrote)."""
        try:
            msg = rospy.wait_for_message(
                self._depth_topic, Image, timeout=self._fresh_depth_timeout,
            )
        except rospy.ROSException:
            rospy.logwarn_throttle(
                5.0,
                f"[voxel_adapter] fresh depth wait timed out on "
                f"'{self._depth_topic}' after {self._fresh_depth_timeout:.2f}s "
                f"— falling back to cached frame.",
            )
            return False
        except rospy.ROSInterruptException:
            return False
        try:
            depth = _imgmsg_to_numpy_depth(msg)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0,
                f"[voxel_adapter] fresh depth decode failed: {exc}",
            )
            return False
        with self._depth_lock:
            self._depth = depth
            self._depth_stamp = msg.header.stamp
        return True

    # ------------------------------------------------------------------
    # Planner-side helpers
    # ------------------------------------------------------------------
    def _get_planner(self, arm):
        planner = self._planners.get(arm)
        if planner is not None:
            return planner
        planner = self._VoxelObstacleAStar(
            arm,
            output_dir=self._output_dir,
            preload=self._preload,
            use_numba=self._use_numba,
        )
        self._planners[arm] = planner
        return planner

    def warmup(self, arms=("left", "right")):
        """Optionally pre-load planners at node startup so the first grasp
        doesn't pay the ~1 s numba JIT cost."""
        for arm in arms:
            try:
                self._get_planner(arm)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn(
                    f"[voxel_adapter] warmup failed for arm='{arm}': {exc}"
                )

    # ------------------------------------------------------------------
    # Point-cloud -> voxel-index conversion
    # ------------------------------------------------------------------
    def _snapshot_depth_and_K(self):
        with self._info_lock:
            K = None if self._K is None else self._K.copy()
        with self._depth_lock:
            depth = None if self._depth is None else self._depth.copy()
        return depth, K

    def _depth_to_obstacle_points(self, depth, K):
        """Return an (M, 3) array of 3D points in the planner's obstacle frame.

        Fully vectorized: subsample the (H, W) grid by stride, deproject the
        survivors through K to the camera optical frame, then apply the
        static optical->root->obstacle transforms.

        Pre-step: optional 3x3 median filter on the raw depth image. This
        suppresses single-pixel speckle (very common at object edges and
        on shiny surfaces) BEFORE subsampling, which would otherwise alias
        a speckle into a permanent obstacle voxel.
        """
        if self._depth_median_filter:
            # Operates on the original 16U/float depth; output dtype matches.
            depth = _depth_median_filter(depth, size=3)
        stride = self.subsample_stride
        h, w = depth.shape
        u = np.arange(0, w, stride, dtype=np.float64)
        v = np.arange(0, h, stride, dtype=np.float64)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        dd = depth[::stride, ::stride].astype(np.float64) * self.depth_scale

        valid = np.isfinite(dd) & (dd >= self.min_depth) & (dd <= self.max_depth)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        uu = uu[valid]
        vv = vv[valid]
        zz = dd[valid]

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        xx = (uu - cx) * zz / fx
        yy = (vv - cy) * zz / fy
        pts_opt = np.stack([xx, yy, zz], axis=-1)            # (M, 3)
        # Transform into the planner's obstacle frame.
        pts_root = pts_opt @ self._R_root_opt.T + self._t_root_opt
        pts_obs = pts_root + self._t_root_to_obstacle
        return pts_obs

    def _points_to_raw_indices(self, points, planner):
        """Clip (M,3) metric points to the planner grid; return UNIQUE (N,3) cell indices.

        No inflation. Used as the per-frame voxel set fed into the K-of-N
        persistence filter. The dedupe is INTRA-frame (many depth pixels
        collapse into the same 0.01 m cell) — keep it so the persistence
        counter doesn't double-count high-density surfaces.
        """
        vr = planner.vr
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)
        idx = np.floor((points - vr.grid_origin) / vr.voxel_size).astype(np.int64)
        lo = np.array([0, 0, 0], dtype=np.int64)
        hi = vr.grid_shape.astype(np.int64) - 1
        in_grid = np.all((idx >= lo) & (idx <= hi), axis=1)
        idx = idx[in_grid]
        if idx.size == 0:
            return idx
        return np.unique(idx, axis=0)

    def _inflate_indices(self, idx, planner):
        """Apply 6-connected `inflation_cells` expansion to (N,3) cell indices.

        Re-clipped to the planner grid and de-duplicated. No-op when
        `inflation_cells == 0`. Run AFTER the K-of-N persistence filter
        so transient noise voxels don't get amplified into thick masks.
        """
        if self.inflation_cells <= 0 or idx.size == 0:
            return idx
        vr = planner.vr
        lo = np.array([0, 0, 0], dtype=np.int64)
        hi = vr.grid_shape.astype(np.int64) - 1
        n = self.inflation_cells
        shifts = [(0, 0, 0)]
        for d in range(3):
            for k in range(1, n + 1):
                off_pos = [0, 0, 0]; off_pos[d] = k;  shifts.append(tuple(off_pos))
                off_neg = [0, 0, 0]; off_neg[d] = -k; shifts.append(tuple(off_neg))
        expanded = np.concatenate(
            [idx + np.asarray(s, dtype=np.int64) for s in shifts],
            axis=0,
        )
        in_grid = np.all((expanded >= lo) & (expanded <= hi), axis=1)
        return np.unique(expanded[in_grid], axis=0)

    def latest_voxel_obstacles(self, arm, publish=True, fresh=False):
        """Build the (N,3) voxel-index obstacle set from the most recent depth frame.

        Returns (voxel_indices, info_str). voxel_indices may be empty if
        no depth frame has arrived yet.

        When publish=False, history is still updated for `arm` but the
        result is NOT republished to /g1_arm_simple/voxel_obstacles. Used
        by the live-publish timer to warm both arms' histories while still
        publishing only the configured `_live_arm` to RViz, so RViz
        doesn't flicker between two per-arm grids on the same topic.

        When fresh=True, block up to `_fresh_depth_timeout` seconds for a
        new depth frame BEFORE snapshotting the cache, so plan-time calls
        get the camera's current view (not whatever the background
        _depth_cb last buffered). On timeout the cache is reused, matching
        the pre-refactor behavior. Defaults to False because the rospy
        Timer path that drives the RViz live publisher must not block on
        a slow publisher.
        """
        if fresh:
            self._refresh_depth_now()
        depth, K = self._snapshot_depth_and_K()
        if depth is None or K is None:
            missing = []
            if depth is None:
                missing.append(f"depth({self._depth_topic})")
            if K is None:
                missing.append(f"camera_info({self._camera_info_topic})")
            rospy.logwarn_throttle(
                5.0,
                "[voxel_adapter] NO OBSTACLES applied to A* — missing "
                + ", ".join(missing)
                + ". Check 'rostopic hz' on those topics.",
            )
            return (
                np.empty((0, 3), dtype=np.int64),
                "no depth/camera_info yet — planning WITHOUT obstacles",
            )
        planner = self._get_planner(arm)
        pts = self._depth_to_obstacle_points(depth, K)

        # Per-frame voxelize (no inflation); push to history; then apply
        # K-of-N persistence — a voxel survives only if at least
        # `effective_K` of the last `n_frames` frames observed it.
        # `effective_K = min(history_min_obs, n_frames)` ramps K from 1
        # at startup up to its target value once history is full so we
        # don't briefly plan with no obstacles after node start.
        this_frame_voxels = self._points_to_raw_indices(pts, planner)
        with self._history_lock:
            history = self._voxel_history.setdefault(
                arm, deque(maxlen=self._history_size)
            )
            # Always append, including empty frames. Skipping empty frames
            # froze the rolling window: stale entries could sit in the
            # deque indefinitely if the camera briefly saw open space.
            history.append(this_frame_voxels)
            n_frames = len(history)
            non_empty = [v for v in history if v.size > 0]
            if non_empty:
                stacked = np.concatenate(non_empty, axis=0)
                uniq, counts = np.unique(stacked, axis=0, return_counts=True)
                effective_K = min(self.history_min_obs, n_frames)
                persistent = uniq[counts >= effective_K]
            else:
                effective_K = min(self.history_min_obs, n_frames)
                persistent = np.empty((0, 3), dtype=np.int64)
            self._latest_raw_pts = pts

        # Publish the raw single-frame deprojected cloud (NOT a history
        # union) so RViz sees the camera's real surface — the K-of-N
        # filter then decides what becomes a hard obstacle.
        if publish:
            self._publish_raw_cloud(self._latest_raw_pts)

        voxels = self._inflate_indices(persistent, planner)

        if voxels.size == 0 and pts.shape[0] > 0:
            vr = planner.vr
            grid_lo = np.asarray(vr.grid_origin, dtype=np.float64)
            grid_hi = grid_lo + np.asarray(vr.grid_shape, dtype=np.float64) \
                * float(vr.voxel_size)
            cloud_lo = pts.min(axis=0)
            cloud_hi = pts.max(axis=0)
            # Only warn when the cloud bbox doesn't overlap the planner grid
            # at all — that's the genuine wrong-extrinsics / wrong-frame
            # case. When the bboxes overlap but no points fall in cells, the
            # camera is simply looking past an empty manipulation workspace
            # (floor + far walls outside the grid), which is normal idle.
            # logwarn_once so even the real config error doesn't spam.
            bboxes_disjoint = bool(
                np.any(cloud_hi < grid_lo) or np.any(cloud_lo > grid_hi)
            )
            if bboxes_disjoint:
                cloud_mean = pts.mean(axis=0)
                shift = 0.5 * (grid_lo + grid_hi) - cloud_mean
                rospy.logwarn_once(
                    f"[voxel_adapter] depth arrived ({pts.shape[0]} "
                    f"points this frame, {this_frame_voxels.shape[0]} unique cells, "
                    f"{persistent.shape[0]} survived K={effective_K}/"
                    f"{self._history_size}) but NO voxels fell inside the planner grid.\n"
                    f"  cloud bbox (obstacle frame): "
                    f"x[{cloud_lo[0]:+.3f}, {cloud_hi[0]:+.3f}] "
                    f"y[{cloud_lo[1]:+.3f}, {cloud_hi[1]:+.3f}] "
                    f"z[{cloud_lo[2]:+.3f}, {cloud_hi[2]:+.3f}]\n"
                    f"  cloud mean : "
                    f"({cloud_mean[0]:+.3f}, {cloud_mean[1]:+.3f}, {cloud_mean[2]:+.3f})\n"
                    f"  grid bbox  : "
                    f"x[{grid_lo[0]:+.3f}, {grid_hi[0]:+.3f}] "
                    f"y[{grid_lo[1]:+.3f}, {grid_hi[1]:+.3f}] "
                    f"z[{grid_lo[2]:+.3f}, {grid_hi[2]:+.3f}] "
                    f"(voxel_size={vr.voxel_size:.3f})\n"
                    f"  to overlap mean -> grid center, set "
                    f"~voxel_obstacle_frame_offset to "
                    f"[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}]\n"
                    f"  (or fix camera extrinsics/min-max depth so the cloud "
                    f"already lands in torso_link).",
                )
        if publish:
            self._publish_voxel_obstacles(voxels, planner)
        return voxels, (
            f"arm={arm} cloud_points={pts.shape[0]} "
            f"this_frame_cells={this_frame_voxels.shape[0]} "
            f"persistent={persistent.shape[0]} inflated={voxels.shape[0]} "
            f"stride={self.subsample_stride} "
            f"frames={n_frames}/{self._history_size} "
            f"K={effective_K}/{self.history_min_obs}"
        )

    def _publish_raw_cloud(self, points):
        """Publish the deprojected cloud (obstacle frame) for RViz inspection.

        Independent of the grid-clipped voxel publisher so it stays visible
        even when no points fall inside the planner grid (the diagnostic
        case). Empty clouds are skipped — keep the last good frame on screen.
        """
        if self._raw_pub is None or points.shape[0] == 0:
            return
        header = Header(stamp=rospy.Time.now(), frame_id=self._obs_frame)
        msg = pc2.create_cloud_xyz32(
            header, points.astype(np.float32).tolist(),
        )
        try:
            self._raw_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0, f"[voxel_adapter] raw cloud publish failed: {exc}"
            )

    def _publish_voxel_obstacles(self, voxels, planner):
        """Publish voxel centers as a PointCloud2 for RViz inspection.

        Skipped when the publisher is disabled. Empty clouds are also
        skipped so RViz keeps the last non-empty view — otherwise a
        single bad frame would blank the display.
        """
        if self._obs_pub is None or voxels.size == 0:
            return
        if self._publish_first_only and self._obs_published:
            return
        vr = planner.vr
        centers = (voxels.astype(np.float32) + np.float32(0.5)) * \
                  np.float32(vr.voxel_size) + \
                  vr.grid_origin.astype(np.float32)
        header = Header(stamp=rospy.Time.now(), frame_id=self._obs_frame)
        msg = pc2.create_cloud_xyz32(header, centers.tolist())
        try:
            self._obs_pub.publish(msg)
            self._obs_published = True
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0, f"[voxel_adapter] obstacle cloud publish failed: {exc}"
            )

    # ------------------------------------------------------------------
    # Two-stage wrist_roll helpers
    # ------------------------------------------------------------------
    # Plan A* with wrist_roll forced to 0 so the other four joints reach
    # the pregrasp pose with the hand un-rotated, then append a trailing
    # waypoint that rolls the wrist to its IK target value in place. This
    # keeps the goal voxel inside the (narrow) wrist_roll grid even when
    # the IK output is out of bounds, and isolates the wrist rotation as
    # a single standalone motion at the end of the path.
    @staticmethod
    def _split_wrist_roll(q_goal_deg):
        plan_goal = [float(v) for v in q_goal_deg]
        wrist_target = plan_goal[4]
        if abs(wrist_target) <= 1e-9:
            return plan_goal, None
        plan_goal[4] = 0.0
        return plan_goal, wrist_target

    @staticmethod
    def _append_wrist_roll_waypoint(path_deg, wrist_target):
        if wrist_target is None or len(path_deg) == 0:
            return path_deg
        last = path_deg[-1]
        final = [float(last[0]), float(last[1]), float(last[2]),
                 float(last[3]), float(wrist_target)]
        return list(path_deg) + [final]

    # ------------------------------------------------------------------
    # Goal-reachability check
    # ------------------------------------------------------------------
    # The soft-priority fallback A* can silently land far from the IK
    # goal when the masked grid blocks the goal cell (or when it's on a
    # disconnected island). Compare the planner's final cell to the goal
    # cell on the discretized joint grid; if they are farther than
    # max_cell_diff cells on any axis (default 1 ≈ 4 deg) the goal is
    # unreachable in this scene and the caller should not drive the arm
    # to a wrong-pose path. Callers may relax max_cell_diff for tasks
    # whose final approach (e.g. press) tolerates a coarser landing.
    @staticmethod
    def _check_goal_reachable(planner, plan_goal_deg, path_deg, max_cell_diff=1):
        if len(path_deg) == 0:
            return False, "empty path", None, None, None
        goal_idx = tuple(int(v) for v in planner.joints_to_idx(plan_goal_deg))
        final_idx = tuple(int(v) for v in planner.joints_to_idx(path_deg[-1]))
        cell_diff = [abs(g - f) for g, f in zip(goal_idx, final_idx)]
        if max(cell_diff) > max_cell_diff:
            return False, "no neighbor", goal_idx, final_idx, cell_diff
        return True, "ok", goal_idx, final_idx, cell_diff

    # ------------------------------------------------------------------
    # Planning entry point
    # ------------------------------------------------------------------
    def plan_joint_path_deg(self, arm, q_start_deg, q_goal_deg,
                            max_cell_diff=1, task_label="Pregrasp"):
        """Call astar_staged; return (path_deg, info_str).

        ``max_cell_diff`` bounds how far the planner's final cell may sit
        from the IK goal cell (per-axis, in joint-grid cells ≈ 4 deg).
        Default 1 keeps the strict grasp-pregrasp tolerance; tasks whose
        final approach tolerates a coarser landing (press) may pass a
        larger value.

        ``task_label`` prefixes the unreachable-goal error message so
        operators can tell which goal failed (default ``"Pregrasp"`` for
        grasp callers; the press handler passes ``"Prepress"``).
        """
        planner = self._get_planner(arm)
        # Plan-time path: force a fresh depth fetch so A* sees the current
        # scene, not whatever the background _depth_cb last buffered.
        voxels, info = self.latest_voxel_obstacles(arm, fresh=True)
        plan_goal, wrist_target = self._split_wrist_roll(q_goal_deg)
        # Log every planner input. Lets the operator verify that the real-
        # robot plan sees the SAME inputs as the simulation:
        #   - q_start: viewer uses a hardcoded HANG_START; real robot uses
        #     the live joint state (rospy ~astar_start_from_current=True),
        #     so arms not exactly at home produce different plans.
        #   - q_goal / plan_goal: should match if the IK pipeline is the
        #     same; wrist_target is split out for the two-stage rotation.
        #   - voxel_count: depends on the camera snapshot at this instant.
        # If these differ from the viewer run, A* can legitimately reach a
        # different cell even though the planner code is identical.
        voxel_count = int(voxels.shape[0]) if voxels.size else 0
        rospy.loginfo(
            "[voxel_adapter] plan inputs: "
            f"arm={arm}, "
            f"q_start (deg)={[round(float(v), 2) for v in q_start_deg]}, "
            f"q_goal (deg)={[round(float(v), 2) for v in q_goal_deg]} "
            f"(plan_goal after wrist split="
            f"{[round(float(v), 2) for v in plan_goal]}, "
            f"wrist_target={wrist_target}), "
            f"voxel_count={voxel_count}"
        )
        path_idx, path_deg = planner.astar_staged(
            q_start_deg=list(q_start_deg),
            q_goal_deg=plan_goal,
            voxel_indices=voxels if voxels.size else None,
            defer_joint=self.defer_joint,
            defer_cost_mult=self.defer_cost_mult,
            h_weight=self.h_weight,
            shortcut=self.shortcut,
            verbose=False,
        )
        ok, why, goal_idx, final_idx, cell_diff = self._check_goal_reachable(
            planner, plan_goal, path_deg, max_cell_diff=max_cell_diff,
        )
        # Always log the IK-goal-vs-final comparison (joint values + cell
        # indices). Lets the operator verify in real runs whether the
        # executed final configuration actually IS the IK goal cell or just
        # a near-neighbor. Discretization is ~4 deg/cell.
        ik_goal_joints_deg = [round(float(v), 2) for v in plan_goal]
        final_joints_deg = (
            [round(float(v), 2) for v in path_deg[-1]] if path_deg else []
        )
        rospy.loginfo(
            "[voxel_adapter] plan: "
            f"IK goal cell={goal_idx}, final cell={final_idx}, "
            f"cell diff={cell_diff} (step ≈ 4 deg/cell) | "
            f"IK goal joints (deg)={ik_goal_joints_deg}, "
            f"final waypoint joints (deg)={final_joints_deg}, "
            f"reachable={ok}."
        )
        if not ok:
            err_msg = (
                f"{task_label} not reachable (final cell too far from IK goal). "
                "Abort execution. "
                f"IK goal cell={goal_idx}, planned final cell={final_idx}, "
                f"per-axis cell diff={cell_diff} (step ≈ 4 deg/cell). "
                f"IK goal joints (deg)={ik_goal_joints_deg}, "
                f"planned final joints (deg)={final_joints_deg}."
            )
            rospy.logerr(f"[voxel_adapter] {err_msg}")
            raise RuntimeError(err_msg)
        path_deg = self._append_wrist_roll_waypoint(path_deg, wrist_target)
        return path_deg, f"{info} waypoints={len(path_deg)} reachable={ok}"

    def plan_joint_path_deg_no_obstacles(self, arm, q_start_deg, q_goal_deg):
        """Same as plan_joint_path_deg but explicitly disables voxel obstacles.

        Used by arm_node_abs's debug_print_both_paths flag to plan a
        baseline against which the obstacle-aware path can be compared.
        Does NOT touch the depth subscription or publish obstacles.
        """
        planner = self._get_planner(arm)
        plan_goal, wrist_target = self._split_wrist_roll(q_goal_deg)
        path_idx, path_deg = planner.astar_staged(
            q_start_deg=list(q_start_deg),
            q_goal_deg=plan_goal,
            voxel_indices=None,
            defer_joint=self.defer_joint,
            defer_cost_mult=self.defer_cost_mult,
            h_weight=self.h_weight,
            shortcut=self.shortcut,
            verbose=False,
        )
        ok, why, goal_idx, final_idx, cell_diff = self._check_goal_reachable(
            planner, plan_goal, path_deg
        )
        # Diagnostic-only path: warn but do NOT raise. The caller
        # (_debug_log_voxel_vs_no_obstacles) just logs both branches.
        if not ok:
            ik_goal_joints_deg = [round(float(v), 2) for v in plan_goal]
            final_joints_deg = (
                [round(float(v), 2) for v in path_deg[-1]] if path_deg else []
            )
            rospy.logwarn(
                "[voxel_adapter] cannot reach pregrasp (no_obstacles): "
                f"final cell {final_idx} not neighbor of IK goal cell "
                f"{goal_idx} (cell diff={cell_diff}, step ≈ 4 deg/cell). "
                f"IK goal joints (deg)={ik_goal_joints_deg}, "
                f"final waypoint joints (deg)={final_joints_deg}."
            )
        path_deg = self._append_wrist_roll_waypoint(path_deg, wrist_target)
        return path_deg, f"obstacles=disabled waypoints={len(path_deg)} reachable={ok}"
