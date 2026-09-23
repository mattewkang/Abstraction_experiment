#!/usr/bin/env python3
"""ROS1 node: detect berries in the robot head-camera stream, classify each by
ripeness (black / red / white), and report every berry's 3D position in the
robot torso frame (torso_link) so a picker can choose what to grasp.

Mirrors object_detector_node.py's contract: inference is on-demand. The node
caches only CameraInfo in the background and runs nothing until
/perception/get_berries is called. Each service call fetches one fresh RGB
plus a short stack of depth frames (~depth_temporal_frames) via
rospy.wait_for_message and per-pixel median-filters the depth stack to kill
single-frame RealSense noise, runs the berry detector + the ripeness
classifier, samples the median depth, back-projects to torso_link, and returns
ALL detected berries at once (parallel arrays) alongside publishing the
standard pose / json / annotated outputs.

Colour-aware depth alignment
----------------------------
A leaf between a berry and the camera puts the bbox centre on foreground
foliage, so its depth is the leaf's, not the berry's. Two cases:
  1. The bbox centre matches the predicted ripeness colour (a patch majority,
     not a single pixel) -> the centre is on the berry, sample there directly.
  2. The centre colour differs (a leaf occludes it) or the centre has no depth
     -> find the LARGEST connected colour-matching region inside the bbox (the
     visible berry surface), and sample depth at that region's centre. Using
     the largest connected region rather than the mean of every matching pixel
     keeps the sample on the actual berry when HSV noise or a neighbouring
     same-ripeness berry clips a bbox edge.
Disable with ~color_align:=false.

Subscribes
----------
/camera/color/image_raw                  (sensor_msgs/Image, bgr8/rgb8)
/camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1 mm)
/camera/color/camera_info                (sensor_msgs/CameraInfo)

Publishes (only when /perception/get_berries fires inference)
---------
/perception/berries_pose   (geometry_msgs/PoseArray) - berry centres in torso_link
/perception/berries_json   (std_msgs/String)         - per-berry metadata
/perception/berries_annotated_image (sensor_msgs/Image) - debug visualisation

The depth -> torso_link math is identical to object_detector_node.py (analytic
extrinsic from ~camera_translation / ~camera_rotation_ypr plus the fixed optical
rotation); it does not depend on the TF tree. The emitted frame label is
torso_link rather than object_detector_node's base_link, but the two refer to
the same physical frame in this project's convention, so the extrinsic is
unchanged.
"""

import json
import math
import os
import sys
import threading
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from g1_camera.srv import GetBerries, GetBerriesResponse


# ---------------------------------------------------------------------------
# cv_bridge replacement (Noetic's apt cv_bridge is built against numpy 1.x and
# crashes under the workspace's numpy 2.x). Same helper as object_detector_node.
# ---------------------------------------------------------------------------
_ENC_TO_DTYPE_CH = {
    "bgr8":   (np.uint8,   3),
    "rgb8":   (np.uint8,   3),
    "mono8":  (np.uint8,   1),
    "16UC1":  (np.uint16,  1),
    "mono16": (np.uint16,  1),
    "32FC1":  (np.float32, 1),
}


def imgmsg_to_cv2(msg: Image, desired_encoding: str = "passthrough") -> np.ndarray:
    enc = msg.encoding if desired_encoding == "passthrough" else desired_encoding
    if enc not in _ENC_TO_DTYPE_CH:
        raise ValueError(f"Unsupported encoding: {enc}")
    dtype, ch = _ENC_TO_DTYPE_CH[enc]
    buf = np.frombuffer(msg.data, dtype=dtype)
    img = buf.reshape(msg.height, msg.width, ch) if ch > 1 \
        else buf.reshape(msg.height, msg.width)
    if msg.is_bigendian and dtype().itemsize > 1:
        img = img.byteswap().view(img.dtype)
    if desired_encoding == "bgr8" and msg.encoding == "rgb8":
        img = img[..., ::-1]
    elif desired_encoding == "rgb8" and msg.encoding == "bgr8":
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def cv2_to_imgmsg(img: np.ndarray, encoding: str = "bgr8") -> Image:
    if encoding not in _ENC_TO_DTYPE_CH:
        raise ValueError(f"Unsupported encoding: {encoding}")
    dtype, ch = _ENC_TO_DTYPE_CH[encoding]
    if img.dtype != dtype:
        img = img.astype(dtype)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    msg = Image()
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = img.shape[1] * ch * dtype().itemsize
    msg.data = img.tobytes()
    return msg


# ---------------------------------------------------------------------------
# Geometry helpers (identical to object_detector_node.py)
# ---------------------------------------------------------------------------
def euler_ypr_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


# camera_color_optical_frame -> camera_link, matching g1_slam transforms.launch.
R_OPTICAL_TO_LINK = euler_ypr_to_matrix(-1.5708, 0.0, -1.5708)

# Ripeness classes that carry a colour gate for depth re-alignment.
_COLOR_CLASSES = ("red", "black", "white")

# Per-ripeness draw colour (BGR) for the annotated debug image.
_CLS_COLOR = {
    "red":   (60, 60, 220),
    "black": (60, 60, 60),
    "white": (230, 230, 230),
}
_DEFAULT_COLOR = (0, 200, 0)
_REALIGN_COLOR = (255, 0, 255)  # magenta dot when depth hopped to a colour blob


def berry_color_mask(region_bgr: np.ndarray, label: str) -> np.ndarray:
    """Boolean mask of pixels in `region_bgr` matching ripeness `label`.

    Heuristic HSV gates (OpenCV HSV: H 0-179, S/V 0-255), tuned against live
    frames; deliberately loose so a berry surface is not missed.
    """
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if label == "red":
        return ((h <= 10) | (h >= 170)) & (s >= 90) & (v >= 50)
    if label == "black":
        return v <= 70
    if label == "white":
        return (s <= 45) & (v >= 160)
    return np.ones(region_bgr.shape[:2], dtype=bool)


def resolve_model_path(model_path: str) -> str:
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    if os.path.exists(model_path):
        return model_path
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_model_path = os.path.join(package_dir, model_path)
    if os.path.exists(package_model_path):
        return package_model_path
    return model_path


# ---------------------------------------------------------------------------
# Berry detector wrapper (ultralytics direct, torch-only)
# ---------------------------------------------------------------------------
class BerryDetector:
    def __init__(self, model_path: str, device: str, conf: float, iou: float):
        from ultralytics import YOLO
        self.weights_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self._yolo = YOLO(model_path)
        self.names = dict(self._yolo.names)

    def __call__(self, bgr: np.ndarray):
        res = self._yolo.predict(bgr, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            out.append({
                "label": self.names[int(k)],
                "score": float(c),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })
        return out


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class BerryDetectorNode:
    def __init__(self):
        rospy.init_node("berry_detector_node")
        p = rospy.get_param

        # ---- frames / extrinsic / depth (mirror object_detector_node) -----
        self.base_frame = p("~base_frame", "torso_link")
        self.optical_frame = p("~camera_optical_frame",
                               "camera_color_optical_frame")
        self.depth_scale = float(p("~depth_scale", 0.001))
        self.min_depth = float(p("~min_depth", 0.15))
        self.max_depth = float(p("~max_depth", 5.0))
        self.patch_r = int(p("~depth_patch_radius", 3))
        # Number of depth frames median-stacked per service call. RealSense
        # depth on a ~1 cm berry wobbles several mm/cm frame-to-frame; a
        # per-pixel temporal median collapses that jitter (and fills transient
        # holes) so a stationary berry returns a stable z. 1 = single frame
        # (legacy). ~5 @15 fps adds ~0.3 s latency per call.
        self.depth_temporal_frames = max(1, int(p("~depth_temporal_frames", 5)))
        # Berry is round: shift the deprojected front surface along the camera
        # ray by its radius to reach the centre. Default 0.005 m = 1 cm berry.
        self.berry_radius_m = float(p("~berry_radius_m", 0.005))
        # Constant residual-calibration trim SUBTRACTED from the base-frame
        # position. Default x=0.02 cancels the measured +2 cm forward bias.
        self.pos_offset = np.array([
            float(p("~x_offset", 0.02)),
            float(p("~y_offset", 0.0)),
            float(p("~z_offset", 0.0)),
        ], dtype=np.float64)

        t = p("~camera_translation", [0.0576235, 0.02753, 0.44487])
        r = p("~camera_rotation_ypr", [0.0, 0.8307767239493009, 0.0])
        self.t_base_opt = np.asarray(t, dtype=np.float64).reshape(3)
        R_base_cam = euler_ypr_to_matrix(float(r[0]), float(r[1]), float(r[2]))
        self.R_base_opt = R_base_cam @ R_OPTICAL_TO_LINK

        # ---- colour-aware depth alignment ---------------------------------
        self.color_align = bool(p("~color_align", True))
        self.color_min_pixels = int(p("~color_min_pixels", 15))
        # Fraction of the centre patch that must match the ripeness colour for
        # the bbox centre to count as "on the berry" (case 1: align directly to
        # the centre). Below it, a leaf is assumed to occlude the centre and the
        # sample hops to the largest colour-matching region (case 2). A single
        # centre pixel is too noisy, so the decision is a patch majority.
        self.color_center_frac = float(p("~color_center_frac", 0.5))

        # ---- topics -------------------------------------------------------
        self.rgb_topic = p("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = p("~depth_topic",
                             "/camera/aligned_depth_to_color/image_raw")
        info_topic = p("~info_topic", "/camera/color/camera_info")
        out_pose = p("~berries_pose_topic", "/perception/berries_pose")
        out_json = p("~berries_json_topic", "/perception/berries_json")
        out_img = p("~annotated_topic", "/perception/berries_annotated_image")
        self.publish_annotated = bool(p("~publish_annotated", True))
        self.service_frame_timeout = float(p("~service_frame_timeout_sec", 1.0))

        # ---- detector + ripeness classifier (ultralytics / torch) ---------
        configured_model = p("~model", "models/unified_detector_yolo11s.pt")
        resolved_model = resolve_model_path(configured_model)
        if not os.path.exists(resolved_model):
            rospy.logfatal("Berry detector weights not found: %s", resolved_model)
            sys.exit(1)
        device = p("~device", "cpu")
        self.detector = BerryDetector(
            model_path=resolved_model, device=device,
            conf=float(p("~conf_threshold", 0.25)),
            iou=float(p("~iou_threshold", 0.45)),
        )

        # YOLO bboxes are tight; pad before the ripeness crop for context.
        self.classifier_pad_frac = float(p("~classifier_pad_frac", 0.10))
        self.classifier = None
        cls_path = str(p("~classifier",
                         "models/unified_classifier_yolo11s_cls.pt") or "")
        if cls_path:
            resolved_cls = resolve_model_path(cls_path)
            if os.path.exists(resolved_cls):
                # ButtonClassifier is a generic torch YOLOv8-cls wrapper; reuse
                # it for berry ripeness. roslaunch may not put this script's dir
                # on sys.path, so inject it before the sibling import.
                _here = os.path.dirname(os.path.abspath(__file__))
                if _here not in sys.path:
                    sys.path.insert(0, _here)
                from button_classifier import ButtonClassifier
                self.classifier = ButtonClassifier(resolved_cls, device=device)
                rospy.loginfo("ripeness classifier loaded: %s (classes=%s)",
                              resolved_cls, self.classifier.class_names)
            else:
                rospy.logwarn("ripeness classifier weights not found at %s; "
                              "labels stay 'berry'.", resolved_cls)

        # ---- ROS I/O ------------------------------------------------------
        self.K: Optional[np.ndarray] = None
        self._info_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self.pub_pose = rospy.Publisher(out_pose, PoseArray, queue_size=5)
        self.pub_json = rospy.Publisher(out_json, String, queue_size=5)
        self.pub_img = (rospy.Publisher(out_img, Image, queue_size=2)
                        if self.publish_annotated else None)
        self.service = rospy.Service(
            p("~get_berries_service", "/perception/get_berries"),
            GetBerries, self._handle_get_berries)

        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=1)
        rospy.loginfo(
            "berry_detector_node ready. weights=%s device=%s color_align=%s "
            "berry_radius=%.4fm offset=(%.3f,%.3f,%.3f)",
            self.detector.weights_path, device, self.color_align,
            self.berry_radius_m, self.pos_offset[0], self.pos_offset[1],
            self.pos_offset[2])

    # -----------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        with self._info_lock:
            self.K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    def _temporal_median_depth(self, frames: List[np.ndarray]) -> np.ndarray:
        """Per-pixel median across the depth stack, ignoring zero/no-data, in
        the raw depth units (same as a single frame, so depth_scale still
        applies downstream). Holes with no valid sample in any frame become
        NaN, which `_sample_depth` / `_sample_depth_masked` already treat as
        invalid (NaN > 0 is False). Collapses single-frame RealSense depth
        noise so a stationary berry's z stops wobbling between calls."""
        if len(frames) == 1:
            d = frames[0].astype(np.float32)
            d[d <= 0] = np.nan
            return d
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        stack[stack <= 0] = np.nan
        # nanmedian warns on all-NaN pixels (no valid sample in any frame);
        # that is the intended "hole" result, so silence the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        return med.astype(np.float32)

    def _sample_depth(self, depth: np.ndarray, u: int, v: int) -> float:
        """Median-of-patch depth in metres at (u,v), or NaN if invalid."""
        h, w = depth.shape[:2]
        r = self.patch_r
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth[v0:v1, u0:u1]
        if patch.size == 0:
            return float("nan")
        vals = patch[patch > 0].astype(np.float32) * self.depth_scale
        if vals.size == 0:
            return float("nan")
        z = float(np.median(vals))
        if not (self.min_depth <= z <= self.max_depth):
            return float("nan")
        return z

    def _sample_depth_masked(self, depth, x0, y0, x1, y1, mask):
        """Locate the berry behind a leaf: find the LARGEST connected
        colour-matching region inside the bbox (the visible berry surface),
        and return (median depth over that region's valid-depth pixels,
        full-frame u, full-frame v of the region centre).

        Using the largest connected component — not the mean of every
        colour-matching pixel — is what keeps the sample on the actual berry:
        scattered HSV noise or a neighbouring same-ripeness berry clipping a
        bbox edge would otherwise drag the centroid off-target and mix two
        depths into the median. Returns (nan, None, None) when no region clears
        ``~color_min_pixels`` valid-depth pixels."""
        sub = depth[y0:y1, x0:x1].astype(np.float32) * self.depth_scale
        depth_ok = (sub >= self.min_depth) & (sub <= self.max_depth)

        # Connected components of the colour mask; each label is one region.
        num, labels, _stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        best_lbl, best_area = -1, 0
        for lbl in range(1, num):  # skip background (label 0)
            area = int(np.count_nonzero((labels == lbl) & depth_ok))
            if area > best_area:
                best_lbl, best_area = lbl, area
        if best_lbl < 0 or best_area < self.color_min_pixels:
            return float("nan"), None, None

        blob = (labels == best_lbl) & depth_ok
        z = float(np.median(sub[blob]))

        # Region centre; snap to the nearest in-region pixel because the
        # centroid can fall in a concave notch outside the mask.
        cu, cv = centroids[best_lbl]  # (x, y) in sub-image coords
        ys, xs = np.nonzero(blob)
        j = int(np.argmin((xs - cu) ** 2 + (ys - cv) ** 2))
        return z, x0 + int(xs[j]), y0 + int(ys[j])

    def _pixel_to_base(self, u, v, z, K) -> np.ndarray:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        p_surf = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z],
                          dtype=np.float64)
        ray_norm = float(np.linalg.norm(p_surf))
        if self.berry_radius_m > 0.0 and ray_norm > 1e-6:
            p_opt = p_surf + (p_surf / ray_norm) * self.berry_radius_m
        else:
            p_opt = p_surf
        return self.R_base_opt @ p_opt + self.t_base_opt - self.pos_offset

    # -----------------------------------------------------------------------
    def _run_inference(self) -> Tuple[List[dict], bool, Optional[rospy.Time]]:
        """Fetch one fresh RGB + a temporally median-filtered depth stack,
        detect + classify + locate every berry. Returns (records, ok, stamp).
        ok is False on missing CameraInfo or a wait_for_message timeout.
        Serialised by _inference_lock."""
        with self._inference_lock:
            with self._info_lock:
                K = None if self.K is None else self.K.copy()
            if K is None:
                rospy.logwarn_throttle(5.0, "Waiting for CameraInfo...")
                return [], False, None

            try:
                rgb_msg = rospy.wait_for_message(
                    self.rgb_topic, Image, timeout=self.service_frame_timeout)
                depth_msgs = [
                    rospy.wait_for_message(
                        self.depth_topic, Image,
                        timeout=self.service_frame_timeout)
                    for _ in range(self.depth_temporal_frames)
                ]
            except rospy.ROSException as exc:
                rospy.logwarn("wait_for_message timed out: %s", exc)
                return [], False, None

            try:
                bgr = imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                depth_frames = [
                    imgmsg_to_cv2(m, desired_encoding="passthrough")
                    for m in depth_msgs
                ]
            except Exception as e:  # noqa: BLE001
                rospy.logerr("frame decode failed: %s", e)
                return [], False, None
            # Per-pixel temporal median over the depth stack (mm, holes -> NaN),
            # consumed unchanged by _sample_depth / _sample_depth_masked.
            depth = self._temporal_median_depth(depth_frames)
            stamp = rgb_msg.header.stamp

            detections = self.detector(bgr)
            h, w = bgr.shape[:2]

            # Ripeness classification (one batched forward pass).
            if self.classifier is not None and detections:
                pad = self.classifier_pad_frac
                crops, idxs = [], []
                for i, det in enumerate(detections):
                    bx1, by1, bx2, by2 = det["bbox"]
                    bw, bh = bx2 - bx1, by2 - by1
                    x1c = max(0, int(round(bx1 - bw * pad)))
                    y1c = max(0, int(round(by1 - bh * pad)))
                    x2c = min(w, int(round(bx2 + bw * pad)))
                    y2c = min(h, int(round(by2 + bh * pad)))
                    if x2c > x1c and y2c > y1c:
                        crops.append(bgr[y1c:y2c, x1c:x2c])
                        idxs.append(i)
                if crops:
                    preds = self.classifier.predict_batch(crops)
                    for i, (name, prob) in zip(idxs, preds):
                        detections[i]["ripeness"] = name
                        detections[i]["ripeness_score"] = prob

            records = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                ripeness = det.get("ripeness", "berry")
                ripeness_score = det.get("ripeness_score", 0.0)
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))
                realigned = False
                z = self._sample_depth(depth, u, v)

                # Colour-aware depth alignment.
                #   Case 1: the bbox centre matches the ripeness colour -> the
                #           centre is on the berry, keep the centre sample.
                #   Case 2: the centre colour differs (a leaf occludes it) or
                #           the centre has no depth -> hop the sample to the
                #           centre of the largest colour-matching region.
                if self.color_align and ripeness in _COLOR_CLASSES:
                    bx0, by0 = max(0, int(round(x1))), max(0, int(round(y1)))
                    bx1i, by1i = min(w, int(round(x2))), min(h, int(round(y2)))
                    if bx1i > bx0 and by1i > by0:
                        mask = berry_color_mask(bgr[by0:by1i, bx0:bx1i], ripeness)
                        mu, mv = u - bx0, v - by0
                        # Patch majority around the centre (robust to a single
                        # noisy/specular centre pixel), sized to the depth patch.
                        r = self.patch_r
                        wy0, wy1 = max(0, mv - r), min(mask.shape[0], mv + r + 1)
                        wx0, wx1 = max(0, mu - r), min(mask.shape[1], mu + r + 1)
                        win = mask[wy0:wy1, wx0:wx1]
                        center_on_color = (
                            win.size > 0
                            and float(win.mean()) >= self.color_center_frac)
                        if not center_on_color or not math.isfinite(z):
                            zc, uc, vc = self._sample_depth_masked(
                                depth, bx0, by0, bx1i, by1i, mask)
                            if math.isfinite(zc):
                                z, u, v, realigned = zc, uc, vc, True

                rec = {
                    "ripeness": ripeness,
                    "ripeness_score": float(ripeness_score),
                    "score": det["score"],
                    "uv": [u, v],
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "realigned": realigned,
                    "valid_depth": math.isfinite(z),
                }
                if math.isfinite(z):
                    p_base = self._pixel_to_base(u, v, z, K)
                    rec["position_base"] = [float(p_base[0]), float(p_base[1]),
                                            float(p_base[2])]
                records.append(rec)

            self._publish(records, stamp, bgr)
            return records, True, stamp

    def _publish(self, records, stamp, bgr):
        header = Header(stamp=stamp, frame_id=self.base_frame)
        pose_array = PoseArray(header=header)
        for r in records:
            if not r["valid_depth"]:
                continue
            px, py, pz = r["position_base"]
            pose = Pose()
            pose.position = Point(px, py, pz)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        self.pub_pose.publish(pose_array)
        self.pub_json.publish(String(data=json.dumps({
            "stamp": stamp.to_sec() if stamp is not None else 0.0,
            "frame_id": self.base_frame,
            "berries": records,
        })))
        if self.pub_img is not None:
            self._publish_debug_image(bgr, records, stamp)

    def _handle_get_berries(self, request):
        response = GetBerriesResponse()
        response.frame_id = self.base_frame
        ripeness_filter = request.ripeness.strip()

        records, ok, _ = self._run_inference()
        if not ok:
            response.success = False
            response.count = 0
            response.message = ("No detection result available "
                                "(camera frame not yet received)")
            return response

        berries = [r for r in records if r["valid_depth"]]
        if ripeness_filter:
            berries = [r for r in berries
                       if r["ripeness"] == ripeness_filter]

        response.success = True
        response.count = len(berries)
        for r in berries:
            px, py, pz = r["position_base"]
            response.x.append(px)
            response.y.append(py)
            response.z.append(pz)
            response.ripeness_labels.append(r["ripeness"])
            response.det_scores.append(r["score"])
            response.ripeness_scores.append(r["ripeness_score"])
        filt = f" ripeness='{ripeness_filter}'" if ripeness_filter else ""
        response.message = (f"{response.count} berr"
                            f"{'y' if response.count == 1 else 'ies'} "
                            f"with valid depth{filt}")
        return response

    # -----------------------------------------------------------------------
    def _publish_debug_image(self, bgr, records, stamp):
        vis = bgr.copy()
        for r in records:
            x1, y1, x2, y2 = (int(c) for c in r["bbox"])
            color = _CLS_COLOR.get(r["ripeness"], _DEFAULT_COLOR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            u, v = r["uv"]
            cv2.circle(vis, (int(u), int(v)), 3,
                       _REALIGN_COLOR if r["realigned"] else color, -1)
            if r["valid_depth"]:
                bx, by, bz = r["position_base"]
                txt = (f'{r["ripeness"]} {r["ripeness_score"]:.2f} '
                       f'b=({bx:+.2f},{by:+.2f},{bz:+.2f})m')
            else:
                txt = f'{r["ripeness"]} {r["ripeness_score"]:.2f} (no depth)'
            cv2.putText(vis, txt, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"berries: {len(records)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        msg = cv2_to_imgmsg(vis, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.optical_frame
        self.pub_img.publish(msg)


def main():
    try:
        BerryDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:  # noqa: BLE001
        rospy.logfatal("berry_detector_node crashed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
