#!/usr/bin/env python3
"""ROS1 node: detect objects in the robot head camera stream and publish
their 3D position in the robot base frame.

Inference is on-demand: the node caches the latest synchronised
RGB+depth+intrinsics frame on every callback but does NOT run YOLO until
/perception/get_object_position is called. Each service call runs a fresh
inference on the cached frame and publishes the standard pose / json /
annotated outputs alongside the service response. GPU/CPU stays idle
between service calls.

Subscribes
----------
/camera/color/image_raw                 (sensor_msgs/Image, bgr8/rgb8)
/camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1 mm)
/camera/color/camera_info               (sensor_msgs/CameraInfo)

Publishes (only when /perception/get_object_position fires inference)
---------
/perception/objects_pose   (geometry_msgs/PoseArray) - positions in base frame
/perception/objects_json   (std_msgs/String)         - per-detection metadata
/perception/annotated_image (sensor_msgs/Image)      - debug visualisation
/tf_static base_link -> camera_color_optical_frame (only when
    ~publish_static_tf is true; off by default because g1_slam owns this
    TF chain via URDF + g1_slam/launch/transforms.launch).

The transform used is
    base_link  --(translation, yaw-pitch-roll)-->  camera_link
    camera_link --(fixed optical rotation)---->   camera_color_optical_frame
so that a point (X,Y,Z) obtained by back-projecting the aligned depth
(which lives in the optical frame) can be brought into the base frame.
This composition is computed internally from ~camera_translation /
~camera_rotation_ypr and is independent of the TF tree, so projection
results do not require the static TF above to be published.
"""

import hashlib
import json
import math
import os
import sys
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

import rospy
import tf2_ros
from geometry_msgs.msg import (Point, Pose, PoseArray, Quaternion,
                               TransformStamped)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from g1_camera.srv import GetObjectPosition, GetObjectPositionResponse


# ---------------------------------------------------------------------------
# cv_bridge replacement
# ---------------------------------------------------------------------------
# Noetic's apt-installed cv_bridge is built against numpy 1.x and fails under
# numpy 2 with "numpy.dtype size changed". We only need a couple of encodings
# (bgr8 / rgb8 / mono8 / 16UC1 / 32FC1), so convert manually.
_ENC_TO_DTYPE_CH = {
    "bgr8":   (np.uint8,   3),
    "rgb8":   (np.uint8,   3),
    "mono8":  (np.uint8,   1),
    "8UC1":   (np.uint8,   1),
    "8UC3":   (np.uint8,   3),
    "16UC1":  (np.uint16,  1),
    "mono16": (np.uint16,  1),
    "32FC1":  (np.float32, 1),
}


def imgmsg_to_cv2(msg: Image, desired_encoding: str = "passthrough"
                  ) -> np.ndarray:
    enc = msg.encoding if desired_encoding == "passthrough" else desired_encoding
    if enc not in _ENC_TO_DTYPE_CH:
        raise ValueError(f"Unsupported encoding: {enc}")
    dtype, ch = _ENC_TO_DTYPE_CH[enc]
    buf = np.frombuffer(msg.data, dtype=dtype)
    img = buf.reshape(msg.height, msg.width, ch) if ch > 1 \
        else buf.reshape(msg.height, msg.width)
    if msg.is_bigendian and dtype().itemsize > 1:
        img = img.byteswap().view(img.dtype)
    # If caller explicitly asked for bgr8 but source was rgb8 (or vice versa)
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
# Small geometry helpers (no tf_conversions dep required)
# ---------------------------------------------------------------------------
def euler_ypr_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Right-handed, ROS convention."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def matrix_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w)."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


# Fixed rotation: camera_color_optical_frame -> camera_link.
# Matches g1_bottle_localization's static transform:
#   camera_link -> camera_color_optical_frame yaw=-1.5708 pitch=0 roll=-1.5708
R_OPTICAL_TO_LINK = euler_ypr_to_matrix(-1.5708, 0.0, -1.5708)


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------
class YoloDetector:
    def __init__(self, model_path: str, device: str,
                 conf: float, iou: float,
                 target_classes: Optional[List[str]]):
        from ultralytics import YOLO
        self.weights_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self._yolo = YOLO(model_path)
        self.names = dict(self._yolo.names)
        if target_classes:
            wanted = set(target_classes)
            self.class_filter = [i for i, n in self.names.items() if n in wanted]
            if not self.class_filter:
                rospy.logwarn("target_classes %s match no model class; "
                              "keeping all.", target_classes)
                self.class_filter = None
        else:
            self.class_filter = None

    def __call__(self, bgr: np.ndarray):
        kwargs = dict(conf=self.conf, iou=self.iou, device=self.device,
                      verbose=False)
        if self.class_filter is not None:
            kwargs["classes"] = self.class_filter
        res = self._yolo.predict(bgr, **kwargs)[0]
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


def resolve_model_path(model_path: str) -> Tuple[str, bool]:
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path, False
    if os.path.exists(model_path):
        return model_path, False
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_model_path = os.path.join(package_dir, model_path)
    if os.path.exists(package_model_path):
        return package_model_path, False
    return model_path, False


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(_PACKAGE_DIR, "config", "model_weights.yaml")


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_weight(weight_path: str) -> None:
    """Fatal-exit if a manifest-pinned weight is missing or its sha256 drifts.

    Looks up the matching entry in config/model_weights.yaml by file basename.
    Weights not listed in the manifest are skipped (caller's responsibility).
    For listed weights, missing or mismatch is unrecoverable and aborts the
    node so the operator runs fetch_weights.py before retrying.
    """
    try:
        with open(_MANIFEST_PATH) as fh:
            spec = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        rospy.logfatal(
            "model_weights.yaml missing at %s; run fetch_weights.py.",
            _MANIFEST_PATH,
        )
        sys.exit(1)
    basename = os.path.basename(weight_path)
    entry = next(
        (v for v in (spec.get("weights") or {}).values()
         if os.path.basename(v.get("filename", "")) == basename),
        None,
    )
    if entry is None:
        return
    if not os.path.exists(weight_path):
        rospy.logfatal(
            "Pinned weight missing: %s. Run "
            "`rosrun g1_camera fetch_weights.py` to download from HF Hub.",
            weight_path,
        )
        sys.exit(1)
    digest = _sha256_of(weight_path)
    expected = entry.get("sha256", "")
    if digest != expected:
        rospy.logfatal(
            "Pinned weight sha256 mismatch: %s (expected %s, got %s). "
            "Re-run fetch_weights.py to reconcile.",
            weight_path, expected, digest,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node("object_detector_node")

        # ---- params -------------------------------------------------------
        p = rospy.get_param
        self.base_frame = p("~base_frame", "base_link")
        self.optical_frame = p("~camera_optical_frame",
                               "camera_color_optical_frame")
        self.depth_scale = float(p("~depth_scale", 0.001))
        self.min_depth = float(p("~min_depth", 0.15))
        self.max_depth = float(p("~max_depth", 5.0))
        self.patch_r = int(p("~depth_patch_radius", 3))
        self.bottle_radius_m = float(p("~bottle_radius_m", 0.033))
        # Per-class ray-direction offset applied to the deprojected surface
        # depth. For round objects (bottle) we shift along the camera
        # ray by the object radius so the returned point is the object
        # centre, not the front face. For flat objects (button) the
        # surface IS the press target so the offset must be 0.0.
        # Classes not listed in ~class_radius_m fall back to
        # ~bottle_radius_m. config/params.yaml is the authoritative source.
        self.class_radius_m = dict(p("~class_radius_m", {}) or {})
        self.publish_annotated = bool(p("~publish_annotated", True))
        # How long a service call will wait for the FIRST frame to arrive when
        # the cache is still empty (node just started, or camera down). Frames
        # are no longer fetched per call via wait_for_message -- see the
        # streaming cache below -- so this only bounds cold start.
        self.service_frame_timeout = float(p("~service_frame_timeout_sec", 1.0))

        # ---- streaming frame cache ----------------------------------------
        # RGB + depth are subscribed continuously and the newest pair is
        # cached, instead of pulling one pair per service call. Two reasons:
        #   1. wait_for_message subscribes, waits, unsubscribes on every call.
        #      On a slow or hiccuping stream (the D435i drops to <1 Hz when it
        #      renegotiates USB) that window misses and grasp_bottle fails with
        #      "No detection result available" even though the camera is alive.
        #   2. Continuous inference keeps /perception/annotated_image and
        #      objects_pose/json live, so RViz / rqt_image_view show the
        #      current scene rather than freezing on the last service call.
        #
        # ~inference_rate_hz drives the background inference timer. 0 disables
        # it and restores pure on-demand behaviour (cache still fills, so the
        # service is still immune to problem 1). Measured cost of one pass at
        # 1280x720 on this CPU is ~82 ms, so 5 Hz uses roughly 40% of one
        # core-set; raise only if you have headroom.
        self.inference_rate_hz = float(p("~inference_rate_hz", 5.0))
        # Safety gate. A cached frame means the service can answer instantly,
        # but it must never answer from a stale one: the returned xyz drives a
        # real arm motion, so acting on a frame from before the object (or the
        # arm) moved would put the gripper in the wrong place. Detections
        # derived from frames older than this are refused rather than returned.
        self.max_frame_age = float(p("~max_frame_age_sec", 1.0))
        # RGB and depth are cached independently, so their stamps can differ.
        # Warn (throttled) past this skew; the pixel->depth lookup assumes the
        # two describe the same instant.
        self.max_rgb_depth_skew = float(p("~max_rgb_depth_skew_sec", 0.2))
        # g1_slam already publishes the full base_link -> ... ->
        # camera_color_optical_frame TF chain via URDF + transforms.launch.
        # Publishing another static parent here would create a TF2 double-
        # parent conflict, so the broadcast is opt-in.
        self.publish_static_tf = bool(p("~publish_static_tf", False))

        t = p("~camera_translation", [0.04765, 0.0, 0.46268])
        r = p("~camera_rotation_ypr", [0.0, 0.8378, 0.0])  # yaw, pitch, roll
        yaw, pitch, roll = float(r[0]), float(r[1]), float(r[2])
        self.t_base_cam = np.asarray(t, dtype=np.float64).reshape(3)
        self.R_base_cam = euler_ypr_to_matrix(yaw, pitch, roll)
        # Combined: base <- optical  (T_base_opt = T_base_link * T_link_opt)
        self.R_base_opt = self.R_base_cam @ R_OPTICAL_TO_LINK
        self.t_base_opt = self.t_base_cam.copy()  # link & optical coincide in origin

        self.rgb_topic = p("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = p("~depth_topic",
                             "/camera/aligned_depth_to_color/image_raw")
        info_topic = p("~info_topic", "/camera/color/camera_info")

        out_pose = p("~objects_pose_topic", "/perception/objects_pose")
        out_json = p("~objects_json_topic", "/perception/objects_json")
        out_img = p("~annotated_topic", "/perception/annotated_image")

        # ---- detector -----------------------------------------------------
        # Default matches config/params.yaml so `rosrun` without a param file
        # behaves the same as `roslaunch object_detector.launch`.
        configured_model = p("~model", "models/unified_detector_yolo11s.pt")
        resolved_model, used_fallback = resolve_model_path(configured_model)
        if used_fallback:
            rospy.logwarn("Requested model '%s' not found; using '%s' instead.",
                          configured_model, resolved_model)
        verify_pinned_weight(resolved_model)
        self.detector = YoloDetector(
            model_path=resolved_model,
            device=p("~device", "cpu"),
            conf=float(p("~conf_threshold", 0.35)),
            iou=float(p("~iou_threshold", 0.45)),
            target_classes=list(p("~target_classes", []) or []),
        )

        # ---- optional fine-grained classifier for button --------------
        # Every `button` detection is cropped from the RGB frame
        # and classified into the trained button classes
        # (up, down, 1, B1..B6, open, close, emergency_yellow, phone).
        # If the predicted probability >= ~button_classifier_conf, the
        # primary `label` is replaced with the fine class so downstream
        # consumers (e.g. /perception/get_object_position) can ask for
        # "up" / "B2" / "open" directly.
        self.button_classifier = None
        self.button_classifier_conf = float(p("~button_classifier_conf", 0.15))
        # YOLO bboxes are tight; training crops had more background context
        # around the button. Padding the bbox before classification typically
        # boosts confidence by 10-30 percentage points on live frames.
        self.button_pad_frac = float(p("~button_classifier_pad_frac", 0.15))
        button_cls_path = str(p("~button_classifier", "models/button_cls.pt") or "")
        if button_cls_path:
            resolved_cls, _ = resolve_model_path(button_cls_path)
            verify_pinned_weight(resolved_cls)
            if os.path.exists(resolved_cls):
                try:
                    # roslaunch may not add this script's directory to
                    # sys.path, so the sibling module wouldn't be found
                    # via a normal `from button_classifier import ...`.
                    # Add it explicitly before the import.
                    _here = os.path.dirname(os.path.abspath(__file__))
                    if _here not in sys.path:
                        sys.path.insert(0, _here)
                    from button_classifier import ButtonClassifier
                    self.button_classifier = ButtonClassifier(
                        resolved_cls, device=p("~device", "cpu"))
                    rospy.loginfo(
                        "button classifier loaded: %s (classes=%d, conf>=%.2f -> rename)",
                        resolved_cls, len(self.button_classifier.class_names),
                        self.button_classifier_conf,
                    )
                except Exception as e:  # noqa: BLE001
                    rospy.logwarn("button classifier disabled: %s", e)
            else:
                rospy.loginfo(
                    "button classifier weights not found at %s; skipping.",
                    resolved_cls)

        # How _handle_get_object_position ranks valid detections when the
        # caller passes index. Default "z_desc" -> index 0 is the physically
        # highest detection (largest z in base_link). Use cases:
        #   z_desc     -- pick the upper button / upper shelf item.
        #   z_asc      -- pick the lower one.
        #   score_desc -- pick the highest-confidence one (legacy default).
        self.sort_by = str(p("~sort_by", "z_desc")).strip().lower()
        if self.sort_by not in ("z_desc", "z_asc", "score_desc"):
            rospy.logwarn(
                "Unknown ~sort_by='%s'; falling back to 'z_desc'", self.sort_by
            )
            self.sort_by = "z_desc"

        # ---- ROS I/O ------------------------------------------------------
        self.K: Optional[np.ndarray] = None
        self._info_lock = threading.Lock()
        # _inference_lock serialises the background timer against concurrent
        # service calls so the model is not double-booked. CameraInfo is
        # cached by _info_cb under _info_lock because intrinsics are static
        # per stream.
        self._inference_lock = threading.Lock()

        # Newest RGB / depth message, replaced in place by the subscribers.
        # _frame_lock is held only for the pointer swap, never across decode
        # or inference, so a slow inference pass cannot stall the callbacks.
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        # Newest completed inference: (records, stamp). Served directly to
        # get_object_position when still within ~max_frame_age_sec.
        self._result_lock = threading.Lock()
        self._latest_records: List[dict] = []
        self._latest_stamp: Optional[rospy.Time] = None

        self.pub_pose = rospy.Publisher(out_pose, PoseArray, queue_size=5)
        self.pub_json = rospy.Publisher(out_json, String, queue_size=5)
        self.pub_img = (rospy.Publisher(out_img, Image, queue_size=2)
                        if self.publish_annotated else None)
        self.position_service = rospy.Service(
            p("~get_object_position_service", "/perception/get_object_position"),
            GetObjectPosition,
            self._handle_get_object_position,
        )

        if self.publish_static_tf:
            self.static_tf = tf2_ros.StaticTransformBroadcaster()
            self._publish_static_tf()
        else:
            rospy.loginfo(
                "publish_static_tf disabled; relying on g1_slam transforms.launch "
                "for %s -> %s.", self.base_frame, self.optical_frame,
            )

        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=1)
        # queue_size=1: only the newest frame matters, never a backlog.
        # buff_size must exceed one message (1280x720x3 ~= 2.7 MB) or rospy
        # reassembles across TCP reads and lags badly on large images.
        rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb,
                         queue_size=1, buff_size=2 ** 24)

        if self.inference_rate_hz > 0.0:
            self._inference_timer = rospy.Timer(
                rospy.Duration(1.0 / self.inference_rate_hz),
                self._inference_timer_cb,
            )
            rospy.loginfo("continuous inference at %.1f Hz",
                          self.inference_rate_hz)
        else:
            self._inference_timer = None
            rospy.loginfo("continuous inference disabled "
                          "(~inference_rate_hz=0); on-demand only")

        rospy.loginfo(
            "object_detector_node ready. weights=%s device=%s",
            self.detector.weights_path, self.detector.device,
        )

    # -----------------------------------------------------------------------
    def _rgb_cb(self, msg: Image):
        with self._frame_lock:
            self._latest_rgb = msg

    def _depth_cb(self, msg: Image):
        with self._frame_lock:
            self._latest_depth = msg

    def _inference_timer_cb(self, _event):
        """Background pass. Skipped silently while no frames have arrived, so
        a downed camera costs nothing. Never raises into the timer thread."""
        try:
            self._run_inference()
        except Exception as e:  # noqa: BLE001
            rospy.logerr_throttle(5.0, "background inference failed: %s", e)

    def _take_frames(self, wait_timeout: float = 0.0):
        """Return the newest cached (rgb_msg, depth_msg), or (None, None).

        Waits up to wait_timeout for the cache to fill, which only matters on
        cold start; steady state returns immediately."""
        deadline = rospy.get_time() + max(0.0, wait_timeout)
        while not rospy.is_shutdown():
            with self._frame_lock:
                rgb, depth = self._latest_rgb, self._latest_depth
            if rgb is not None and depth is not None:
                return rgb, depth
            if rospy.get_time() >= deadline:
                return None, None
            rospy.sleep(0.02)
        return None, None

    # -----------------------------------------------------------------------
    def _publish_static_tf(self):
        """Broadcast base_link -> camera_color_optical_frame so downstream
        consumers can use tf if they want."""
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = self.optical_frame
        msg.transform.translation.x = float(self.t_base_opt[0])
        msg.transform.translation.y = float(self.t_base_opt[1])
        msg.transform.translation.z = float(self.t_base_opt[2])
        qx, qy, qz, qw = matrix_to_quat(self.R_base_opt)
        msg.transform.rotation = Quaternion(qx, qy, qz, qw)
        self.static_tf.sendTransform(msg)

    def _info_cb(self, msg: CameraInfo):
        with self._info_lock:
            self.K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    # -----------------------------------------------------------------------
    def _sample_depth(self, depth: np.ndarray, u: int, v: int) -> float:
        """Median-of-patch depth in metres. Returns NaN if invalid."""
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

    def _pixel_to_camera(self, u: float, v: float, z: float,
                          K: np.ndarray) -> np.ndarray:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def _surface_to_object_center(self, p_surface_opt: np.ndarray,
                                   radius_m: float) -> np.ndarray:
        # Push the deprojected surface point along the camera ray by
        # `radius_m`. For flat objects pass 0.0 to leave the surface
        # untouched (the surface IS the press/touch target).
        if radius_m <= 0.0:
            return p_surface_opt
        ray_norm = float(np.linalg.norm(p_surface_opt))
        if ray_norm <= 1e-6:
            return p_surface_opt
        return p_surface_opt + (p_surface_opt / ray_norm) * radius_m

    def _camera_to_base(self, p_opt: np.ndarray) -> np.ndarray:
        return self.R_base_opt @ p_opt + self.t_base_opt

    # -----------------------------------------------------------------------
    def _run_inference(self, wait_timeout: float = 0.0
                       ) -> Tuple[List[dict], Optional[rospy.Time]]:
        """Run YOLO on the newest cached RGB + depth pair, publish pose /
        json / annotated_image, cache and return (records, rgb stamp).

        Returns ([], None) when CameraInfo is missing or no frame has been
        received yet (waiting up to wait_timeout for the first one).
        Serialised by _inference_lock so the background timer and a concurrent
        service call do not race the model."""
        with self._inference_lock:
            with self._info_lock:
                K = None if self.K is None else self.K.copy()
            if K is None:
                rospy.logwarn_throttle(5.0, "Waiting for CameraInfo...")
                return [], None

            rgb_msg, depth_msg = self._take_frames(wait_timeout)
            if rgb_msg is None or depth_msg is None:
                rospy.logwarn_throttle(
                    5.0, "no camera frame cached yet (rgb=%s depth=%s)",
                    rgb_msg is not None, depth_msg is not None,
                )
                return [], None

            skew = abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec())
            if skew > self.max_rgb_depth_skew:
                rospy.logwarn_throttle(
                    5.0,
                    "RGB/depth skew %.3f s exceeds %.3f s; depth lookup may "
                    "not describe the same instant as the detection",
                    skew, self.max_rgb_depth_skew,
                )

            try:
                bgr = imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                depth = imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            except Exception as e:  # noqa: BLE001
                rospy.logerr("frame decode failed: %s", e)
                return [], None
            stamp = rgb_msg.header.stamp

            detections = self.detector(bgr)

            # Fine-grained classification for `button` detections —
            # one batched forward pass, regardless of how many buttons.
            if self.button_classifier is not None and detections:
                h, w = bgr.shape[:2]
                pad = float(self.button_pad_frac)
                crops, idxs = [], []
                for i, det in enumerate(detections):
                    if det["label"] != "button":
                        continue
                    bx1, by1, bx2, by2 = det["bbox"]
                    bw = bx2 - bx1
                    bh = by2 - by1
                    bx1 -= bw * pad
                    by1 -= bh * pad
                    bx2 += bw * pad
                    by2 += bh * pad
                    x1c = max(0, int(round(bx1)))
                    y1c = max(0, int(round(by1)))
                    x2c = min(w, int(round(bx2)))
                    y2c = min(h, int(round(by2)))
                    if x2c > x1c and y2c > y1c:
                        crops.append(bgr[y1c:y2c, x1c:x2c])
                        idxs.append(i)
                if crops:
                    preds = self.button_classifier.predict_batch(crops)
                    for i, (name, prob) in zip(idxs, preds):
                        detections[i]["button_class"] = name
                        detections[i]["button_class_score"] = prob
                        if prob >= self.button_classifier_conf:
                            detections[i]["label"] = name

            header = Header(stamp=stamp, frame_id=self.base_frame)
            pose_array = PoseArray(header=header)
            records = []

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))
                z = self._sample_depth(depth, u, v)
                if not math.isfinite(z):
                    rec = {
                        "label": det["label"],
                        "score": det["score"],
                        "uv": [u, v],
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "valid_depth": False,
                    }
                    if "button_class" in det:
                        rec["button_class"] = det["button_class"]
                        rec["button_class_score"] = det["button_class_score"]
                    records.append(rec)
                    continue

                p_surface_opt = self._pixel_to_camera(u, v, z, K)
                label = det["label"]
                radius_m = float(self.class_radius_m.get(label,
                                                         self.bottle_radius_m))
                p_opt = self._surface_to_object_center(p_surface_opt, radius_m)
                p_base = self._camera_to_base(p_opt)

                pose = Pose()
                pose.position = Point(float(p_base[0]), float(p_base[1]),
                                      float(p_base[2]))
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)

                rec = {
                    "label": label,
                    "score": det["score"],
                    "uv": [u, v],
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "valid_depth": True,
                    "radius_m": radius_m,
                    "position_camera_optical": [float(p_opt[0]),
                                                float(p_opt[1]),
                                                float(p_opt[2])],
                    "surface_camera_optical": [float(p_surface_opt[0]),
                                               float(p_surface_opt[1]),
                                               float(p_surface_opt[2])],
                    "position_base": [float(p_base[0]), float(p_base[1]),
                                      float(p_base[2])],
                }
                if "button_class" in det:
                    rec["button_class"] = det["button_class"]
                    rec["button_class_score"] = det["button_class_score"]
                records.append(rec)

            self.pub_pose.publish(pose_array)
            self.pub_json.publish(String(data=json.dumps({
                "stamp": stamp.to_sec() if stamp is not None else 0.0,
                "frame_id": self.base_frame,
                "detections": records,
            })))

            if self.pub_img is not None:
                self._publish_debug_image(bgr, records, stamp)

            with self._result_lock:
                self._latest_records = records
                self._latest_stamp = stamp

            return records, stamp

    def _fresh_records(self) -> Tuple[List[dict], Optional[rospy.Time], str]:
        """Newest inference result if it is inside ~max_frame_age_sec.

        Returns (records, stamp, reason). reason is "" when usable, otherwise
        why it was refused -- surfaced in the service message so a stale or
        absent camera is diagnosable from the service response alone."""
        with self._result_lock:
            records, stamp = self._latest_records, self._latest_stamp
        if stamp is None:
            return [], None, "no inference result yet"
        age = (rospy.Time.now() - stamp).to_sec()
        if age > self.max_frame_age:
            return [], stamp, (
                f"last camera frame is {age:.1f}s old "
                f"(max {self.max_frame_age:.1f}s) - camera stalled?"
            )
        return records, stamp, ""

    def _handle_get_object_position(self, request):
        response = GetObjectPositionResponse()
        requested_label = request.label.strip()
        requested_index = int(request.index)

        frame_id = self.base_frame

        # Prefer the newest background result: it is already computed, so the
        # call returns immediately instead of blocking a caller (grasp_bottle)
        # for an inference pass. Fall back to running one inline when the
        # timer is off (~inference_rate_hz=0) or its result has aged out --
        # the latter also covers a camera that just came back.
        records, stamp, stale_reason = self._fresh_records()
        if stale_reason:
            records, stamp = self._run_inference(
                wait_timeout=self.service_frame_timeout
            )
            if records or stamp is not None:
                stale_reason = ""

        if not records:
            response.success = False
            response.frame_id = frame_id
            response.message = (
                f"No detection result available ({stale_reason})"
                if stale_reason else
                "No detection result available "
                "(camera frame not yet received)"
            )
            return response

        valid_records = [r for r in records if r.get("valid_depth")]
        if requested_label:
            valid_records = [r for r in valid_records
                             if r.get("label") == requested_label]

        if not valid_records:
            response.success = False
            response.frame_id = frame_id
            if requested_label:
                response.message = f"No valid detection found for label '{requested_label}'"
            else:
                response.message = "No valid detection found"
            return response

        # Rank valid detections per the configured ~sort_by:
        #   z_desc     -> upper one first  (default; what "press upper button" needs)
        #   z_asc      -> lower one first
        #   score_desc -> highest-confidence first (legacy)
        # All sorts are tie-broken by descending score so reruns are stable
        # when two records share a z bucket (e.g., perfectly aligned buttons).
        if self.sort_by == "z_asc":
            valid_records.sort(key=lambda r: (
                float(r["position_base"][2]), -float(r.get("score", 0.0))
            ))
        elif self.sort_by == "score_desc":
            valid_records.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        else:  # z_desc (default)
            valid_records.sort(key=lambda r: (
                -float(r["position_base"][2]), -float(r.get("score", 0.0))
            ))

        if requested_index < 0 or requested_index >= len(valid_records):
            response.success = False
            response.frame_id = frame_id
            response.message = (
                f"Requested index {requested_index} out of range; "
                f"available valid detections: {len(valid_records)}"
            )
            return response

        record = valid_records[requested_index]
        px, py, pz = record["position_base"]
        response.success = True
        response.x = float(px)
        response.y = float(py)
        response.z = float(pz)
        response.frame_id = frame_id
        response.message = (
            f"Selected detection label={record['label']} "
            f"score={record['score']:.3f} z={record['position_base'][2]:.3f} "
            f"radius={record.get('radius_m', 0.0):.3f} "
            f"index={requested_index} sort_by={self.sort_by}"
        )
        return response

    # -----------------------------------------------------------------------
    def _publish_debug_image(self, bgr: np.ndarray, records, stamp):
        vis = bgr.copy()
        for r in records:
            x1, y1, x2, y2 = (int(v) for v in r["bbox"])
            color = (0, 200, 0) if r["valid_depth"] else (80, 80, 200)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cls_suffix = ""
            if "button_class" in r:
                cls_suffix = f' / {r["button_class"]} {r["button_class_score"]:.2f}'
            if r["valid_depth"]:
                x, y, z = r["position_base"]
                txt = f'{r["label"]} {r["score"]:.2f}{cls_suffix} ' \
                      f'b=({x:+.2f},{y:+.2f},{z:+.2f})m'
            else:
                txt = f'{r["label"]} {r["score"]:.2f}{cls_suffix} (no depth)'
            cv2.putText(vis, txt, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        msg = cv2_to_imgmsg(vis, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.optical_frame
        self.pub_img.publish(msg)


def main():
    try:
        ObjectDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:  # noqa: BLE001
        rospy.logfatal("object_detector_node crashed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
