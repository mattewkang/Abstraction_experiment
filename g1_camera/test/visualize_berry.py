#!/usr/bin/env python3
"""Standalone live visualizer for the berry detector + ripeness classifier,
with colour-aware depth back-projection to per-berry 3D positions.

Debug / preview tool only — NOT a ROS node in the production pipeline. It
exists purely to eyeball detection quality before the berry-picking pipeline
is wired up; berry_detector_node.py is the production on-demand node.

The depth -> position math mirrors berry_detector_node.py: a per-pixel temporal
median over a sliding window of depth frames (--depth-temporal-frames) to kill
single-frame RealSense noise, median depth over a patch at a sample pixel,
back-projection through the colour intrinsics K, a ray-direction shift by the
berry radius (front surface -> centre), then the analytic torso_link transform
composed from camera_translation / camera_rotation_ypr and the fixed optical
rotation. Extrinsic + depth parameters are loaded from config/params.yaml so
this tool stays in sync with the production node; CLI flags override values.

Colour-aware depth alignment (--no-color-align to disable):
    A leaf between the berry and the camera puts the bbox centre on foreground
    foliage, so its depth is the leaf's, not the berry's. Two cases (mirroring
    the node):
      1. The bbox centre matches the predicted ripeness colour (a patch
         majority around the centre, --color-center-frac) -> sample at the
         centre directly.
      2. Otherwise (leaf occlusion) or no centre depth -> sample at the centre
         of the LARGEST connected colour-matching region in the bbox, and the
         drawn sample point hops there (magenta dot). Needs the classifier (the
         label supplies the expected colour).

Pipeline per frame:
    unified_detector_yolo11s.pt        -> boxes (button / bottle / berry)
    crop each box -> unified_classifier_yolo11s_cls.pt -> {black, red, white}
    colour-aware depth sample -> 3D point in torso_link
    draw box + "<ripeness> det=<conf> cls=<prob> b=(x,y,z)m".

Run (camera must already be publishing — see g1_perception realsense_d435i.launch):
    conda run -n g1_python_310 python g1_camera/test/visualize_berry.py

Keys:  q / ESC = quit,  s = save current annotated frame to /tmp.
"""

import argparse
import math
import os
import threading
import warnings
from collections import deque

import cv2
import numpy as np
import yaml

import rospy
from sensor_msgs.msg import CameraInfo, Image

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARAMS_PATH = os.path.join(_PKG_DIR, "config", "params.yaml")

# Ripeness classes that have a colour gate for depth re-alignment.
_COLOR_CLASSES = ("red", "black", "white")

# Per-class draw colour (BGR) keyed by ripeness label.
_CLS_COLOR = {
    "red":   (60, 60, 220),
    "black": (60, 60, 60),
    "white": (230, 230, 230),
}
_DEFAULT_COLOR = (0, 200, 0)
_REALIGN_COLOR = (255, 0, 255)  # magenta dot when depth hopped to a colour blob


# Minimal imgmsg->cv2: Noetic's apt cv_bridge is built against numpy 1.x and
# crashes under the workspace's numpy 2.x, so decode the raw buffer ourselves.
_ENC = {
    "bgr8":  (np.uint8, 3),
    "rgb8":  (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
}


def imgmsg_to_array(msg: Image, want_bgr: bool = False) -> np.ndarray:
    enc = msg.encoding
    if enc not in _ENC:
        raise ValueError(f"Unsupported encoding: {enc}")
    dtype, ch = _ENC[enc]
    buf = np.frombuffer(msg.data, dtype=dtype)
    img = buf.reshape(msg.height, msg.width, ch) if ch > 1 \
        else buf.reshape(msg.height, msg.width)
    if want_bgr and enc == "rgb8":
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def temporal_median_depth(frames):
    """Per-pixel median across the depth stack, ignoring zero/no-data, in the
    raw depth units (depth_scale still applies downstream). Holes become NaN
    (NaN > 0 is False, so the samplers treat them as invalid). Mirrors
    berry_detector_node._temporal_median_depth; collapses single-frame
    RealSense depth noise so a stationary berry's z stops wobbling."""
    if len(frames) == 1:
        d = frames[0].astype(np.float32)
        d[d <= 0] = np.nan
        return d
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    return med.astype(np.float32)


def berry_color_mask(region_bgr: np.ndarray, label: str) -> np.ndarray:
    """Boolean mask of pixels in `region_bgr` matching the ripeness `label`.

    Heuristic HSV gates (OpenCV HSV: H 0-179, S/V 0-255). Tune against live
    frames — these are deliberately loose so a berry surface is not missed,
    at the cost of occasionally accepting similarly-coloured background.
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


# --- geometry helpers (ported verbatim from object_detector_node.py) -------
def euler_ypr_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Right-handed, ROS convention."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


# Fixed camera_color_optical_frame -> camera_link rotation, matching
# g1_slam/launch/transforms.launch (yaw=-1.5708 pitch=0 roll=-1.5708).
R_OPTICAL_TO_LINK = euler_ypr_to_matrix(-1.5708, 0.0, -1.5708)


def _load_params() -> dict:
    try:
        with open(_PARAMS_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        rospy.logwarn("params.yaml not found at %s; using built-in defaults.",
                      _PARAMS_PATH)
        return {}


class BerryLocalizer:
    """Holds extrinsic + depth params and turns a sample pixel into a torso_link
    position, mirroring object_detector_node.py's depth pipeline. A debug-only
    constant xyz offset (subtracted from the result) lets the operator null a
    residual calibration error without touching the shared params.yaml."""

    def __init__(self, args, params: dict):
        self.depth_scale = float(args.depth_scale
                                 if args.depth_scale is not None
                                 else params.get("depth_scale", 0.001))
        self.min_depth = float(params.get("min_depth", 0.15))
        self.max_depth = float(params.get("max_depth", 5.0))
        self.patch_r = int(params.get("depth_patch_radius", 3))
        # Berry is a round object: shift the deprojected front surface along the
        # camera ray by its radius to reach the centre. Default 0.005 m = 1 cm
        # diameter (params.yaml has no `berry` entry; the production node would
        # fall back to the much larger bottle_radius_m).
        self.berry_radius_m = float(args.berry_radius)
        # Debug residual trim, SUBTRACTED from the base-frame position.
        self.pos_offset = np.array([args.x_offset, args.y_offset,
                                    args.z_offset], dtype=np.float64)

        t = params.get("camera_translation", [0.0576235, 0.02753, 0.44487])
        r = params.get("camera_rotation_ypr", [0.0, 0.8307767239493009, 0.0])
        self.t_base_opt = np.asarray(t, dtype=np.float64).reshape(3)
        R_base_cam = euler_ypr_to_matrix(float(r[0]), float(r[1]), float(r[2]))
        self.R_base_opt = R_base_cam @ R_OPTICAL_TO_LINK

        self.base_frame = params.get("base_frame", "torso_link")

        # Colour-aware re-alignment thresholds (mirror berry_detector_node).
        self.color_min_pixels = int(args.color_min_pixels)
        self.color_center_frac = float(args.color_center_frac)

    def sample_depth(self, depth: np.ndarray, u: int, v: int) -> float:
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

    def sample_depth_masked(self, depth: np.ndarray, x0: int, y0: int,
                            x1: int, y1: int, mask: np.ndarray):
        """Largest connected colour-matching region inside the bbox (the visible
        berry surface): median depth (m) over that region's valid-depth pixels,
        plus the full-frame (u,v) of its centre. Mirrors
        berry_detector_node._sample_depth_masked. Using the largest connected
        region rather than the mean of every matching pixel keeps the sample on
        the actual berry when HSV noise or a neighbouring berry clips a bbox
        edge. Returns (nan, None, None) when no region clears color_min_pixels."""
        sub = depth[y0:y1, x0:x1].astype(np.float32) * self.depth_scale
        depth_ok = (sub >= self.min_depth) & (sub <= self.max_depth)
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
        cu, cv = centroids[best_lbl]  # (x, y) in sub-image coords
        ys, xs = np.nonzero(blob)
        j = int(np.argmin((xs - cu) ** 2 + (ys - cv) ** 2))
        return z, x0 + int(xs[j]), y0 + int(ys[j])

    def pixel_to_base(self, u: float, v: float, z: float,
                      K: np.ndarray) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic",
                    default="/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--detector",
                    default=os.path.join(_PKG_DIR, "models",
                                         "unified_detector_yolo11s.pt"))
    ap.add_argument("--classifier",
                    default=os.path.join(_PKG_DIR, "models",
                                         "unified_classifier_yolo11s_cls.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detector confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--pad", type=float, default=0.10,
                    help="bbox padding fraction before classification")
    ap.add_argument("--berry-radius", type=float, default=0.005,
                    help="ray-direction shift (m) from front surface to berry "
                         "centre; default 0.005 = 1 cm diameter berry. 0 keeps "
                         "the front surface.")
    ap.add_argument("--x-offset", type=float, default=0.02,
                    help="debug: metres SUBTRACTED from base-frame x to null a "
                         "residual calibration error (does not touch "
                         "params.yaml). Default 0.02 corrects the observed "
                         "+2 cm x bias.")
    ap.add_argument("--y-offset", type=float, default=0.0,
                    help="debug: metres subtracted from base-frame y")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="debug: metres subtracted from base-frame z")
    ap.add_argument("--depth-scale", type=float, default=None,
                    help="override params.yaml depth_scale (0.001 for 16UC1 mm)")
    ap.add_argument("--color-min-pixels", type=int, default=15,
                    help="min valid-depth pixels in the largest colour-matching "
                         "region to trust a re-aligned depth")
    ap.add_argument("--color-center-frac", type=float, default=0.5,
                    help="centre-patch colour fraction for the 'centre on berry' "
                         "test (case 1: align directly to the centre)")
    ap.add_argument("--depth-temporal-frames", type=int, default=5,
                    help="depth frames median-stacked (sliding window) to kill "
                         "single-frame jitter; 1 = single frame (legacy)")
    ap.add_argument("--no-color-align", action="store_true",
                    help="disable colour-aware depth re-alignment")
    ap.add_argument("--no-classify", action="store_true",
                    help="draw detector boxes only, skip ripeness classifier")
    ap.add_argument("--no-depth", action="store_true",
                    help="skip depth/position, draw 2D detection only")
    args = ap.parse_args()

    params = _load_params()
    loc = None if args.no_depth else BerryLocalizer(args, params)

    from ultralytics import YOLO
    rospy.loginfo("loading detector: %s", args.detector)
    detector = YOLO(args.detector)
    classifier = None
    if not args.no_classify:
        rospy.loginfo("loading classifier: %s", args.classifier)
        classifier = YOLO(args.classifier)

    # Colour re-alignment needs the classifier's label to know the expected
    # colour; silently a no-op without it.
    do_color_align = (classifier is not None) and (not args.no_color_align)

    win = "berry detection (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {"frame": None, "n": 0}
    shared = {"depth": None, "K": None}
    lock = threading.Lock()

    def on_info(msg: CameraInfo):
        with lock:
            shared["K"] = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    # Rolling depth buffer for the temporal median (mirrors the node's per-call
    # depth-stack median; here it is a sliding window over the live stream).
    depth_buf = deque(maxlen=max(1, args.depth_temporal_frames))

    def on_depth(msg: Image):
        try:
            d = imgmsg_to_array(msg)
        except Exception as e:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "depth decode failed: %s", e)
            return
        with lock:
            depth_buf.append(d)
            shared["depth"] = temporal_median_depth(list(depth_buf))

    def on_image(msg: Image):
        try:
            bgr = imgmsg_to_array(msg, want_bgr=True)
        except Exception as e:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "decode failed: %s", e)
            return

        with lock:
            depth = shared["depth"]
            K = None if shared["K"] is None else shared["K"].copy()

        res = detector.predict(bgr, conf=args.conf, iou=args.iou,
                               device=args.device, verbose=False)[0]
        vis = bgr.copy()
        h, w = bgr.shape[:2]
        boxes = res.boxes
        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            det_conf = float(boxes.conf[i])
            label, color = "berry", _DEFAULT_COLOR
            cls_txt = ""

            if classifier is not None:
                pad_x = (x2 - x1) * args.pad
                pad_y = (y2 - y1) * args.pad
                cx1 = max(0, int(round(x1 - pad_x)))
                cy1 = max(0, int(round(y1 - pad_y)))
                cx2 = min(w, int(round(x2 + pad_x)))
                cy2 = min(h, int(round(y2 + pad_y)))
                crop = bgr[cy1:cy2, cx1:cx2]
                if crop.size:
                    cres = classifier.predict(crop, device=args.device,
                                              verbose=False)[0]
                    top = int(cres.probs.top1)
                    prob = float(cres.probs.top1conf)
                    label = cres.names[top]
                    color = _CLS_COLOR.get(label, _DEFAULT_COLOR)
                    cls_txt = f" cls={prob:.2f}"

            # --- colour-aware depth -> torso_link position -------------------
            pos_txt = ""
            if loc is not None and depth is not None and K is not None:
                u = int(round((x1 + x2) * 0.5))
                v = int(round((y1 + y2) * 0.5))
                realigned = False
                z = loc.sample_depth(depth, u, v)  # default: bbox centre

                if do_color_align and label in _COLOR_CLASSES:
                    bx0, by0 = max(0, int(round(x1))), max(0, int(round(y1)))
                    bx1, by1 = min(w, int(round(x2))), min(h, int(round(y2)))
                    if bx1 > bx0 and by1 > by0:
                        mask = berry_color_mask(bgr[by0:by1, bx0:bx1], label)
                        mu, mv = u - bx0, v - by0
                        # Patch majority around the centre (robust to a single
                        # noisy/specular pixel), sized to the depth patch.
                        pr = loc.patch_r
                        wy0, wy1 = max(0, mv - pr), min(mask.shape[0], mv + pr + 1)
                        wx0, wx1 = max(0, mu - pr), min(mask.shape[1], mu + pr + 1)
                        win = mask[wy0:wy1, wx0:wx1]
                        center_on_color = (
                            win.size > 0
                            and float(win.mean()) >= loc.color_center_frac)
                        # Re-align when the centre is off the berry colour OR
                        # the centre depth is unusable (likely a foreground
                        # leaf / depth hole). The largest-region gate lives in
                        # sample_depth_masked.
                        if not center_on_color or not math.isfinite(z):
                            zc, uc, vc = loc.sample_depth_masked(
                                depth, bx0, by0, bx1, by1, mask)
                            if math.isfinite(zc):
                                z, u, v, realigned = zc, uc, vc, True

                if math.isfinite(z):
                    pb = loc.pixel_to_base(u, v, z, K)
                    pos_txt = (f" b=({pb[0]:+.2f},{pb[1]:+.2f},{pb[2]:+.2f})m")
                else:
                    pos_txt = " (no depth)"
                cv2.circle(vis, (u, v), 3,
                           _REALIGN_COLOR if realigned else color, -1)

            ix1, iy1, ix2, iy2 = (int(round(c)) for c in (x1, y1, x2, y2))
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), color, 2)
            txt = f"{label} det={det_conf:.2f}{cls_txt}{pos_txt}"
            cv2.putText(vis, txt, (ix1, max(12, iy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        status = f"berries: {n}"
        if loc is not None and (depth is None or K is None):
            status += "  [waiting depth/info]"
        cv2.putText(vis, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        state["frame"] = vis

    rospy.init_node("visualize_berry", anonymous=True)
    rospy.Subscriber(args.topic, Image, on_image, queue_size=1,
                     buff_size=2 ** 24)
    if loc is not None:
        rospy.Subscriber(args.depth_topic, Image, on_depth, queue_size=1,
                         buff_size=2 ** 24)
        rospy.Subscriber(args.info_topic, CameraInfo, on_info, queue_size=1)
        rospy.loginfo(
            "depth ON: scale=%.4f patch_r=%d berry_radius=%.4fm "
            "offset(x,y,z)=(%.3f,%.3f,%.3f) color_align=%s base_frame=%s",
            loc.depth_scale, loc.patch_r, loc.berry_radius_m,
            loc.pos_offset[0], loc.pos_offset[1], loc.pos_offset[2],
            do_color_align, loc.base_frame)
    rospy.loginfo("subscribed to %s; waiting for frames...", args.topic)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if state["frame"] is not None:
            cv2.imshow(win, state["frame"])
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and state["frame"] is not None:
            path = f"/tmp/berry_vis_{state['n']:04d}.png"
            cv2.imwrite(path, state["frame"])
            rospy.loginfo("saved %s", path)
            state["n"] += 1
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
