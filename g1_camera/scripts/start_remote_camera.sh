#!/usr/bin/env bash
# Start the RealSense D435i ROS driver on G1 PC2 over SSH (remote_camera:=true).
#
# Adapted from g1_intellect's g1_perception/scripts/start_remote_camera.sh. This
# trimmed repo has no g1_perception, so the script lives in g1_camera and is
# invoked as `rosrun g1_camera start_remote_camera.sh`. The PC2 side is
# unchanged: PC2 runs its own g1_intellect checkout, so the launch file started
# there is still g1_perception/realsense_d435i.launch.
#
# Camera only: this script does NOT touch PC2's clock. Time synchronization is a
# separate, once-per-boot step (scripts/setup_remote_timesync.sh), and in this
# repo it is OPTIONAL and off by default - there is no SLAM here, and the grasp
# path uses each frame on its own rather than pairing frames across the PC1/PC2
# clocks. Enable it with `G1_SYNC_TIME=1 ./start_demo.sh` if you need it.
# The ROS master IP is resolved from the G1_NETWORK_INTERFACE environment
# variable so the command is host-agnostic. PC2 connection details are
# overridable via environment variables.
#
# Idempotent: any RealSense already running on PC2 is stopped first, so the fresh
# launch grabs the USB device cleanly and anchors its frame timestamps to the
# current (chrony-synced) PC2 clock. A camera process that predates a clock step
# keeps emitting stamps on the old clock, which the SLAM-side clock-skew
# diagnostic flags; restarting SLAM alone never clears that, restarting the
# camera does.
#
# Usage:
#   rosrun g1_camera start_remote_camera.sh
set -euo pipefail

# PC2 connection (override via environment, or ~/.g1_pc2_env).
[ -f "$HOME/.g1_pc2_env" ] && source "$HOME/.g1_pc2_env"
REMOTE_USER="${G1_PC2_USER:-unitree}"
REMOTE_IP="${G1_PC2_IP:-192.168.123.164}"
REMOTE_PASS="${G1_PC2_PASS:-123}"
# PC2 keeps the full g1_intellect checkout; that workspace supplies both
# g1_perception (this launch) and g1_hands (start_remote_hands.sh).
REMOTE_WS="${G1_PC2_WS:-\$HOME/Workspace/g1_intellect/catkin_ws}"

# Stream config, passed to PC2's roslaunch as explicit args. Pinned here (not
# env-overridable) so this PC1 script is the single source of truth: PC2 runs
# its own checkout of realsense_d435i.launch, which may be stale, and explicit
# args override whatever defaults that copy carries. Keep these in sync with the
# launch file's arg defaults. Color runs 1280x720 for detail while depth stays
# 848x480; both at 15 fps. NOTE: visual relocalization (resolution-dependent
# bag-of-words) wants the color resolution to match the rtabmap map-build
# resolution, so a map recorded at 848x480 relocalizes worse against 1280x720
# color until it is rebuilt at this resolution. CAM_RESET resets the device on
# startup to clear the recurring D435i USB stall (errno 11 / uvc watchdog on
# EP 130 / depth stream start failure); costs ~3 s.
#
# g1_grasp_ros note: object_detector_node reads intrinsics from
# /camera/color/camera_info on every frame, so the 1280x720 color stream needs
# no retuning here. It does require the aligned depth topic
# (/camera/aligned_depth_to_color/image_raw); realsense_d435i.launch is expected
# to enable align_depth itself, as it does in g1_intellect.
CAM_COLOR_W=1280
CAM_COLOR_H=720
CAM_DEPTH_W=848
CAM_DEPTH_H=480
CAM_FPS=15
CAM_RESET=true

# Resolve the ROS master IP from G1_NETWORK_INTERFACE.
NIF="${G1_NETWORK_INTERFACE:-}"
if [[ -z "$NIF" ]]; then
  echo "ERROR: G1_NETWORK_INTERFACE is not set. Export it in ~/.bashrc, e.g." >&2
  echo "       export G1_NETWORK_INTERFACE=ens33" >&2
  exit 1
fi
MASTER_IP="$(ip -4 -o addr show dev "$NIF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
if [[ -z "$MASTER_IP" ]]; then
  echo "ERROR: no IPv4 address on interface '$NIF'." >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "ERROR: sshpass not found. Install it: sudo apt-get install -y sshpass" >&2
  exit 1
fi

# Stop any RealSense already running on PC2 so the new launch grabs the USB
# device cleanly and anchors its timestamps to the current PC2 clock. The
# trailing `true` keeps the remote command zero-exit when nothing matched.
echo "Stopping any existing RealSense on PC2..."
sshpass -p "${REMOTE_PASS}" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${REMOTE_USER}@${REMOTE_IP}" \
  'pkill -f realsense2_camera; pkill -f rs_camera; pkill -f realsense_d435i; true' \
  >/dev/null 2>&1 || true
sleep 2

echo "Starting RealSense on PC2 ${REMOTE_USER}@${REMOTE_IP}"
echo "  ROS master: http://${MASTER_IP}:11311  (via ${NIF})"
echo "  Stream: color ${CAM_COLOR_W}x${CAM_COLOR_H} / depth ${CAM_DEPTH_W}x${CAM_DEPTH_H} @ ${CAM_FPS} fps, initial_reset=${CAM_RESET}"

# MASTER_IP / REMOTE_IP / CAM_* expand locally (double quotes); \$HOME on PC2.
sshpass -p "${REMOTE_PASS}" ssh -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${REMOTE_USER}@${REMOTE_IP}" "bash -lc '
    source /opt/ros/noetic/setup.bash
    source ${REMOTE_WS}/devel/setup.bash
    export ROS_MASTER_URI=http://${MASTER_IP}:11311
    export ROS_IP=${REMOTE_IP}
    roslaunch g1_perception realsense_d435i.launch \
      color_width:=${CAM_COLOR_W} color_height:=${CAM_COLOR_H} \
      depth_width:=${CAM_DEPTH_W} depth_height:=${CAM_DEPTH_H} \
      color_fps:=${CAM_FPS} depth_fps:=${CAM_FPS} \
      initial_reset:=${CAM_RESET}
  '"
