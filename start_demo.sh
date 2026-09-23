#!/usr/bin/env bash
# One-shot demo bring-up from this workstation (PC1).
#
# The D435i camera and both Revo2 hands are wired to G1 PC2, not to PC1, so they
# are started remotely over SSH by their own package scripts
# (`g1_camera/scripts/start_remote_camera.sh`,
# `g1_hands/scripts/start_remote_hands.sh`). Those scripts own the PC2
# connection details (`G1_PC2_USER` / `G1_PC2_IP` / `G1_PC2_PASS`) and supply
# the PC2 password themselves, so nothing here asks for a password.
#
# PC1 keeps the arm node: it drives the arms over Unitree DDS on
# G1_NETWORK_INTERFACE and owns the voxel A* bitmaps, so it must run where the
# weights are. This project has no main_process, so the last terminal launches
# grasp_demo.launch with the camera/hands sub-systems switched off - they are
# already up on PC2 and re-launching them here would fight for the node names.
#
# The launches started on PC2 come from PC2's own g1_intellect checkout
# (g1_perception/realsense_d435i.launch, g1_hands/revo2_dual_hand_node.launch);
# see docs/REMOTE_BRINGUP.md.

set -euo pipefail

# The remote scripts resolve the ROS master IP from this interface; fail early
# and loudly rather than inside a spawned terminal.
: "${G1_NETWORK_INTERFACE:?set in ~/.bashrc}"

# Same reasoning: all three remote scripts hard-require sshpass, and a missing
# binary would otherwise only surface inside REMOTE_CAMERA/REMOTE_HANDS after
# roscore is already up and the old master has been killed.
command -v sshpass >/dev/null 2>&1 || {
  echo "ERROR: sshpass not found (needed to reach PC2). sudo apt-get install -y sshpass" >&2
  exit 1
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$REPO_DIR/catkin_ws"

killall -9 roscore rosmaster roslaunch rviz 2>/dev/null || true

# `killall` above only catches master/launcher/rviz. ROS node processes
# survive and auto-reconnect to the new master via rospy retry, then block
# fresh spawns from claiming their names. Kill those orphans too.
pkill -9 -f "catkin_ws/devel/lib/" 2>/dev/null || true
pkill -9 -f "g1_grasp_ros/.*\.py" 2>/dev/null || true
pkill -9 -f "/opt/ros/noetic/lib/robot_state_publisher" 2>/dev/null || true
pkill -9 -f "/opt/ros/noetic/lib/tf2_ros/static_transform_publisher" 2>/dev/null || true
pkill -9 -f "/opt/ros/noetic/lib/rosout/rosout" 2>/dev/null || true
sleep 1

gnome-terminal --title="ROSCORE" -- bash -ic '
echo "[Terminal 1] starting roscore..."
roscore
exec bash
'

sleep 3

# Clock alignment is OFF by default in this project. g1_intellect ran it here
# because rgbd_sync's 50 ms window makes SLAM/relocalization sensitive to PC1
# <-> PC2 skew; this repo has no SLAM, and the grasp path uses each frame on its
# own (object_detector_node back-projects the aligned depth pixel at the bbox
# centre, and the voxel map is rebuilt per plan), so it does not pair frames
# across the two clocks.
#
# Opt in when you want it - it is blocking, idempotent and PC2-side only, but it
# needs an NTP server running on PC1:
#     G1_SYNC_TIME=1 ./start_demo.sh
if [[ "${G1_SYNC_TIME:-0}" == "1" ]]; then
  echo "[Setup] aligning PC1<->PC2 clock (rosrun g1_camera setup_remote_timesync.sh)..."
  set +u
  source /opt/ros/noetic/setup.bash
  source "$WS_DIR/devel/setup.bash"
  set -u
  rosrun g1_camera setup_remote_timesync.sh
else
  echo "[Setup] skipping clock sync (G1_SYNC_TIME=1 to enable)"
fi

gnome-terminal --title="REMOTE_CAMERA" -- bash -ic '
echo "[Terminal 2] waiting for local roscore..."
until rostopic list >/dev/null 2>&1; do
  sleep 1
done

echo "[Terminal 2] starting remote realsense on G1 PC2..."
rosrun g1_camera start_remote_camera.sh
exec bash
'

sleep 3

gnome-terminal --title="REMOTE_HANDS" -- bash -ic '
echo "[Terminal 3] waiting for local roscore..."
until rostopic list >/dev/null 2>&1; do
  sleep 1
done

echo "[Terminal 3] starting remote g1_hands on G1 PC2..."
rosrun g1_hands start_remote_hands.sh
exec bash
'

sleep 3

gnome-terminal --title="OBJECT_DETECTOR" -- bash -ic '
echo "[Terminal 4] waiting for /camera/color/image_raw from PC2..."
until rostopic info /camera/color/image_raw >/dev/null 2>&1; do
  sleep 1
done

echo "[Terminal 4] launching object_detector (YOLO, PC1-side)..."
roslaunch g1_camera object_detector.launch
exec bash
'

sleep 3

gnome-terminal --title="BERRY_DETECTOR" -- bash -ic '
echo "[Terminal 5] waiting for /camera/color/image_raw from PC2..."
until rostopic info /camera/color/image_raw >/dev/null 2>&1; do
  sleep 1
done

echo "[Terminal 5] launching berry_detector (on-demand /perception/get_berries, PC1-side)..."
roslaunch g1_camera berry_detector.launch
exec bash
'

sleep 3

gnome-terminal --title="ARM" -- bash -ic '
echo "[Terminal 6] waiting for local roscore..."
until rostopic list >/dev/null 2>&1; do
  sleep 1
done

sleep 3
echo "[Terminal 6] launching arm node (camera/hands already up on PC2, detectors in their own terminals)..."
roslaunch g1_arm_abs grasp_demo.launch bringup_camera:=false bringup_hands:=false bringup_berry_detector:=false
exec bash
'

sleep 3

# Live detection view. object_detector_node republishes each processed frame
# with the YOLO boxes drawn on it (~publish_annotated, default true), so this
# shows exactly what the detector accepted - the same boxes that feed
# get_object_position. Nothing here runs YOLO a second time.
#
# Caveat worth knowing while debugging: the detector is on-demand. It only runs
# inference when grasp_bottle (or a manual get_object_position call) asks it to,
# so this window stays on the last processed frame between calls rather than
# streaming live. A static picture here does NOT mean the camera has stalled -
# check `rostopic hz /camera/color/image_raw` for that.
#
# Set G1_VIEW=0 to skip (e.g. over SSH with no X display).
if [[ "${G1_VIEW:-1}" == "1" ]] && [[ -n "${DISPLAY:-}" ]]; then
  gnome-terminal --title="DETECTION_VIEW" -- bash -ic '
  echo "[Terminal 7] waiting for /perception/annotated_image..."
  until rostopic info /perception/annotated_image >/dev/null 2>&1; do
    sleep 1
  done

  echo "[Terminal 7] opening rqt_image_view on the annotated detections..."
  rqt_image_view /perception/annotated_image
  exec bash
  '
else
  echo "[Setup] skipping detection view (G1_VIEW=0 or no DISPLAY)"
fi

cat <<'EOS'

Bring-up dispatched. When all seven terminals settle:

    rosservice call /g1_arm_simple/grasp_bottle     # grasp demo
    rosservice call /g1_arm_simple/pick_berry       # pick one berry
    rosservice call /g1_arm_simple/press_elevator_button \
        "{arm: right, x: 0.35, y: -0.1, z: 0.2, pre_press_dx: 0, wait: true, timeout: 0}"
    rosservice call /g1_arm_simple/GoHome           # release + both arms home

Terminal 7 (DETECTION_VIEW) shows what the detector accepted. It updates only
when inference runs - i.e. on grasp_bottle or a manual get_object_position
call - so a still image between calls is normal, not a stalled camera.
To see the raw camera stream instead:  rqt_image_view /camera/color/image_raw

Terminal 6 moves both arms to home on startup (~init_to_home_on_start).
Clear the space around the arms before it runs.
EOS
