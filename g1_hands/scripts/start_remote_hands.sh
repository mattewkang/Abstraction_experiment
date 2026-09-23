#!/usr/bin/env bash
# Start both Revo2 dexterous hands on G1 PC2 over SSH.
#
# Companion to g1_camera/scripts/start_remote_camera.sh and built to the same
# shape: PC2 connection details live here, the ROS master IP is resolved from
# G1_NETWORK_INTERFACE, and PC2 runs its own g1_intellect checkout, so the
# launch started there is that workspace's g1_hands/revo2_dual_hand_node.launch.
#
# The hands hang off PC2's USB/Modbus serial ports (ttyUSB0 left, ttyUSB1
# right), so they cannot be launched from PC1.
#
# Bring-up does NOT move the fingers: revo2_tactile_node's async_init only opens
# Modbus and starts a read-only tactile poll. The hands move solely on
# /g1_hand_{left,right}/{pre_grasp,grasp_5f,release} service calls.
#
# Idempotent: a stale hand node on PC2 keeps the Modbus port open and would make
# the fresh launch fail to claim the serial device, so it is killed first.
#
# Usage:
#   rosrun g1_hands start_remote_hands.sh
set -euo pipefail

# PC2 connection (override via environment, or ~/.g1_pc2_env).
[ -f "$HOME/.g1_pc2_env" ] && source "$HOME/.g1_pc2_env"
REMOTE_USER="${G1_PC2_USER:-unitree}"
REMOTE_IP="${G1_PC2_IP:-192.168.123.164}"
REMOTE_PASS="${G1_PC2_PASS:-123}"
REMOTE_WS="${G1_PC2_WS:-\$HOME/Workspace/g1_intellect/catkin_ws}"

# Serial ports on PC2. Swapping these swaps left and right hands, so they are
# pinned here rather than left to the launch file's defaults.
LEFT_PORT="${G1_LEFT_HAND_PORT:-/dev/ttyUSB0}"
RIGHT_PORT="${G1_RIGHT_HAND_PORT:-/dev/ttyUSB1}"
LEFT_SLAVE_ID="${G1_LEFT_HAND_SLAVE_ID:-126}"
RIGHT_SLAVE_ID="${G1_RIGHT_HAND_SLAVE_ID:-127}"

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

# A stale node holds the Modbus port open and blocks the new one.
echo "Stopping any existing hand nodes on PC2..."
sshpass -p "${REMOTE_PASS}" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${REMOTE_USER}@${REMOTE_IP}" \
  'pkill -f revo2_tactile_node; true' \
  >/dev/null 2>&1 || true
sleep 2

echo "Starting Revo2 hands on PC2 ${REMOTE_USER}@${REMOTE_IP}"
echo "  ROS master: http://${MASTER_IP}:11311  (via ${NIF})"
echo "  Ports: left ${LEFT_PORT} (id ${LEFT_SLAVE_ID}) / right ${RIGHT_PORT} (id ${RIGHT_SLAVE_ID})"

# MASTER_IP / REMOTE_IP / port vars expand locally (double quotes); \$HOME on PC2.
sshpass -p "${REMOTE_PASS}" ssh -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${REMOTE_USER}@${REMOTE_IP}" "bash -lc '
    source /opt/ros/noetic/setup.bash
    source ${REMOTE_WS}/devel/setup.bash
    export ROS_MASTER_URI=http://${MASTER_IP}:11311
    export ROS_IP=${REMOTE_IP}
    for p in ${LEFT_PORT} ${RIGHT_PORT}; do
      [ -e \"\$p\" ] || { echo \"ERROR: \$p missing - is that hand plugged in?\" >&2; exit 1; }
    done
    roslaunch g1_hands revo2_dual_hand_node.launch \
      left_port:=${LEFT_PORT}   left_slave_id:=${LEFT_SLAVE_ID} \
      right_port:=${RIGHT_PORT} right_slave_id:=${RIGHT_SLAVE_ID}
  '"
