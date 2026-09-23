#!/usr/bin/env bash
# Discipline G1 PC2's clock to PC1 via chrony. Blocking, idempotent, PC2-side
# only. Once per boot, BEFORE start_remote_camera.sh.
#
# Adapted from g1_intellect's g1_perception/scripts/setup_remote_timesync.sh;
# lives in g1_camera here because this repo has no g1_perception. Invoked as
#   rosrun g1_camera setup_remote_timesync.sh
#
# Why it must run first: the D435i stamps frames with PC2's clock, and the
# depth/colour pairing downstream tolerates only tens of ms of skew. A skewed
# PC2 clock mispairs RGB-D frames, and object_detector_node's depth
# back-projection then reports the grasp target in the wrong place. Aligning at
# the OS level means nothing in ROS has to re-stamp. A camera process that
# predates a clock step keeps emitting old-clock stamps, which is why the camera
# is started only after this returns.
set -euo pipefail

# PC2 connection (override via environment, or ~/.g1_pc2_env).
[ -f "$HOME/.g1_pc2_env" ] && source "$HOME/.g1_pc2_env"
REMOTE_USER="${G1_PC2_USER:-unitree}"
REMOTE_IP="${G1_PC2_IP:-192.168.123.164}"
REMOTE_PASS="${G1_PC2_PASS:-123}"
MAX_OFFSET="${G1_MAX_CLOCK_OFFSET:-0.05}"   # seconds

# Resolve the ROS master / NTP server IP from G1_NETWORK_INTERFACE.
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

echo "Aligning PC2 (${REMOTE_IP}) to PC1 (${MASTER_IP}) via ${NIF}"

# PC2 syncs *to* PC1, so PC1 must serve NTP on the robot subnet. Without a
# listener the remote chrony silently never converges, so fail loudly here.
if ! ss -lun 2>/dev/null | grep -q ':123\b'; then
  cat >&2 <<'WARN'
ERROR: nothing is serving NTP on PC1 (no listener on UDP/123).
       PC2 cannot discipline its clock to a server that is not running.
       On PC1:
           sudo apt install chrony
           echo "allow 192.168.123.0/24" | sudo tee -a /etc/chrony/chrony.conf
           echo "local stratum 10"       | sudo tee -a /etc/chrony/chrony.conf
           sudo systemctl restart chrony
       ("local stratum 10" lets PC1 serve even with no upstream internet clock,
       which is the normal case on the robot subnet.)
WARN
  exit 1
fi

# MASTER_IP / MAX_OFFSET / REMOTE_PASS expand locally; the rest runs on PC2.
sshpass -p "${REMOTE_PASS}" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "${REMOTE_USER}@${REMOTE_IP}" \
  "MASTER_IP='${MASTER_IP}' MAX_OFFSET='${MAX_OFFSET}' SUDO_PASS='${REMOTE_PASS}' bash -s" <<'REMOTE'
set -eo pipefail

# -S feeds the sudo password on stdin; passwordless sudo also works.
sudo_run() {
  if [ -n "$SUDO_PASS" ]; then printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"
  else sudo -n "$@"; fi
}

command -v chronyc >/dev/null || {
  echo "ERROR: chrony not installed on PC2: sudo apt install chrony" >&2; exit 1; }

CONF=/etc/chrony/chrony.conf
if ! grep -qE "^server[[:space:]]+${MASTER_IP}\b" "$CONF" 2>/dev/null; then
  echo "Adding 'server $MASTER_IP iburst prefer' to $CONF"
  printf 'server %s iburst prefer\n' "$MASTER_IP" | sudo_run tee -a "$CONF" >/dev/null
  sudo_run systemctl restart chrony
  sleep 3
else
  echo "$CONF already points at $MASTER_IP"
fi

# Force an immediate step rather than waiting for chrony to slew.
sudo_run chronyc -a makestep >/dev/null 2>&1 || chronyc makestep >/dev/null 2>&1 || true

for _ in $(seq 1 20); do
  OFF="$(chronyc tracking 2>/dev/null | awk -F'[:[:space:]]+' '/System time/{print $4}')"
  [ -n "$OFF" ] || { sleep 1; continue; }
  if awk -v o="$OFF" -v m="$MAX_OFFSET" 'BEGIN{exit !(o<m)}'; then
    echo "PC2 clock aligned: system time offset ${OFF}s (< ${MAX_OFFSET}s)"
    exit 0
  fi
  sleep 1
done
echo "WARNING: PC2 offset still ${OFF:-unknown}s after 20s - RGB-D pairing may degrade" >&2
exit 0
REMOTE
