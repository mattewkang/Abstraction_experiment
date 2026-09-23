# PC1 / PC2 split bring-up

## Who runs what

| | PC1 (this workstation) | G1 PC2 |
| --- | --- | --- |
| roscore | yes (ROS master) | attaches to PC1 |
| RealSense D435i | - | wired here -> `start_remote_camera.sh` |
| Revo2 hands (ttyUSB0/1) | - | wired here -> `start_remote_hands.sh` |
| object_detector (YOLO) | yes, consumes PC2's topics | - |
| arm node + voxel A* | yes (DDS on `G1_NETWORK_INTERFACE`, owns the ~28 GB bitmaps) | - |
| chrony | must **serve** NTP | disciplined to PC1 |

The arm node stays on PC1 because it needs the voxel A* bitmaps and the DDS NIC.
The camera and hands cannot move: they are physically wired to PC2.

## One-shot

```bash
source ~/g1_env.sh
./start_demo.sh
```

Opens six terminals: ROSCORE, REMOTE_CAMERA, REMOTE_HANDS, OBJECT_DETECTOR,
BERRY_DETECTOR, ARM (plus DETECTION_VIEW when a display is available).
Then:

```bash
rosservice call /g1_arm_simple/grasp_bottle
rosservice call /g1_arm_simple/GoHome
```

## What runs on PC2

PC2 keeps its own full **g1_intellect** checkout at
`$HOME/Workspace/g1_intellect/catkin_ws`, and the remote scripts source that
workspace. So the launch files started remotely are PC2's, not this repo's:

| PC1 script | starts on PC2 |
| --- | --- |
| `rosrun g1_camera start_remote_camera.sh` | `g1_perception realsense_d435i.launch` |
| `rosrun g1_hands start_remote_hands.sh` | `g1_hands revo2_dual_hand_node.launch` |

This is why the camera script names `g1_perception` even though this repo has no
such package: it is a PC2-side package name. Stream geometry (color 1280x720,
depth 848x480, 15 fps, `initial_reset:=true`) is pinned in the PC1 script and
passed as explicit roslaunch args, so a stale PC2 checkout cannot change it.

`object_detector_node` reads intrinsics from `/camera/color/camera_info` on
every frame, so the 1280x720 color stream needs no retuning on this side. It
does require `/camera/aligned_depth_to_color/image_raw`, which
`realsense_d435i.launch` is expected to publish.

## Credentials

Defaults are inline in each script — stock Unitree `unitree` @
`192.168.123.164`, password `123` — so a clean checkout runs with no setup.
Override in `~/.g1_pc2_env` (chmod 600, never committed) only when PC2 differs:

```bash
export G1_PC2_USER=unitree
export G1_PC2_IP=192.168.123.164
export G1_PC2_PASS=''    # empty or unset -> falls back to the stock password
# export G1_PC2_WS='/home/unitree/Workspace/g1_intellect/catkin_ws'
```

`G1_PC2_WS` must be a path valid **on PC2**. Setting a PC1 path such as
`~/catkin_ws` silently breaks the remote launch.

All three remote scripts require `sshpass`:

```bash
sudo apt install sshpass
```

Change the stock password and you should switch to keys instead:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id $G1_PC2_USER@$G1_PC2_IP
```

## Clock alignment (optional, OFF by default)

`start_demo.sh` skips it. Enable with:

```bash
G1_SYNC_TIME=1 ./start_demo.sh
```

g1_intellect ran this on every bring-up because `rgbd_sync`'s 50 ms window makes
SLAM and visual relocalization sensitive to PC1/PC2 skew. **This repo has no
SLAM**, and the grasp path never pairs frames across the two clocks:
`object_detector_node` back-projects the aligned depth pixel at the bbox centre
from a single frame, and the voxel obstacle map is rebuilt per plan. So skew
costs nothing here.

It is still worth enabling if you add SLAM, record bags for offline replay, or
correlate PC1 and PC2 logs by timestamp.

PC1 must **serve** NTP for it to work (the script fails loudly if it does not):

```bash
sudo apt install chrony
echo "allow 192.168.123.0/24" | sudo tee -a /etc/chrony/chrony.conf
echo "local stratum 10"       | sudo tee -a /etc/chrony/chrony.conf
sudo systemctl restart chrony
```

`local stratum 10` lets PC1 serve even with no upstream internet clock, which is
the normal case on the robot subnet.

## Safety

- Hand bring-up does **not** move the fingers. `revo2_tactile_node.async_init`
  only opens Modbus and starts a read-only tactile poll; motion happens solely
  on `/g1_hand_*/{pre_grasp,grasp_5f,release}` calls.
- The **arm** node is different: `~init_to_home_on_start` drives both arms to
  home as soon as terminal 5 comes up. Clear the space around the arms first.
- Stop the arm node with Ctrl-C, never `kill -9`: shutdown ramps the motion-mode
  weight 1 -> 0 before releasing. A hard kill drops the arms.
