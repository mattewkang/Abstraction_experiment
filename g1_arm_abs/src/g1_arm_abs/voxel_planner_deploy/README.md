# Voxel-based Obstacle-Aware A* Planner (Unitree G1 arms)

Drop-in replacement for a static reachable-map A* planner. Accepts **arbitrary
obstacles at query time** (AABBs or voxel indices), builds a dynamic reachable
mask from a precomputed per-voxel config bitmap, then runs A* on a 5D joint
grid (4°/step, 5 joints per arm).

---

## Package contents

```
g1_arm_abs/src/g1_arm_abs/voxel_planner_deploy/
├── README.md                            # this file
├── decompress_once.py                   # one-time decompress of voxel_config_bitmap_*.uint8.gz
├── voxel_phase3_query.py                # VoxelReachability: obstacle -> reachable mask
├── voxel_phase4_astar.py                # VoxelObstacleAStar: A* + compression + staged
└── voxel_numba_kernels.py               # numba JIT (OR reduce, 5D A*)

g1_arm_abs/models/abstraction_map/       # bitmaps + sidecars, package .gitignore'd
    ├── voxel_phase2_left.json              # metadata (joint ranges, grid shape, etc.)
    ├── voxel_phase2_right.json
    ├── feasibility_mask_left.uint8         # 60 KB, per-config self-collision mask
    ├── feasibility_mask_right.uint8
    ├── feasible_indices_left.int32         # 1.4 MB, compact-storage index list
    ├── feasible_indices_right.int32        # 1.4 MB
    ├── sparse_voxel_map_left.int32         # 216 KB, dense-voxel → sparse-slot map
    ├── sparse_voxel_map_right.int32        # 216 KB
    ├── voxel_config_bitmap_left.uint8      # 1.42 GB
    └── voxel_config_bitmap_right.uint8     # 1.42 GB
```

Total on-disk: **~2.85 GB** (raw bitmaps shipped as-is, no compression). The
bitmaps use **compact storage** — only reachable voxels are materialised, which
requires the two `.int32` sidecar files above.

---

## Install

```bash
pip install numpy numba
```

Dependencies: `numpy >= 1.22`, `numba >= 0.65`. No MuJoCo / URDF / pandas
needed at runtime.

---

## Minimal usage

```python
from voxel_phase4_astar import VoxelObstacleAStar

planner = VoxelObstacleAStar("left")   # or "right". ~1 s first call (numba warmup)

# Axis-aligned-box obstacles in the torso_link frame (x forward, y left, z up).
obstacles = [
    ((0.20, -0.40, -0.13), (0.40, 0.40, -0.09)),   # the table
    # ((x_min, y_min, z_min), (x_max, y_max, z_max)),   # more boxes...
]

path_idx, path_deg = planner.astar_staged(
    q_start_deg=[0, 20, 0, 0, 0],     # wrist_roll MUST be 0 — the only sampled value
    q_goal_deg=[20, 20, 10, 30, 0],   # wrist_roll MUST be 0 here too
    world_obstacles=obstacles,
    defer_joint=2,         # yaw is joint index 2 (shoulder_yaw)
    defer_cost_mult=2,     # soft yaw penalty when strict 2-stage is infeasible
    h_weight=1.0,
    shortcut=True,         # 5D hyperrect bounding-box merge on the path
)

# path_deg is [[pitch, roll, yaw, elbow, 0.0], ...] in degrees — wrist always 0.
# After the planner reaches q_goal, snap wrist_roll to its real target value
# in a final controller step (see "wrist_roll handling" note below).
# Between consecutive waypoints, any interpolation (PTP / independent-axis /
# trapezoidal-velocity) is safe — all intermediate grid cells inside the 5D
# bounding box between the two are reachable.
```

---

## Obstacle API

All obstacles are **in the torso_link frame**, axis-aligned, in **metres**.

### Joint order (degrees, always this order)
```
[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll]
```
The "third joint" users sometimes want to defer is **`shoulder_yaw`** at index 2.

### Joint limits (hard)
| joint | left (deg)     | right (deg)    |
|-------|----------------|----------------|
| pitch | [-90, 60]      | [-90, 60]      |
| roll  | [  0, 40]      | [-40,  0]      |
| yaw   | [-60, 60]      | [-60, 60]      |
| elbow | [-60, 90]      | [-60, 90]      |
| wrist | **{ 0 } only** | **{ 0 } only** |

> **wrist_roll is NOT planned.** The bitmap was sampled with `wrist_roll = 0°`
> only (sample_counts last dim = 1). At deploy time, navigate the first 4
> joints to the goal with the planner (passing `wrist_roll = 0` as both start
> and goal), then snap `wrist_roll` to its target value as a final post-step.
> The planner is not aware of hand orientation and assumes the hand is at
> roll = 0 throughout motion. **Caveat:** the post-step rotation around the
> hand's long axis can sweep the (asymmetric) rubber-hand mesh into nearby
> obstacles. Only do the wrist snap once the rest of the arm is at the goal
> pose AND you know the goal area is clear of obstacles within ~hand-radius.

### Workspace box (obstacles outside are silently clipped)
| arm   | x (m)        | y (m)         | z (m)         | grid shape   |
|-------|--------------|---------------|---------------|--------------|
| left  | [0.00, 0.60] | [-0.20, 0.70] | [-0.40, 0.40] | 30 × 45 × 40 |
| right | [0.00, 0.60] | [-0.70, 0.20] | [-0.40, 0.40] | 30 × 45 × 40 |
Voxel size = 0.02 m. Of the 54 000 voxels per arm, only ~33 000 are reachable
and stored in the compact bitmap.

### Formats accepted by every A* call
```python
# A) list of ((xyz_min), (xyz_max)) AABBs
world_obstacles = [((xmin, ymin, zmin), (xmax, ymax, zmax)), ...]

# B) pre-computed voxel indices (e.g. from point cloud / segmentation)
voxel_indices = np.array([[ix, iy, iz], ...])    # shape (N, 3)
# or linear indices (N,) int64:
voxel_indices = np.array([12345, 12346, ...], dtype=np.int64)

# C) mixed — both applied as union
planner.astar_staged(..., world_obstacles=..., voxel_indices=..., ...)
```

The query hashes the (sorted, unique) voxel-index set and caches the resulting
forbidden mask (LRU size = 4 by default). Repeating the same obstacle set
across multiple queries is effectively free.

---

## Planning methods

All three accept the same obstacle arguments.

| method | behaviour | typical ms |
|--------|-----------|------------|
| `astar_numba(q_start, q_goal, ...)` | plain grid A*, numba kernel | 10–50 |
| `astar_priority(q_start, q_goal, joint_costs=(1,1,2,1,1), ...)` | per-joint edge cost | 15–60 |
| `astar_staged(q_start, q_goal, defer_joint=2, defer_cost_mult=2, ...)` | **recommended.** Strict 2-stage: freeze `defer_joint` while other joints reach goal, then move `defer_joint` alone. Falls back to weighted A* when the strict plan is infeasible. Applies 5D hyperrect shortcut compression. | 15–80 |

### What `astar_staged` guarantees
- Whenever the strict 2-stage succeeds: `defer_joint` (yaw) moves **only in the
  final segment**; prior waypoints hold it at the start value.
- Fallback mode: `defer_joint` still moves as few times as possible (segmented
  hyperrect preserves the temporal ordering of yaw transitions).
- Compressed path length: typically 3–13 waypoints (vs 30–40 for raw A*).

### Path execution note
Consecutive waypoints in the compressed path are the corners of an
axis-aligned 5D bounding box whose interior is fully reachable. This means
**any interpolation** between two consecutive waypoints is collision-free —
independent-axis motion, PTP, trapezoidal velocity, whatever the controller
prefers. No need for joint-space linear interpolation specifically.

---

## Resource usage (default config: memmap + numba)

| item | value |
|------|-------|
| RAM (USS, single arm) | ~67 MB |
| RAM (USS, both arms)  | ~130 MB |
| Startup | ~1 s (numba JIT warmup) |
| Disk | ~2.85 GB |

If you are memory-constrained, you can pass `use_numba=False` to trade ~25 MB
of RAM for slower OR / A* kernels. Do not pass `preload=True` unless you
have 18 GB+ RAM — that mode pins the whole bitmap in RAM.

---

## Replacing an existing static-map planner

Old code (static, table baked in):
```python
planner = ReachableMapPlanner("reachable_map_left_step4.json")
path_idx, path_joints = planner.astar(q_start, q_goal)
```

New code (dynamic obstacles):
```python
planner = VoxelObstacleAStar("left")
path_idx, path_joints = planner.astar_staged(
    q_start, q_goal,
    world_obstacles=[((0.20, -0.40, -0.13), (0.40, 0.40, -0.09))],   # table
    defer_joint=2, defer_cost_mult=2,
)
```

Return format is the same: `path_joints[i] = [pitch, roll, yaw, elbow, 0.0]`
in degrees (wrist_roll is always 0 — see "wrist_roll handling" above; the
controller is responsible for snapping it to the real target after the plan
completes). The one behavioural change beyond wrist handling: the **number of
waypoints is much smaller** (hyperrect merges raw grid steps into corners), so
controllers that assumed every waypoint differs by exactly 4° need to
interpolate between consecutive waypoints themselves.

---

## File-path policy

`voxel_phase2_{arm}.json` contains absolute paths (`bitmap_path`,
`feasibility_path`, `feasible_indices_path`, `sparse_voxel_map_path`) recorded
at sampling time. The loader in `VoxelReachability` **first** looks for each
file next to the JSON by its canonical name, and only falls back to the
absolute path on miss. Relocating the `models/abstraction_map/` directory
to any machine works out of the box as long as the files stay siblings.

---

## Troubleshooting

- `ModuleNotFoundError: numba` — `pip install numba`
- `FileNotFoundError: voxel_config_bitmap_*.uint8` — the bitmap is missing from
  `models/abstraction_map/`. Re-extract the deploy package.
- `FileNotFoundError: feasible_indices_*.int32` / `sparse_voxel_map_*.int32` — the
  compact-storage sidecar files are missing from `models/abstraction_map/`. Re-extract the
  deploy package; the compact layout requires them.
- `RuntimeError: No reachable configuration ...` — the start or goal is in a
  region disconnected by the current obstacle set. Try a different start/goal,
  or shrink the obstacle.
- Path planning takes several seconds — start and goal are far apart through a
  highly constrained reachable region. Consider increasing `h_weight` (e.g.
  `h_weight=1.5`) for faster suboptimal paths.
