"""
Phase 4: A* joint-space path planning with dynamic voxel-obstacle awareness.

Given an obstacle set (either a list of (xyz_min, xyz_max) AABBs in the
torso_link frame, or pre-computed occupied voxels), this builds a 5D
reachable grid from the Phase 2 bitmap and runs A* from q_start to q_goal.

Two A* variants are provided:
  - astar:      reference implementation with tuple keys (slower)
  - astar_fast: flat-index pre-allocated implementation (3-5x faster),
                supports weighted A* via h_weight > 1 for more speedup.

Usage (as library):
    planner = VoxelObstacleAStar("left")
    path_idx, path_deg = planner.astar_fast(
        q_start_deg=[0,0,0,0,0],
        q_goal_deg=[20,20,10,30,-10],
        world_obstacles=[((0.2,-0.4,-0.13),(0.4,0.4,-0.09))],
        h_weight=1.5,
    )
"""

import argparse
import heapq
import os
import time
from collections import deque
from heapq import heappush, heappop

import numpy as np

from voxel_phase3_query import VoxelReachability

try:
    from voxel_numba_kernels import astar_5d as _astar_5d_nb
    from voxel_numba_kernels import astar_5d_priority as _astar_5d_priority_nb
    _HAS_NUMBA_ASTAR = True
except Exception as _nb_import_err:
    # Silent fallback used to mask missing-numba at import time, which then
    # surfaced much later as "numba A* priority kernel not available" from
    # astar_staged. Print the actual cause so the interpreter running the
    # node reveals it in startup logs.
    import sys as _sys
    print(
        f"[voxel_phase4_astar] numba kernels NOT loaded "
        f"({type(_nb_import_err).__name__}: {_nb_import_err}); "
        f"the voxel A* will raise when called.",
        file=_sys.stderr,
    )
    _HAS_NUMBA_ASTAR = False


def format_seconds(sec):
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def compress_path(path):
    """Merge consecutive steps moving in the same direction.

    Keeps only the 'corners' of a grid A* path. Works on both index tuples
    and degree lists, as long as each element is a sequence of numbers.

    Example:
        [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2)]
        -> [(0,0), (3,0), (3,2)]
    """
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    dims = len(path[0])
    prev_delta = tuple(path[1][d] - path[0][d] for d in range(dims))
    for i in range(1, len(path) - 1):
        next_delta = tuple(path[i + 1][d] - path[i][d] for d in range(dims))
        if next_delta != prev_delta:
            out.append(path[i])
            prev_delta = next_delta
    out.append(path[-1])
    return out


def _hyperrect_fully_reachable(grid, a_idx, b_idx):
    """Return True iff every grid cell in the axis-aligned bounding box
    defined by index tuples a_idx and b_idx has grid == 1.
    """
    dims = len(a_idx)
    slicer = []
    for d in range(dims):
        lo = int(a_idx[d]); hi = int(b_idx[d])
        if lo > hi:
            lo, hi = hi, lo
        slicer.append(slice(lo, hi + 1))
    # uint8 grid: .min() == 1 iff every cell is 1.
    return bool(grid[tuple(slicer)].min())


def compress_path_hyperrect_segmented(path_idx, grid, fixed_joint):
    """Apply hyperrect shortcut within segments where `fixed_joint` is constant.
    The waypoints at which `fixed_joint` transitions are preserved, which keeps
    the temporal ordering of the deferred joint's moves.
    """
    if len(path_idx) <= 1:
        return list(path_idx)
    boundaries = [0]
    for i in range(1, len(path_idx)):
        if path_idx[i][fixed_joint] != path_idx[i - 1][fixed_joint]:
            boundaries.append(i)
    boundaries.append(len(path_idx))

    out = []
    for s in range(len(boundaries) - 1):
        seg = path_idx[boundaries[s]:boundaries[s + 1]]
        if not seg:
            continue
        comp = compress_path_hyperrect(seg, grid)
        if not out:
            out.extend(comp)
        else:
            # seg's first point is adjacent to out's last via a single step in
            # fixed_joint; keep both (no dedup).
            out.extend(comp)
    return out


def _line_clear(grid, a_idx, b_idx):
    """Sample the straight line from a_idx to b_idx in joint-index space and
    check every intermediate grid cell is reachable. Less conservative than
    the full 5D bounding box — safe to merge if the controller executes the
    segment by joint-space linear interpolation."""
    a = np.array(a_idx, dtype=float)
    b = np.array(b_idx, dtype=float)
    max_step = int(np.max(np.abs(b - a)))
    if max_step == 0:
        return True
    n_samples = max_step + 1
    for t in np.linspace(0.0, 1.0, n_samples):
        idx = np.rint(a + (b - a) * t).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(grid.shape)):
            return False
        if grid[tuple(int(v) for v in idx)] == 0:
            return False
    return True


def compress_path_linecheck(path_idx, grid):
    """Greedy shortcut using joint-space LINE checks (not 5D bounding box).
    Much more aggressive than hyperrect — merges any (A, B) where every cell
    along the straight-line interpolation is reachable. Safe for controllers
    that execute each segment via joint-space PTP / linear interpolation."""
    n = len(path_idx)
    if n <= 2:
        return list(path_idx)
    out = [tuple(int(v) for v in path_idx[0])]
    anchor = 0
    j = 1
    while j < n:
        if _line_clear(grid, path_idx[anchor], path_idx[j]):
            j += 1
            continue
        corner = tuple(int(v) for v in path_idx[j - 1])
        if corner != out[-1]:
            out.append(corner)
        anchor = j - 1
    last = tuple(int(v) for v in path_idx[-1])
    if last != out[-1]:
        out.append(last)
    return out


def compress_path_hyperrect(path_idx, grid):
    """Greedy shortcut: walk along the path and extend the anchor as long as
    the 5D bounding box between anchor and next candidate is fully reachable.

    path_idx : list of index tuples (same dims as grid)
    grid     : uint8 N-D array (1 = reachable)

    The returned waypoints are safe to execute with joint-space linear
    interpolation between consecutive entries.
    """
    n = len(path_idx)
    if n <= 2:
        return list(path_idx)
    out = [tuple(int(v) for v in path_idx[0])]
    anchor = 0
    j = 1
    while j < n:
        # Extend while hyperrect(anchor, j) stays fully reachable.
        if _hyperrect_fully_reachable(grid, path_idx[anchor], path_idx[j]):
            j += 1
            continue
        # path_idx[j] would break the hyperrect; commit path_idx[j-1] as corner.
        corner = tuple(int(v) for v in path_idx[j - 1])
        if corner != out[-1]:
            out.append(corner)
        anchor = j - 1
        # Re-check from the new anchor with the same j
    last = tuple(int(v) for v in path_idx[-1])
    if last != out[-1]:
        out.append(last)
    return out


class VoxelObstacleAStar:
    def __init__(self, arm, output_dir=None, preload=False, use_numba=True):
        self.vr = VoxelReachability(arm, output_dir=output_dir, preload=preload,
                                    use_numba=use_numba)
        self.arm = arm
        self.sample_counts = list(self.vr.sample_counts)
        self.dims = len(self.sample_counts)
        self.deg_lists = self.vr.deg_lists
        self.jmin = np.array([float(dl[0]) for dl in self.deg_lists])
        # Assume uniform step per joint.
        self.step = float(self.deg_lists[0][1] - self.deg_lists[0][0]) if len(self.deg_lists[0]) > 1 else 1.0
        self.use_numba_astar = bool(use_numba) and _HAS_NUMBA_ASTAR
        if self.use_numba_astar:
            self._warmup_astar()

    def _warmup_astar(self):
        sc = self.sample_counts
        grid_flat = np.zeros(int(np.prod(sc)), dtype=np.uint8)
        grid_flat[0] = 1
        _astar_5d_nb(grid_flat,
                     int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3]), int(sc[4]),
                     0, 0, 0, 0, 0, 0, 0, 100)

    # ---- grid construction --------------------------------------------------

    def build_reachable_grid(self, world_obstacles=None, voxel_indices=None):
        """Return a (dim1, dim2, dim3, dim4, dim5) uint8 grid, 1 = reachable."""
        mask = self.vr.reachable_mask(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )
        bits = np.unpackbits(mask, bitorder="little")[:self.vr.total_configs]
        return bits.reshape(self.sample_counts).astype(np.uint8)

    # ---- coordinate helpers -------------------------------------------------

    def joints_to_idx(self, q_deg):
        arr = np.asarray(q_deg, dtype=float)
        raw = (arr - self.jmin) / self.step
        return tuple(int(v) for v in np.rint(raw).astype(int))

    def idx_to_joints(self, idx):
        return [float(self.deg_lists[d][idx[d]]) for d in range(self.dims)]

    def in_bounds(self, idx):
        return all(0 <= idx[d] < self.sample_counts[d] for d in range(self.dims))

    # ---- A* -----------------------------------------------------------------

    def nearest_reachable(self, idx, grid):
        """BFS on the reachable grid to find the closest reachable cell."""
        if self.in_bounds(idx) and grid[idx] == 1:
            return idx
        start = tuple(np.clip(np.array(idx), 0,
                              np.array(self.sample_counts) - 1).astype(int))
        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for d in range(self.dims):
                for delta in (-1, 1):
                    nxt = list(cur)
                    nxt[d] += delta
                    nxt = tuple(nxt)
                    if nxt in visited or not self.in_bounds(nxt):
                        continue
                    if grid[nxt] == 1:
                        return nxt
                    visited.add(nxt)
                    queue.append(nxt)
        raise RuntimeError("No reachable configuration in grid under current obstacle.")

    def astar(self, q_start_deg, q_goal_deg,
              world_obstacles=None, voxel_indices=None,
              verbose=True):
        t0 = time.time()
        grid = self.build_reachable_grid(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )
        t_grid = time.time() - t0
        reachable_count = int(grid.sum())

        start_idx = self.joints_to_idx(q_start_deg)
        goal_idx = self.joints_to_idx(q_goal_deg)

        if not (self.in_bounds(start_idx) and grid[start_idx] == 1):
            start_idx = self.nearest_reachable(start_idx, grid)
        if not (self.in_bounds(goal_idx) and grid[goal_idx] == 1):
            goal_idx = self.nearest_reachable(goal_idx, grid)

        if verbose:
            print(f"grid build {t_grid*1000:.1f} ms, reachable {reachable_count} / "
                  f"{self.vr.total_configs}")
            print(f"start_idx={start_idx} goal_idx={goal_idx}")

        open_heap = [(0, start_idx)]
        g_cost = {start_idx: 0}
        parent = {}
        visited_count = 0
        goal_arr = np.array(goal_idx)

        while open_heap:
            _, current = heapq.heappop(open_heap)
            visited_count += 1
            if current == goal_idx:
                path_idx = [current]
                while path_idx[-1] != start_idx:
                    path_idx.append(parent[path_idx[-1]])
                path_idx.reverse()
                path_deg = [self.idx_to_joints(p) for p in path_idx]
                if verbose:
                    elapsed = time.time() - t0
                    print(f"A* done in {elapsed*1000:.1f} ms, visited={visited_count}, "
                          f"waypoints={len(path_idx)}")
                return path_idx, path_deg

            for d in range(self.dims):
                for delta in (-1, 1):
                    nxt = list(current)
                    nxt[d] += delta
                    nxt = tuple(nxt)
                    if not self.in_bounds(nxt) or grid[nxt] == 0:
                        continue
                    new_cost = g_cost[current] + 1
                    if nxt not in g_cost or new_cost < g_cost[nxt]:
                        g_cost[nxt] = new_cost
                        h = int(np.sum(np.abs(np.array(nxt) - goal_arr)))
                        heapq.heappush(open_heap, (new_cost + h, nxt))
                        parent[nxt] = current

        raise RuntimeError("A* failed: no path from start to goal under current obstacle.")

    # ---- fast A* ------------------------------------------------------------

    # ---- numba A* ------------------------------------------------------------

    def astar_numba(self, q_start_deg, q_goal_deg,
                    world_obstacles=None, voxel_indices=None,
                    h_weight=1.0, verbose=True):
        """A* using the numba-JIT'd 5D search kernel. Same signature."""
        if not self.use_numba_astar:
            raise RuntimeError("numba A* kernel not available")
        t0 = time.time()
        grid = self.build_reachable_grid(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )
        grid_flat = np.ascontiguousarray(grid.ravel(), dtype=np.uint8)
        t_grid = time.time() - t0

        sc = self.sample_counts
        sc0, sc1, sc2, sc3, sc4 = (int(s) for s in sc)
        stride0 = sc1 * sc2 * sc3 * sc4
        stride1 = sc2 * sc3 * sc4
        stride2 = sc3 * sc4
        stride3 = sc4

        start_idx = self.joints_to_idx(q_start_deg)
        goal_idx = self.joints_to_idx(q_goal_deg)
        if not (self.in_bounds(start_idx) and grid[start_idx] == 1):
            start_idx = self.nearest_reachable(start_idx, grid)
        if not (self.in_bounds(goal_idx) and grid[goal_idx] == 1):
            goal_idx = self.nearest_reachable(goal_idx, grid)

        def coord_to_flat(c):
            return int(c[0]) * stride0 + int(c[1]) * stride1 + int(c[2]) * stride2 \
                   + int(c[3]) * stride3 + int(c[4])

        start_flat = coord_to_flat(start_idx)
        goal_flat = coord_to_flat(goal_idx)
        g0, g1, g2, g3, g4 = (int(c) for c in goal_idx)
        hw100 = int(round(h_weight * 100))

        visited, found, parent = _astar_5d_nb(
            grid_flat, sc0, sc1, sc2, sc3, sc4,
            start_flat, goal_flat, g0, g1, g2, g3, g4, hw100,
        )
        if not found:
            raise RuntimeError("A* failed: no path under current obstacle.")

        # Reconstruct path
        path_flat = [goal_flat]
        while path_flat[-1] != start_flat:
            p = int(parent[path_flat[-1]])
            if p < 0:
                raise RuntimeError("Broken parent chain.")
            path_flat.append(p)
        path_flat.reverse()

        path_idx = []
        for f in path_flat:
            c0 = f // stride0;  r = f - c0 * stride0
            c1 = r // stride1;  r = r - c1 * stride1
            c2 = r // stride2;  r = r - c2 * stride2
            c3 = r // stride3;  c4 = r - c3 * stride3
            path_idx.append((c0, c1, c2, c3, c4))
        path_deg = [self.idx_to_joints(p) for p in path_idx]

        if verbose:
            elapsed = time.time() - t0
            print(f"astar_numba done in {elapsed*1000:.1f} ms, "
                  f"visited={visited}, waypoints={len(path_idx)}")
        return path_idx, path_deg

    # ---- priority A* (per-joint edge cost) ---------------------------------

    def astar_priority(self, q_start_deg, q_goal_deg,
                       world_obstacles=None, voxel_indices=None,
                       joint_costs=(1, 1, 3, 1, 1),
                       h_weight=1.0, verbose=True):
        """A* with per-joint edge costs. Larger cost on a dim = A* avoids moving
        that joint unless necessary. Default (1,1,3,1,1) triples yaw cost."""
        if not self.use_numba_astar:
            raise RuntimeError("numba A* priority kernel not available")
        t0 = time.time()
        grid = self.build_reachable_grid(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )
        grid_flat = np.ascontiguousarray(grid.ravel(), dtype=np.uint8)

        sc = self.sample_counts
        sc0, sc1, sc2, sc3, sc4 = (int(s) for s in sc)
        stride0 = sc1 * sc2 * sc3 * sc4
        stride1 = sc2 * sc3 * sc4
        stride2 = sc3 * sc4
        stride3 = sc4

        start_idx = self.joints_to_idx(q_start_deg)
        goal_idx = self.joints_to_idx(q_goal_deg)
        if not (self.in_bounds(start_idx) and grid[start_idx] == 1):
            start_idx = self.nearest_reachable(start_idx, grid)
        if not (self.in_bounds(goal_idx) and grid[goal_idx] == 1):
            goal_idx = self.nearest_reachable(goal_idx, grid)

        def to_flat(c):
            return int(c[0]) * stride0 + int(c[1]) * stride1 + int(c[2]) * stride2 \
                   + int(c[3]) * stride3 + int(c[4])

        start_flat = to_flat(start_idx)
        goal_flat = to_flat(goal_idx)
        g0, g1, g2, g3, g4 = (int(c) for c in goal_idx)
        hw100 = int(round(h_weight * 100))
        c0, c1, c2, c3, c4 = (int(x) for x in joint_costs)

        visited, found, parent = _astar_5d_priority_nb(
            grid_flat, sc0, sc1, sc2, sc3, sc4,
            start_flat, goal_flat, g0, g1, g2, g3, g4, hw100,
            c0, c1, c2, c3, c4,
        )
        if not found:
            raise RuntimeError("priority A* failed.")

        path_flat = [goal_flat]
        while path_flat[-1] != start_flat:
            p = int(parent[path_flat[-1]])
            if p < 0:
                raise RuntimeError("Broken parent chain.")
            path_flat.append(p)
        path_flat.reverse()

        path_idx = []
        for f in path_flat:
            c0_ = f // stride0;  r = f - c0_ * stride0
            c1_ = r // stride1;  r = r - c1_ * stride1
            c2_ = r // stride2;  r = r - c2_ * stride2
            c3_ = r // stride3;  c4_ = r - c3_ * stride3
            path_idx.append((c0_, c1_, c2_, c3_, c4_))
        path_deg = [self.idx_to_joints(p) for p in path_idx]

        if verbose:
            elapsed = time.time() - t0
            print(f"astar_priority done in {elapsed*1000:.1f} ms, "
                  f"visited={visited}, waypoints={len(path_idx)}, "
                  f"joint_costs={joint_costs}")
        return path_idx, path_deg

    # ---- staged A* (defer one joint) ---------------------------------------

    def _astar_on_grid(self, grid, start_idx, goal_idx, h_weight,
                       joint_costs=None):
        """Run the numba A* kernel on a pre-built (possibly masked) reachable grid.
        Returns (path_idx, path_deg). Raises RuntimeError if no path exists.
        If joint_costs is provided (length-5 tuple/list), uses the priority
        kernel with per-joint edge costs; otherwise uses uniform-cost A*."""
        if not self.use_numba_astar:
            raise RuntimeError("numba A* kernel not available for _astar_on_grid")

        grid_flat = np.ascontiguousarray(grid.ravel(), dtype=np.uint8)
        sc = self.sample_counts
        sc0, sc1, sc2, sc3, sc4 = (int(s) for s in sc)
        stride0 = sc1 * sc2 * sc3 * sc4
        stride1 = sc2 * sc3 * sc4
        stride2 = sc3 * sc4
        stride3 = sc4

        # Both endpoints must be reachable on this (possibly masked) grid.
        if grid[tuple(start_idx)] == 0:
            raise RuntimeError(f"start {tuple(start_idx)} not reachable on masked grid")
        if grid[tuple(goal_idx)] == 0:
            raise RuntimeError(f"goal {tuple(goal_idx)} not reachable on masked grid")

        def to_flat(c):
            return int(c[0]) * stride0 + int(c[1]) * stride1 + int(c[2]) * stride2 \
                   + int(c[3]) * stride3 + int(c[4])

        start_flat = to_flat(start_idx)
        goal_flat = to_flat(goal_idx)
        g0, g1, g2, g3, g4 = (int(c) for c in goal_idx)
        hw100 = int(round(h_weight * 100))

        if joint_costs is not None:
            c0_cost, c1_cost, c2_cost, c3_cost, c4_cost = (int(x) for x in joint_costs)
            visited, found, parent = _astar_5d_priority_nb(
                grid_flat, sc0, sc1, sc2, sc3, sc4,
                start_flat, goal_flat, g0, g1, g2, g3, g4, hw100,
                c0_cost, c1_cost, c2_cost, c3_cost, c4_cost,
            )
        else:
            visited, found, parent = _astar_5d_nb(
                grid_flat, sc0, sc1, sc2, sc3, sc4,
                start_flat, goal_flat, g0, g1, g2, g3, g4, hw100,
            )
        if not found:
            raise RuntimeError("A* kernel failed on masked grid")

        path_flat = [goal_flat]
        while path_flat[-1] != start_flat:
            p = int(parent[path_flat[-1]])
            if p < 0:
                raise RuntimeError("Broken parent chain")
            path_flat.append(p)
        path_flat.reverse()

        path_idx = []
        for f in path_flat:
            c0 = f // stride0;  r = f - c0 * stride0
            c1 = r // stride1;  r = r - c1 * stride1
            c2 = r // stride2;  r = r - c2 * stride2
            c3 = r // stride3;  c4 = r - c3 * stride3
            path_idx.append((c0, c1, c2, c3, c4))
        path_deg = [self.idx_to_joints(p) for p in path_idx]
        return path_idx, path_deg

    def astar_staged(self, q_start_deg, q_goal_deg,
                     world_obstacles=None, voxel_indices=None,
                     defer_joint=2, defer_cost_mult=3,
                     joint_costs=(1, 10, 10, 2, 10),
                     h_weight=1.0, compress=True, shortcut=True,
                     verbose=True):
        """Two-stage planner: `defer_joint` is held fixed during Stage 1 and
        moved alone in Stage 2.

        Stage 1: every joint except `defer_joint` reaches its goal value, while
                 `defer_joint` stays at its start value (grid is masked so only
                 the slice `defer_joint == start_idx[defer_joint]` is reachable).
        Stage 2: only `defer_joint` can move (grid is masked so only the line
                 with all other joints at goal values is reachable).

        On infeasibility, falls back to single-stage astar_numba.

        Returns (path_idx, path_deg). When `compress=True`, consecutive steps
        on the same axis/direction are merged so only corners remain.
        """
        t0 = time.time()

        grid = self.build_reachable_grid(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )

        start_idx_raw = self.joints_to_idx(q_start_deg)
        goal_idx_raw = self.joints_to_idx(q_goal_deg)
        if not (self.in_bounds(start_idx_raw) and grid[start_idx_raw] == 1):
            start_idx_raw = self.nearest_reachable(start_idx_raw, grid)
        if not (self.in_bounds(goal_idx_raw) and grid[goal_idx_raw] == 1):
            goal_idx_raw = self.nearest_reachable(goal_idx_raw, grid)
        start_idx = tuple(int(c) for c in start_idx_raw)
        goal_idx = tuple(int(c) for c in goal_idx_raw)

        intermediate_idx = list(goal_idx)
        intermediate_idx[defer_joint] = start_idx[defer_joint]
        intermediate_idx = tuple(intermediate_idx)

        stage_label = "fallback"
        path_idx = None
        path_deg = None

        try:
            # Stage 1 mask: only cells with defer_joint == start_idx[defer_joint].
            grid_s1 = np.zeros_like(grid)
            sl1 = [slice(None)] * grid.ndim
            sl1[defer_joint] = start_idx[defer_joint]
            grid_s1[tuple(sl1)] = grid[tuple(sl1)]

            p1_idx, p1_deg = self._astar_on_grid(
                grid_s1, start_idx, intermediate_idx, h_weight,
                joint_costs=joint_costs,
            )

            # Stage 2 mask: only cells where every dim except defer_joint equals goal.
            grid_s2 = np.zeros_like(grid)
            sl2 = [slice(None)] * grid.ndim
            for d in range(grid.ndim):
                if d != defer_joint:
                    sl2[d] = goal_idx[d]
            grid_s2[tuple(sl2)] = grid[tuple(sl2)]

            p2_idx, p2_deg = self._astar_on_grid(
                grid_s2, intermediate_idx, goal_idx, h_weight,
                joint_costs=joint_costs,
            )

            # Compress each stage INDEPENDENTLY, then join. This preserves the
            # temporal boundary so the deferred joint only moves in the last
            # compressed segment.
            if shortcut:
                p1_idx = compress_path_linecheck(p1_idx, grid_s1)
                p2_idx = compress_path_linecheck(p2_idx, grid_s2)
                p1_deg = [self.idx_to_joints(p) for p in p1_idx]
                p2_deg = [self.idx_to_joints(p) for p in p2_idx]
            elif compress:
                p1_idx = compress_path(p1_idx); p1_deg = compress_path(p1_deg)
                p2_idx = compress_path(p2_idx); p2_deg = compress_path(p2_deg)

            path_idx = list(p1_idx) + list(p2_idx[1:])
            path_deg = list(p1_deg) + list(p2_deg[1:])
            stage_label = "staged"
            shortcut_done = shortcut
            compress_done = compress
        except RuntimeError as err:
            if verbose:
                print(f"staged plan failed ({err}); "
                      f"falling back to soft-priority A* with defer_joint "
                      f"cost x3.")
            path_idx, path_deg = self.astar_priority(
                q_start_deg, q_goal_deg,
                world_obstacles=world_obstacles,
                voxel_indices=voxel_indices,
                joint_costs=joint_costs,
                h_weight=h_weight, verbose=False,
            )
            stage_label = f"fallback_priority(x{defer_cost_mult})"
            if shortcut:
                # Fallback path can be far from straight. Use joint-space
                # line-of-sight shortcut (not full 5D bounding box) — safely
                # merges any two waypoints whose interpolated line stays in
                # the reachable grid. Fewer waypoints => smoother controller.
                path_idx = compress_path_linecheck(path_idx, grid)
                path_deg = [self.idx_to_joints(p) for p in path_idx]
            elif compress:
                path_idx = compress_path(path_idx)
                path_deg = compress_path(path_deg)
            shortcut_done = shortcut
            compress_done = compress

        if verbose:
            elapsed = time.time() - t0
            print(f"astar_staged [{stage_label}] in {elapsed*1000:.1f} ms, "
                  f"waypoints={len(path_idx)} (defer_joint={defer_joint}, "
                  f"shortcut={shortcut_done}, compress={compress_done})")
        return path_idx, path_deg

    def astar_fast(self, q_start_deg, q_goal_deg,
                   world_obstacles=None, voxel_indices=None,
                   h_weight=1.0, verbose=True):
        """Flat-index A* with preallocated arrays. Same interface as astar()."""
        t0 = time.time()
        grid = self.build_reachable_grid(
            world_obstacles=world_obstacles,
            voxel_indices=voxel_indices,
        )
        grid_flat = grid.ravel()              # (total_configs,) uint8, C-order
        t_grid = time.time() - t0

        sc = self.sample_counts
        sc0, sc1, sc2, sc3, sc4 = (int(s) for s in sc)
        stride0 = sc1 * sc2 * sc3 * sc4
        stride1 = sc2 * sc3 * sc4
        stride2 = sc3 * sc4
        stride3 = sc4
        total = sc0 * stride0

        start_idx = self.joints_to_idx(q_start_deg)
        goal_idx = self.joints_to_idx(q_goal_deg)

        def coord_to_flat(c):
            return int(c[0]) * stride0 + int(c[1]) * stride1 + int(c[2]) * stride2 \
                   + int(c[3]) * stride3 + int(c[4])

        if not (self.in_bounds(start_idx) and grid[start_idx] == 1):
            start_idx = self.nearest_reachable(start_idx, grid)
        if not (self.in_bounds(goal_idx) and grid[goal_idx] == 1):
            goal_idx = self.nearest_reachable(goal_idx, grid)

        start_flat = coord_to_flat(start_idx)
        goal_flat = coord_to_flat(goal_idx)
        g0, g1, g2, g3, g4 = (int(c) for c in goal_idx)

        INF = np.iinfo(np.int32).max
        g_cost = np.full(total, INF, dtype=np.int32)
        parent = np.full(total, -1, dtype=np.int32)
        g_cost[start_flat] = 0

        # First entry: f, counter, flat
        heap = [(0, 0, start_flat)]
        counter = 0
        visited = 0

        hw = float(h_weight)

        if verbose:
            reachable_count = int(grid_flat.sum())
            print(f"grid build {t_grid*1000:.1f} ms, reachable {reachable_count} / {total}")
            print(f"start_idx={start_idx} goal_idx={goal_idx} h_weight={hw}")

        # Pull into locals for speed inside the hot loop.
        _heappush = heappush
        _heappop = heappop

        found = False
        while heap:
            f, _, current = _heappop(heap)
            if current == goal_flat:
                found = True
                break
            cur_g = int(g_cost[current])
            # Skip stale heap entries: current was popped with cost > current best g.
            # For consistent heuristic this does not trigger, but kept as safety.
            # (f - h*hw should equal cur_g; if heap entry is stale it will be >.)
            # We just check by recomputing what f should have been:
            # skip only if current was already closed (we use g_cost as the marker
            # by never pushing to heap unless strictly improving).
            visited += 1

            # Decode coords
            c0 = current // stride0
            r0 = current - c0 * stride0
            c1 = r0 // stride1
            r1 = r0 - c1 * stride1
            c2 = r1 // stride2
            r2 = r1 - c2 * stride2
            c3 = r2 // stride3
            c4 = r2 - c3 * stride3

            new_g = cur_g + 1

            # Dim 0
            if c0 > 0:
                nxt = current - stride0
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs((c0 - 1) - g0) + abs(c1 - g1) + abs(c2 - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            if c0 < sc0 - 1:
                nxt = current + stride0
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs((c0 + 1) - g0) + abs(c1 - g1) + abs(c2 - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            # Dim 1
            if c1 > 0:
                nxt = current - stride1
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs((c1 - 1) - g1) + abs(c2 - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            if c1 < sc1 - 1:
                nxt = current + stride1
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs((c1 + 1) - g1) + abs(c2 - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            # Dim 2
            if c2 > 0:
                nxt = current - stride2
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs((c2 - 1) - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            if c2 < sc2 - 1:
                nxt = current + stride2
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs((c2 + 1) - g2) + abs(c3 - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            # Dim 3
            if c3 > 0:
                nxt = current - stride3
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs(c2 - g2) + abs((c3 - 1) - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            if c3 < sc3 - 1:
                nxt = current + stride3
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs(c2 - g2) + abs((c3 + 1) - g3) + abs(c4 - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            # Dim 4
            if c4 > 0:
                nxt = current - 1
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs(c2 - g2) + abs(c3 - g3) + abs((c4 - 1) - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))
            if c4 < sc4 - 1:
                nxt = current + 1
                if grid_flat[nxt] and new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    h = abs(c0 - g0) + abs(c1 - g1) + abs(c2 - g2) + abs(c3 - g3) + abs((c4 + 1) - g4)
                    counter += 1
                    _heappush(heap, (new_g + int(h * hw), counter, nxt))

        if not found:
            raise RuntimeError("A* failed: no path from start to goal under current obstacle.")

        # Reconstruct path
        path_flat = [goal_flat]
        while path_flat[-1] != start_flat:
            p = int(parent[path_flat[-1]])
            if p < 0:
                raise RuntimeError("Broken parent chain during path reconstruction.")
            path_flat.append(p)
        path_flat.reverse()

        path_idx = []
        for f in path_flat:
            c0 = f // stride0;   r = f - c0 * stride0
            c1 = r // stride1;   r = r - c1 * stride1
            c2 = r // stride2;   r = r - c2 * stride2
            c3 = r // stride3;   c4 = r - c3 * stride3
            path_idx.append((c0, c1, c2, c3, c4))
        path_deg = [self.idx_to_joints(p) for p in path_idx]

        if verbose:
            elapsed = time.time() - t0
            print(f"astar_fast done in {elapsed*1000:.1f} ms, "
                  f"visited={visited}, waypoints={len(path_idx)}")
        return path_idx, path_deg


def _demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--q_start", type=float, nargs=5, default=[0, 20, 0, 0, 0])
    parser.add_argument("--q_goal", type=float, nargs=5, default=[20, 20, 10, 30, -10])
    parser.add_argument("--no_table", action="store_true",
                        help="Skip the default table obstacle.")
    args = parser.parse_args()

    planner = VoxelObstacleAStar(args.arm)
    obstacles = []
    if not args.no_table:
        obstacles.append(((0.20, -0.40, -0.13), (0.40, 0.40, -0.09)))

    print(f"arm={args.arm} q_start={args.q_start} q_goal={args.q_goal}")
    print(f"obstacles={obstacles}")

    path_idx, path_deg = planner.astar(
        args.q_start, args.q_goal,
        world_obstacles=obstacles,
        verbose=True,
    )
    print(f"\nPath length: {len(path_deg)} waypoints")
    for i, p in enumerate(path_deg[:10]):
        print(f"  [{i}] {np.round(p, 2).tolist()}")
    if len(path_deg) > 10:
        print(f"  ... ({len(path_deg) - 10} more)")


if __name__ == "__main__":
    _demo()
