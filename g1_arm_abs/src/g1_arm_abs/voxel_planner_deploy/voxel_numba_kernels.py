"""
Numba-compiled kernels for hot paths:
  - or_reduce_runs: OR-reduce bitmap rows for a sorted, run-encoded voxel set.
  - astar_5d:       A* over 5D uniform grid with flat indices.
"""

import numpy as np
from numba import njit


@njit(cache=True, boundscheck=False, fastmath=True)
def or_reduce_runs(bitmap, lin, out):
    """
    OR-reduce selected rows of `bitmap` into `out`.

    bitmap : (num_voxels, bytes_per_voxel) uint8
    lin    : (N,) int64, sorted unique voxel indices
    out    : (bytes_per_voxel,) uint8, accumulator (modified in place)
    """
    bpv = bitmap.shape[1]
    n = lin.shape[0]
    for k in range(n):
        row = lin[k]
        for j in range(bpv):
            out[j] |= bitmap[row, j]


@njit(cache=True, boundscheck=False, fastmath=True)
def astar_5d_priority(grid_flat,
                      sc0, sc1, sc2, sc3, sc4,
                      start_flat, goal_flat,
                      g0, g1, g2, g3, g4,
                      h_weight_times_100,
                      c0_cost, c1_cost, c2_cost, c3_cost, c4_cost):
    """Same as astar_5d but each dimension has its own edge cost (int)."""
    stride0 = sc1 * sc2 * sc3 * sc4
    stride1 = sc2 * sc3 * sc4
    stride2 = sc3 * sc4
    stride3 = sc4
    total = sc0 * stride0

    INF = np.iinfo(np.int32).max
    g_cost = np.full(total, INF, dtype=np.int64)
    parent = np.full(total, -1, dtype=np.int32)
    g_cost[start_flat] = 0

    cap = 1024
    heap_f = np.empty(cap, dtype=np.int64)
    heap_node = np.empty(cap, dtype=np.int64)
    heap_f[0] = 0
    heap_node[0] = start_flat
    heap_size = 1
    visited = 0
    found = False

    while heap_size > 0:
        f = heap_f[0]
        current = heap_node[0]
        heap_size -= 1
        if heap_size > 0:
            heap_f[0] = heap_f[heap_size]
            heap_node[0] = heap_node[heap_size]
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                smallest = i
                if l < heap_size and heap_f[l] < heap_f[smallest]:
                    smallest = l
                if r < heap_size and heap_f[r] < heap_f[smallest]:
                    smallest = r
                if smallest == i:
                    break
                tf = heap_f[i]; tn = heap_node[i]
                heap_f[i] = heap_f[smallest]; heap_node[i] = heap_node[smallest]
                heap_f[smallest] = tf; heap_node[smallest] = tn
                i = smallest

        if current == goal_flat:
            found = True
            break
        cur_g = g_cost[current]
        visited += 1

        c0_cur = current // stride0
        r0 = current - c0_cur * stride0
        c1_cur = r0 // stride1
        r1 = r0 - c1_cur * stride1
        c2_cur = r1 // stride2
        r2 = r1 - c2_cur * stride2
        c3_cur = r2 // stride3
        c4_cur = r2 - c3_cur * stride3

        for dim in range(5):
            for delta in (-1, 1):
                if dim == 0:
                    nc = c0_cur + delta
                    if nc < 0 or nc >= sc0: continue
                    nxt = current + delta * stride0
                    step_cost = c0_cost
                elif dim == 1:
                    nc = c1_cur + delta
                    if nc < 0 or nc >= sc1: continue
                    nxt = current + delta * stride1
                    step_cost = c1_cost
                elif dim == 2:
                    nc = c2_cur + delta
                    if nc < 0 or nc >= sc2: continue
                    nxt = current + delta * stride2
                    step_cost = c2_cost
                elif dim == 3:
                    nc = c3_cur + delta
                    if nc < 0 or nc >= sc3: continue
                    nxt = current + delta * stride3
                    step_cost = c3_cost
                else:
                    nc = c4_cur + delta
                    if nc < 0 or nc >= sc4: continue
                    nxt = current + delta
                    step_cost = c4_cost

                if grid_flat[nxt] == 0:
                    continue
                new_g = cur_g + step_cost
                if new_g >= g_cost[nxt]:
                    continue
                g_cost[nxt] = new_g
                parent[nxt] = current

                if dim == 0:
                    nn0 = c0_cur + delta; nn1 = c1_cur; nn2 = c2_cur; nn3 = c3_cur; nn4 = c4_cur
                elif dim == 1:
                    nn0 = c0_cur; nn1 = c1_cur + delta; nn2 = c2_cur; nn3 = c3_cur; nn4 = c4_cur
                elif dim == 2:
                    nn0 = c0_cur; nn1 = c1_cur; nn2 = c2_cur + delta; nn3 = c3_cur; nn4 = c4_cur
                elif dim == 3:
                    nn0 = c0_cur; nn1 = c1_cur; nn2 = c2_cur; nn3 = c3_cur + delta; nn4 = c4_cur
                else:
                    nn0 = c0_cur; nn1 = c1_cur; nn2 = c2_cur; nn3 = c3_cur; nn4 = c4_cur + delta

                # Admissible heuristic: weighted Manhattan using each dim's cost.
                h = (abs(nn0 - g0) * c0_cost + abs(nn1 - g1) * c1_cost
                     + abs(nn2 - g2) * c2_cost + abs(nn3 - g3) * c3_cost
                     + abs(nn4 - g4) * c4_cost)
                new_f = new_g + (h * h_weight_times_100) // 100

                if heap_size == heap_f.shape[0]:
                    new_cap = heap_f.shape[0] * 2
                    nf = np.empty(new_cap, dtype=np.int64)
                    nn = np.empty(new_cap, dtype=np.int64)
                    for q in range(heap_size):
                        nf[q] = heap_f[q]
                        nn[q] = heap_node[q]
                    heap_f = nf
                    heap_node = nn

                heap_f[heap_size] = new_f
                heap_node[heap_size] = nxt
                heap_size += 1
                i = heap_size - 1
                while i > 0:
                    p = (i - 1) // 2
                    if heap_f[i] < heap_f[p]:
                        tf = heap_f[i]; tn = heap_node[i]
                        heap_f[i] = heap_f[p]; heap_node[i] = heap_node[p]
                        heap_f[p] = tf; heap_node[p] = tn
                        i = p
                    else:
                        break

    return visited, found, parent


@njit(cache=True, boundscheck=False, fastmath=True)
def astar_5d(grid_flat,
             sc0, sc1, sc2, sc3, sc4,
             start_flat, goal_flat,
             g0, g1, g2, g3, g4,
             h_weight_times_100):
    """
    A* over 5D grid in flat representation.

    Returns (visited_count, parent_array) where parent_array[-1] can be
    traced from goal_flat back to start_flat.  If no path found,
    parent_array[goal_flat] == -1 (unless goal == start).
    """
    stride0 = sc1 * sc2 * sc3 * sc4
    stride1 = sc2 * sc3 * sc4
    stride2 = sc3 * sc4
    stride3 = sc4
    total = sc0 * stride0

    INF = np.iinfo(np.int32).max
    g_cost = np.full(total, INF, dtype=np.int32)
    parent = np.full(total, -1, dtype=np.int32)

    g_cost[start_flat] = 0

    # Binary heap stored as three parallel int64 arrays: f, counter, node.
    cap = 1024
    heap_f = np.empty(cap, dtype=np.int64)
    heap_node = np.empty(cap, dtype=np.int64)
    heap_f[0] = 0
    heap_node[0] = start_flat
    heap_size = 1

    visited = 0
    found = False

    while heap_size > 0:
        # Pop min
        f = heap_f[0]
        current = heap_node[0]
        heap_size -= 1
        if heap_size > 0:
            heap_f[0] = heap_f[heap_size]
            heap_node[0] = heap_node[heap_size]
            # Sift down
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                smallest = i
                if l < heap_size and heap_f[l] < heap_f[smallest]:
                    smallest = l
                if r < heap_size and heap_f[r] < heap_f[smallest]:
                    smallest = r
                if smallest == i:
                    break
                tf = heap_f[i]; tn = heap_node[i]
                heap_f[i] = heap_f[smallest]; heap_node[i] = heap_node[smallest]
                heap_f[smallest] = tf; heap_node[smallest] = tn
                i = smallest

        if current == goal_flat:
            found = True
            break

        cur_g = g_cost[current]
        # Stale heap entry? If f > cur_g + best heuristic, then cur was re-expanded.
        # Using consistent heuristic, we can safely skip stale entries whose f > cur_g + h(current).
        # Simpler: check if this was the best known - if cur_g differs from expected, skip.
        # For uniform-cost + consistent heuristic, once popped it is optimal; no need to check.
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

        # Expand 10 neighbors (5 dims, +/- 1)
        for dim in range(5):
            for delta in (-1, 1):
                if dim == 0:
                    nc0 = c0 + delta
                    if nc0 < 0 or nc0 >= sc0:
                        continue
                    nxt = current + delta * stride0
                elif dim == 1:
                    nc1 = c1 + delta
                    if nc1 < 0 or nc1 >= sc1:
                        continue
                    nxt = current + delta * stride1
                elif dim == 2:
                    nc2 = c2 + delta
                    if nc2 < 0 or nc2 >= sc2:
                        continue
                    nxt = current + delta * stride2
                elif dim == 3:
                    nc3 = c3 + delta
                    if nc3 < 0 or nc3 >= sc3:
                        continue
                    nxt = current + delta * stride3
                else:
                    nc4 = c4 + delta
                    if nc4 < 0 or nc4 >= sc4:
                        continue
                    nxt = current + delta

                if grid_flat[nxt] == 0:
                    continue
                if new_g >= g_cost[nxt]:
                    continue
                g_cost[nxt] = new_g
                parent[nxt] = current

                # Compute h at neighbor
                if dim == 0:
                    nn0 = c0 + delta; nn1 = c1; nn2 = c2; nn3 = c3; nn4 = c4
                elif dim == 1:
                    nn0 = c0; nn1 = c1 + delta; nn2 = c2; nn3 = c3; nn4 = c4
                elif dim == 2:
                    nn0 = c0; nn1 = c1; nn2 = c2 + delta; nn3 = c3; nn4 = c4
                elif dim == 3:
                    nn0 = c0; nn1 = c1; nn2 = c2; nn3 = c3 + delta; nn4 = c4
                else:
                    nn0 = c0; nn1 = c1; nn2 = c2; nn3 = c3; nn4 = c4 + delta

                h = abs(nn0 - g0) + abs(nn1 - g1) + abs(nn2 - g2) + abs(nn3 - g3) + abs(nn4 - g4)
                new_f = new_g + (h * h_weight_times_100) // 100

                # Grow heap if needed
                if heap_size == heap_f.shape[0]:
                    new_cap = heap_f.shape[0] * 2
                    nf = np.empty(new_cap, dtype=np.int64)
                    nn = np.empty(new_cap, dtype=np.int64)
                    for q in range(heap_size):
                        nf[q] = heap_f[q]
                        nn[q] = heap_node[q]
                    heap_f = nf
                    heap_node = nn

                # Push
                heap_f[heap_size] = new_f
                heap_node[heap_size] = nxt
                heap_size += 1
                # Sift up
                i = heap_size - 1
                while i > 0:
                    p = (i - 1) // 2
                    if heap_f[i] < heap_f[p]:
                        tf = heap_f[i]; tn = heap_node[i]
                        heap_f[i] = heap_f[p]; heap_node[i] = heap_node[p]
                        heap_f[p] = tf; heap_node[p] = tn
                        i = p
                    else:
                        break

    return visited, found, parent
