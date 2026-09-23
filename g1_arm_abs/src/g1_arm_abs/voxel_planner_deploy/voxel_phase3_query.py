"""
Phase 3: online reachable-from-obstacle query.

Given a set of occupied voxels in the workspace (representing arbitrary
obstacles), compute which joint configurations are collision-free.

Core operation (per arm):
    forbidden_bitmap = OR over v in obstacle_voxels of voxel_config_bitmap[v]
    reachable_mask   = feasibility_mask AND NOT forbidden_bitmap

Each bit in the returned mask corresponds to one config (linear index into
the 5D joint grid at 4 deg step). The mapping from linear index to joint
degrees is in voxel_phase2_<arm>.json -> deg_lists / sample_counts.

Usage (library):
    from voxel_phase3_query import VoxelReachability
    vr = VoxelReachability("left")
    mask = vr.reachable_mask(obstacle_voxels=[(ix, iy, iz), ...])
    config_indices = np.where(np.unpackbits(mask)[:vr.total_configs])[0]
    joints = vr.config_index_to_degs(config_indices)
"""

import json
import os

import numpy as np

try:
    from voxel_numba_kernels import or_reduce_runs as _or_reduce_nb
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Bitmaps live under the package's models/ tree, not next to this script.
# Path: voxel_planner_deploy/ -> g1_arm_abs/ -> src/ -> package root.
_PACKAGE_ROOT = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(_PACKAGE_ROOT, "models", "abstraction_map")


class VoxelReachability:
    def __init__(self, arm, output_dir=None, preload=False, cache_size=4,
                 use_numba=True):
        self.arm = arm
        output_dir = output_dir or OUTPUT_DIR
        meta_path = os.path.join(output_dir, f"voxel_phase2_{arm}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # LRU cache for forbidden masks keyed by obstacle voxel-set hash.
        self._cache_size = int(cache_size)
        self._mask_cache = {}      # hash -> mask uint8[bytes_per_voxel]
        self._cache_order = []     # list of hashes in LRU order
        self.use_numba = bool(use_numba) and _HAS_NUMBA
        self.meta = meta
        self.total_configs = int(meta["total_configs"])
        self.bytes_per_voxel = int(meta["bytes_per_voxel"])  # COMPACT row width
        self.grid_shape = np.array(meta["grid_shape"], dtype=int)
        self.grid_origin = np.array(meta["grid_origin"], dtype=float)
        self.voxel_size = float(meta["voxel_size"])
        self.num_voxels = int(meta["num_voxels"])
        self.sample_counts = list(meta["sample_counts"])
        self.deg_lists = [np.array(dl, dtype=float) for dl in meta["deg_lists"]]

        # New compact-storage fields.
        self.storage = meta.get("storage", "dense")
        self.num_sparse_voxels = int(meta.get("num_sparse_voxels", self.num_voxels))
        self.N_feasible = int(meta.get("N_feasible", self.total_configs))
        bytes_per_voxel_full = (self.total_configs + 7) // 8
        self.bytes_per_voxel_full = int(meta.get("bytes_per_voxel_full", bytes_per_voxel_full))

        # Resolve data paths: prefer files next to the metadata JSON.
        meta_dir = os.path.dirname(os.path.abspath(meta_path))
        def _resolve(name, fallback_key):
            local = os.path.join(meta_dir, name)
            if os.path.exists(local):
                return local
            return meta[fallback_key]
        bitmap_path = _resolve(f"voxel_config_bitmap_{arm}.uint8", "bitmap_path")
        feasibility_path = _resolve(f"feasibility_mask_{arm}.uint8", "feasibility_path")

        mm = np.memmap(bitmap_path, dtype=np.uint8, mode="r",
                       shape=(self.num_sparse_voxels, self.bytes_per_voxel))
        if preload:
            self.bitmap = np.array(mm, dtype=np.uint8)
        else:
            self.bitmap = mm

        self.feasibility = np.fromfile(feasibility_path, dtype=np.uint8)
        if self.feasibility.size != self.bytes_per_voxel_full:
            raise RuntimeError(
                f"feasibility size {self.feasibility.size} != expected "
                f"{self.bytes_per_voxel_full}"
            )

        # Compact storage: load side tables.
        if self.storage == "compact":
            feas_idx_path = _resolve(f"feasible_indices_{arm}.int32",
                                     "feasible_indices_path")
            sparse_map_path = _resolve(f"sparse_voxel_map_{arm}.int32",
                                       "sparse_voxel_map_path")
            self.feasible_indices = np.fromfile(feas_idx_path, dtype=np.int32)
            if self.feasible_indices.size != self.N_feasible:
                raise RuntimeError(
                    f"feasible_indices size {self.feasible_indices.size} != "
                    f"expected {self.N_feasible}"
                )
            self.sparse_voxel_map = np.fromfile(sparse_map_path, dtype=np.int32)
            if self.sparse_voxel_map.size != self.num_voxels:
                raise RuntimeError(
                    f"sparse_voxel_map size {self.sparse_voxel_map.size} != "
                    f"expected {self.num_voxels}"
                )
        else:
            self.feasible_indices = None
            self.sparse_voxel_map = None

        nx, ny, nz = [int(v) for v in self.grid_shape]
        self._ny_nz = ny * nz
        self._nz = nz

        if self.use_numba:
            self._warmup_numba()

    def _warmup_numba(self):
        dummy_lin = np.array([0], dtype=np.int64)
        dummy_out = np.zeros(self.bytes_per_voxel, dtype=np.uint8)
        _or_reduce_nb(self.bitmap, dummy_lin, dummy_out)

    # ---- coordinate helpers -------------------------------------------------

    def world_to_voxel(self, point_xyz):
        """Map a 3D point (torso_link frame) to (ix, iy, iz); out of grid -> None."""
        p = np.asarray(point_xyz, dtype=float)
        idx = np.floor((p - self.grid_origin) / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.grid_shape):
            return None
        return tuple(int(v) for v in idx)

    def voxels_in_box(self, xyz_min, xyz_max):
        """Return voxel (ix,iy,iz) indices inside an axis-aligned box."""
        lo = np.maximum(
            np.floor((np.asarray(xyz_min) - self.grid_origin) / self.voxel_size).astype(int),
            0,
        )
        hi = np.minimum(
            np.floor((np.asarray(xyz_max) - self.grid_origin) / self.voxel_size).astype(int),
            self.grid_shape - 1,
        )
        if np.any(lo > hi):
            return np.empty((0, 3), dtype=int)
        ix = np.arange(lo[0], hi[0] + 1)
        iy = np.arange(lo[1], hi[1] + 1)
        iz = np.arange(lo[2], hi[2] + 1)
        ii, jj, kk = np.meshgrid(ix, iy, iz, indexing="ij")
        return np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)

    def voxel_linear_index(self, voxel_ijk):
        """(ix,iy,iz) -> linear index. Accepts (3,) or (N,3)."""
        a = np.asarray(voxel_ijk, dtype=int)
        if a.ndim == 1:
            return int(a[0] * self._ny_nz + a[1] * self._nz + a[2])
        return a[:, 0] * self._ny_nz + a[:, 1] * self._nz + a[:, 2]

    # ---- reachability -------------------------------------------------------

    def forbidden_mask(self, voxel_indices):
        """OR the config bitmaps of the given voxels. Returns a COMPACT-space
        forbidden bitmap (length = ceil(N_feasible/8) bytes).

        Strategy: map dense voxel linear indices to sparse bitmap slots (if in
        compact storage), drop out-of-reach voxels. Then OR the corresponding
        bitmap rows using numba / numpy SIMD reduce.
        """
        if voxel_indices is None or len(voxel_indices) == 0:
            return np.zeros(self.bytes_per_voxel, dtype=np.uint8)

        lin = np.asarray(voxel_indices)
        if lin.ndim == 2:
            lin = self.voxel_linear_index(lin)
        lin = np.unique(np.asarray(lin, dtype=np.int64))
        valid = (lin >= 0) & (lin < self.num_voxels)
        lin = lin[valid]

        # Compact storage: map dense → sparse slot, drop -1.
        if self.sparse_voxel_map is not None:
            slots = self.sparse_voxel_map[lin]
            slots = slots[slots >= 0].astype(np.int64)
            slots.sort()
            lin = slots

        if lin.size == 0:
            return np.zeros(self.bytes_per_voxel, dtype=np.uint8)

        # LRU cache lookup: hash the sorted, unique obstacle voxel-index set.
        if self._cache_size > 0:
            key = hash(lin.tobytes())
            cached = self._mask_cache.get(key)
            if cached is not None:
                self._cache_order.remove(key)
                self._cache_order.append(key)
                return cached.copy()

        out = np.zeros(self.bytes_per_voxel, dtype=np.uint8)

        if self.use_numba:
            # Numba path works directly on rows; needs a C-contiguous uint8 view.
            bm = self.bitmap
            if isinstance(bm, np.memmap):
                # np.asarray on memmap returns the same memmap; numba accepts
                # contiguous uint8 arrays including memmaps.
                pass
            _or_reduce_nb(bm, lin.astype(np.int64), out)
        else:
            # Identify contiguous runs in the sorted linear indices.
            diff = np.diff(lin)
            gaps = np.where(diff > 1)[0]
            run_starts = np.concatenate([[0], gaps + 1])
            run_ends = np.concatenate([gaps + 1, [lin.size]])
            for s, e in zip(run_starts, run_ends):
                first = int(lin[s])
                last = int(lin[e - 1])
                if last == first:
                    np.bitwise_or(out, self.bitmap[first], out=out)
                else:
                    block = self.bitmap[first:last + 1]
                    out |= np.bitwise_or.reduce(block, axis=0)

        if self._cache_size > 0:
            self._mask_cache[key] = out
            self._cache_order.append(key)
            while len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._mask_cache.pop(old, None)
            return out.copy()
        return out

    def reachable_mask(self, voxel_indices=None, world_obstacles=None):
        """
        Return a uint8 bitmap (bytes_per_voxel) where bit i = 1 iff config i
        is feasible (no self collision) AND does not collide with any voxel in
        the obstacle set.

        voxel_indices: list of (ix,iy,iz) tuples / (N,3) array / linear indices.
        world_obstacles: list of (xyz_min, xyz_max) AABBs in torso_link frame.
        """
        occ = []
        if voxel_indices is not None:
            occ.append(np.asarray(voxel_indices))
        if world_obstacles is not None:
            for (xyz_min, xyz_max) in world_obstacles:
                v = self.voxels_in_box(xyz_min, xyz_max)
                if v.size:
                    occ.append(v)
        if occ:
            merged = np.concatenate([a.reshape(-1, 3) if a.ndim == 2 else a for a in occ])
            if merged.ndim == 2:
                lin = self.voxel_linear_index(merged)
            else:
                lin = merged.astype(np.int64)
        else:
            lin = np.empty(0, dtype=np.int64)

        forbidden_compact = self.forbidden_mask(lin)

        # Expand compact forbidden bitmap (one bit per FEASIBLE config) back
        # to the full config-index space so that callers always see a bitmap
        # of length ceil(total_configs / 8).
        if self.sparse_voxel_map is not None:
            compact_bits = np.unpackbits(forbidden_compact,
                                         bitorder="little")[:self.N_feasible]
            forbidden_full_bits = np.zeros(self.total_configs, dtype=np.uint8)
            forbidden_full_bits[self.feasible_indices] = compact_bits
            forbidden = np.packbits(forbidden_full_bits, bitorder="little")
            # Pad to bytes_per_voxel_full if packbits produced a shorter array.
            if forbidden.size < self.bytes_per_voxel_full:
                pad = np.zeros(self.bytes_per_voxel_full - forbidden.size, dtype=np.uint8)
                forbidden = np.concatenate([forbidden, pad])
        else:
            forbidden = forbidden_compact

        reachable = self.feasibility & (~forbidden)
        # Zero out trailing bits beyond total_configs
        tail = self.total_configs & 7
        if tail:
            reachable[-1] &= np.uint8((1 << tail) - 1)
        return reachable

    # ---- config index <-> joint values --------------------------------------

    def config_index_to_degs(self, config_indices):
        """Map linear config indices to joint degree tuples."""
        a = np.atleast_1d(np.asarray(config_indices, dtype=np.int64))
        coords = np.empty((a.size, len(self.sample_counts)), dtype=np.int64)
        rem = a.copy()
        for d in range(len(self.sample_counts) - 1, -1, -1):
            s = self.sample_counts[d]
            coords[:, d] = rem % s
            rem //= s
        degs = np.stack(
            [self.deg_lists[d][coords[:, d]] for d in range(len(self.sample_counts))],
            axis=-1,
        )
        return degs if degs.shape[0] > 1 else degs[0]

    def degs_to_config_index(self, degs):
        """Map joint degree values (tuple / (N,5) array) to linear config index."""
        a = np.atleast_2d(np.asarray(degs, dtype=float))
        idx = np.zeros(a.shape[0], dtype=np.int64)
        for d in range(len(self.sample_counts)):
            col = a[:, d]
            pos = np.searchsorted(self.deg_lists[d], col)
            # Exact-match check: value must round to a valid grid sample
            pos = np.clip(pos, 0, len(self.deg_lists[d]) - 1)
            idx = idx * self.sample_counts[d] + pos
        return idx if idx.size > 1 else int(idx[0])

    def reachable_config_indices(self, mask):
        """uint8 bitmap -> array of config indices where bit is 1."""
        bits = np.unpackbits(mask, bitorder="little")[:self.total_configs]
        return np.where(bits)[0]


def _demo():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--table_z_min", type=float, default=-0.13)
    parser.add_argument("--table_z_max", type=float, default=-0.09)
    parser.add_argument("--table_x_min", type=float, default=0.20)
    parser.add_argument("--table_x_max", type=float, default=0.40)
    parser.add_argument("--table_y_min", type=float, default=-0.40)
    parser.add_argument("--table_y_max", type=float, default=0.40)
    args = parser.parse_args()

    vr = VoxelReachability(args.arm)
    print(f"grid_shape={vr.grid_shape.tolist()} voxels={vr.num_voxels} "
          f"configs={vr.total_configs}")

    # Table as axis-aligned box
    obstacle = [
        ((args.table_x_min, args.table_y_min, args.table_z_min),
         (args.table_x_max, args.table_y_max, args.table_z_max)),
    ]
    mask = vr.reachable_mask(world_obstacles=obstacle)
    reachable = vr.reachable_config_indices(mask)

    feasible_bits = np.unpackbits(vr.feasibility, bitorder="little")[:vr.total_configs]
    print(f"feasible (no-self-coll) configs     = {int(feasible_bits.sum())}")
    print(f"reachable with obstacle             = {reachable.size}")
    print(f"fraction remaining                  = "
          f"{reachable.size / max(1, int(feasible_bits.sum())):.3f}")


if __name__ == "__main__":
    _demo()
