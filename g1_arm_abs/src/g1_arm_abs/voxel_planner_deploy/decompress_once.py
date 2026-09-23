"""
Run once after extracting the deploy package.

Decompresses models/abstraction_map/voxel_config_bitmap_{left,right}.uint8.gz
into the raw .uint8 files that VoxelReachability memmaps. The .gz files
can be deleted afterwards to save disk.

    python decompress_once.py
"""

import gzip
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Bitmaps live under the package's models/ tree, not next to this script.
# Path: voxel_planner_deploy/ -> g1_arm_abs/ -> src/ -> package root.
_PACKAGE_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(_PACKAGE_ROOT, "models", "abstraction_map")


def decompress(arm):
    gz = os.path.join(OUT, f"voxel_config_bitmap_{arm}.uint8.gz")
    raw = os.path.join(OUT, f"voxel_config_bitmap_{arm}.uint8")
    if os.path.exists(raw) and os.path.getsize(raw) > 1_000_000_000:
        print(f"[{arm}] already decompressed: {raw}")
        return
    if not os.path.exists(gz):
        print(f"[{arm}] source not found: {gz}")
        return
    t0 = time.time()
    print(f"[{arm}] decompressing {gz} -> {raw} ...")
    with gzip.open(gz, "rb") as fin, open(raw, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    sz = os.path.getsize(raw)
    print(f"[{arm}] wrote {sz/1e9:.2f} GB in {time.time()-t0:.1f} s")


def main():
    arms = sys.argv[1:] or ["left", "right"]
    for arm in arms:
        decompress(arm)
    print("\nDone. You may delete the .uint8.gz files to free disk space.")


if __name__ == "__main__":
    main()
