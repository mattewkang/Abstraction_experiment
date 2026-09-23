"""Startup sha256 verification for HF-pinned weights in g1_arm_abs/models/.

Mirrors g1_camera's verify_pinned_weight() with two adaptations:
- Manifest lookup is keyed by relative-to-models path, because the nested
  model tree has duplicate basenames (best_model.pth and x_scaler.pkl appear
  under grasp_bottle/{left,right}/ and elevator_button/{left,right}/).
- Manifest entries may opt out of the sha256 check via `verify_sha256: false`
  (default true). Used today only by the two ~14 GB voxel_config_bitmap files
  whose full sha256 would add ~30 s to node startup; the existence check
  still runs. `force_sha256_all=True` overrides every opt-out.

Missing manifest, missing weight file, or sha256 drift triggers
rospy.logfatal + sys.exit(1) so failures surface before any service is
advertised.
"""

import hashlib
import os
import sys

import rospy
import yaml


_PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_MANIFEST_PATH = os.path.join(_PACKAGE_ROOT, "config", "model_weights.yaml")
_MODELS_DIR = os.path.join(_PACKAGE_ROOT, "models")


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fatal(msg, *args):
    rospy.logfatal(msg, *args)
    sys.exit(1)


def verify_pinned_weights(force_sha256_all=False):
    """Verify every entry in config/model_weights.yaml against on-disk files.

    Args:
        force_sha256_all: when False (default), entries with
            `verify_sha256: false` in the manifest get existence-only
            checks; when True, every entry is sha256d regardless of the
            manifest flag.
    """
    try:
        with open(_MANIFEST_PATH) as fh:
            spec = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        _fatal(
            "model_weights.yaml missing at %s; run "
            "`rosrun g1_arm_abs fetch_weights.py` first.",
            _MANIFEST_PATH,
        )

    weights = spec.get("weights") or {}
    if not weights:
        _fatal(
            "model_weights.yaml at %s has no weights section.",
            _MANIFEST_PATH,
        )

    skipped = []
    verified = 0
    for name, entry in weights.items():
        if not isinstance(entry, dict):
            _fatal("Manifest entry '%s' is not a mapping.", name)
        rel = entry.get("filename", "")
        expected = entry.get("sha256", "")
        if not rel or not expected:
            _fatal(
                "Manifest entry '%s' missing filename or sha256.", name,
            )
        verify_flag = entry.get("verify_sha256", True)
        if not isinstance(verify_flag, bool):
            _fatal(
                "Manifest entry '%s' has non-bool verify_sha256: %r",
                name, verify_flag,
            )
        target = os.path.join(_MODELS_DIR, rel)
        if not os.path.exists(target):
            _fatal(
                "Pinned weight missing: %s. Run "
                "`rosrun g1_arm_abs fetch_weights.py` to download from "
                "HF Hub.",
                target,
            )
        if not verify_flag and not force_sha256_all:
            skipped.append(rel)
            continue
        digest = _sha256_of(target)
        if digest != expected:
            _fatal(
                "Pinned weight sha256 mismatch: %s "
                "(expected %s, got %s). "
                "Re-run fetch_weights.py to reconcile.",
                target, expected, digest,
            )
        verified += 1

    if skipped:
        rospy.loginfo(
            "weight_verify: verified %d files; existence-only on "
            "%d entries (manifest verify_sha256:false): %s. "
            "Set ~force_full_weight_sha256:=true to override.",
            verified, len(skipped), ", ".join(skipped),
        )
    else:
        rospy.loginfo(
            "weight_verify: verified %d files (full sha256).", verified,
        )


__all__ = ["verify_pinned_weights"]
