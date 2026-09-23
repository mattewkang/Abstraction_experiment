#!/usr/bin/env python3
"""Download model weights pinned in g1_camera/config/model_weights.yaml from HF Hub.

Idempotent: skip re-download when the on-disk file already matches the pinned sha256.
HF in-repo path may be flat or nested; the downloaded file is flattened to
<package>/models/<basename> to match launch-file references.
"""

import hashlib
import os
import pathlib
import shutil
import sys

import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "model_weights.yaml"
DEST_DIR = REPO_ROOT / "models"


def _sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    spec = yaml.safe_load(CONFIG_PATH.read_text())
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for name, entry in spec["weights"].items():
        local_name = pathlib.Path(entry["filename"]).name
        target = DEST_DIR / local_name
        if target.exists() and _sha256_of(target) == entry["sha256"]:
            print(f"[fetch_weights] {name}: cached at {target}")
            continue
        try:
            src = hf_hub_download(
                repo_id=spec["hf_repo"],
                filename=entry["filename"],
                revision=spec["hf_revision"],
                token=token,
            )
        except HfHubHTTPError as err:
            print(
                f"[fetch_weights] {name}: download failed ({err}). "
                "Verify HF_TOKEN or run `hf auth login`.",
                file=sys.stderr,
            )
            return 1
        shutil.copyfile(src, target)
        digest = _sha256_of(target)
        if digest != entry["sha256"]:
            print(
                f"[fetch_weights] {name}: sha256 mismatch "
                f"(expected {entry['sha256']}, got {digest})",
                file=sys.stderr,
            )
            return 1
        print(f"[fetch_weights] {name}: downloaded -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
