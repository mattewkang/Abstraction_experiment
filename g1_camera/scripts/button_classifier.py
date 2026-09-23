"""Button-crop classifier (YOLOv8 cls).

Trained on 13 elevator-button classes:
  1, B1, B2, B3, B4, B5, B6, close, down, emergency_yellow, open, phone, up.

Loads the ultralytics ``.pt`` weights directly with torch. ROS-free, so it is
also usable from offline scripts.

Usage:
  cls = ButtonClassifier("models/button_cls.pt", device="cuda:0")
  label, score = cls.predict(bgr_crop)            # any HxWx3 uint8 BGR
  pairs = cls.predict_batch([bgr1, bgr2, ...])    # batched, same order
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


class ButtonClassifier:
    """Classify a button-bbox crop into one of the trained classes."""

    def __init__(self, weights_path: str, device: str = "cpu", imgsz: int = 128):
        import torch
        from ultralytics import YOLO
        if not os.path.exists(weights_path):
            raise FileNotFoundError(weights_path)
        self._torch = torch
        self.weights_path = weights_path
        self.device = device
        self._yolo = YOLO(weights_path, task="classify")
        names = self._yolo.model.names
        self.class_names: List[str] = [names[i] for i in sorted(names)]
        train_args = getattr(self._yolo.model, "args", None) or {}
        self.imgsz: int = (int(train_args.get("imgsz", imgsz))
                           if isinstance(train_args, dict) else imgsz)
        self._yolo.model.to(device).eval()
        if str(device).startswith("cuda"):
            # Warmup. Some CUDA + cuDNN combinations fail on the first call with
            # CUDNN_STATUS_NOT_INITIALIZED; disabling cuDNN at that point keeps
            # the model usable on CUDA without forcing a CPU fallback.
            try:
                with torch.inference_mode():
                    self._yolo.predict(
                        np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
                        imgsz=self.imgsz, device=device, verbose=False,
                    )
            except RuntimeError as e:
                if "CUDNN_STATUS_NOT_INITIALIZED" in str(e):
                    torch.backends.cudnn.enabled = False
                else:
                    raise

    def _is_valid(self, crop) -> bool:
        return (crop is not None
                and crop.size > 0
                and crop.shape[0] >= 4
                and crop.shape[1] >= 4)

    def predict(self, bgr_crop: np.ndarray) -> Tuple[str, float]:
        """Return (class_name, softmax_prob). Tiny / empty crops -> ('unknown', 0.0)."""
        if not self._is_valid(bgr_crop):
            return "unknown", 0.0
        with self._torch.inference_mode():
            r = self._yolo.predict(
                bgr_crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return self.class_names[int(r.probs.top1)], float(r.probs.top1conf)

    def predict_batch(self, bgr_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batched. Returns one (name, prob) per input, same order. Invalid
        crops get ('unknown', 0.0) without touching the model."""
        if not bgr_crops:
            return []
        keep = [i for i, c in enumerate(bgr_crops) if self._is_valid(c)]
        out: List[Tuple[str, float]] = [("unknown", 0.0)] * len(bgr_crops)
        if not keep:
            return out
        batch = [bgr_crops[i] for i in keep]
        with self._torch.inference_mode():
            results = self._yolo.predict(
                batch, imgsz=self.imgsz, device=self.device, verbose=False)
        for j, i in enumerate(keep):
            r = results[j]
            out[i] = (self.class_names[int(r.probs.top1)], float(r.probs.top1conf))
        return out
