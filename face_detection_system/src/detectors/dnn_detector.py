"""
Deep Learning Face Detector — OpenCV DNN (ResNet-10 SSD).

WHY USE THIS APPROACH
---------------------
* Single Shot Detector (SSD) backbone with ResNet-10 feature extractor.
* Trained on a large face dataset — handles varied poses, lighting, occlusion.
* ~5–20 ms per frame on a modern CPU; <5 ms with GPU (CUDA/OpenCL).
* Included in OpenCV ≥ 3.3 — no extra heavy framework (PyTorch, TensorFlow).
* Best general-purpose choice for desktop / server / embedded GPU scenarios.

ALGORITHM OVERVIEW (SSD, Liu et al., 2016)
-------------------------------------------
1. Feed the image through a truncated ResNet-10 feature network.
2. At multiple feature map scales, predict (class, offset) pairs for
   pre-defined anchor boxes ("default boxes").
3. Apply confidence threshold to class predictions.
4. Apply NMS to merge overlapping high-confidence boxes.
5. Map offsets back to pixel coordinates.

MODEL FILES
-----------
The detector can use one of two pre-trained models:
  A) OpenCV's built-in Caffe SSD model (automatically downloaded):
       deploy.prototxt  +  res10_300x300_ssd_iter_140000.caffemodel
  B) Any ONNX face-detection model (pass onnx_model_path).

AUTO-DOWNLOAD
-------------
If the Caffe model files are absent, the constructor downloads them from the
official OpenCV GitHub release (≈10 MB total) and caches them next to the
executable. Pass download=False to disable this behaviour.
"""
from __future__ import annotations

import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDetector, Detection

# --------------------------------------------------------------------- #
# Model download URLs (official OpenCV GitHub releases)
# --------------------------------------------------------------------- #
_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector"
    "_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


class DNNDetector(BaseDetector):
    """
    ResNet-10 SSD face detector via OpenCV's DNN module.

    Parameters
    ----------
    model_dir          : directory to store / find model files
    onnx_model_path    : optional path to an ONNX model (overrides Caffe model)
    confidence_threshold : suppress detections below this score (0–1)
    input_size         : spatial resolution fed into the network (width, height)
    mean               : BGR pixel mean subtracted during pre-processing
    scale              : pixel scale factor (1/255 for normalised [0,1] models)
    backend            : OpenCV DNN backend constant (cv2.dnn.DNN_BACKEND_*)
    target             : OpenCV DNN target (cv2.dnn.DNN_TARGET_*)
    nms_threshold      : IoU threshold for suppression of overlapping boxes
    download           : automatically download Caffe model if files missing
    """

    def __init__(
        self,
        model_dir: str = "models",
        onnx_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (300, 300),
        mean: Tuple[float, float, float] = (104.0, 177.0, 123.0),
        scale: float = 1.0,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
        nms_threshold: float = 0.4,
        download: bool = True,
    ):
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.mean = mean
        self.scale = scale
        self.nms_threshold = nms_threshold

        # ---- Load model ----
        if onnx_model_path and os.path.isfile(onnx_model_path):
            self._net = cv2.dnn.readNetFromONNX(onnx_model_path)
            self._model_type = "onnx"
        else:
            proto, weights = self._ensure_caffe_model(model_dir, download)
            self._net = cv2.dnn.readNetFromCaffe(proto, weights)
            self._model_type = "caffe_ssd"

        self._net.setPreferableBackend(backend)
        self._net.setPreferableTarget(target)

    # ------------------------------------------------------------------ #
    # Core detection
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        # Build 4-D blob: mean subtraction + resize
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.scale,
            size=self.input_size,
            mean=self.mean,
            swapRB=False,  # frame is already BGR
            crop=False,
        )
        self._net.setInput(blob)
        # Output shape: (1, 1, N, 7)
        # Each detection row: [_, class_id, confidence, x1, y1, x2, y2]
        raw = self._net.forward()

        candidates: List[Detection] = []
        for det in raw[0, 0]:
            conf = float(det[2])
            if conf < self.confidence_threshold:
                continue
            # Coordinates are in [0, 1] relative to input blob size
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            candidates.append(Detection(x1, y1, bw, bh, conf))

        # NMS is already baked into SSD, but run again if models overlap
        return _soft_nms(candidates, iou_threshold=self.nms_threshold)

    # ------------------------------------------------------------------ #
    # Resource management
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        # OpenCV DNN has no explicit destructor — GC handles it
        self._net = None

    # ------------------------------------------------------------------ #
    # Model download helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_caffe_model(model_dir: str, download: bool) -> Tuple[str, str]:
        os.makedirs(model_dir, exist_ok=True)
        proto = os.path.join(model_dir, "deploy.prototxt")
        weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if os.path.isfile(proto) and os.path.isfile(weights):
            return proto, weights

        if not download:
            raise FileNotFoundError(
                f"DNN model files not found in '{model_dir}'. "
                "Pass download=True or supply the files manually."
            )

        print("[DNNDetector] Downloading ResNet-10 SSD face model (~10 MB)…")
        try:
            _download(proto, _PROTOTXT_URL)
            _download(weights, _CAFFEMODEL_URL)
            print("[DNNDetector] Download complete.")
        except Exception as exc:
            raise RuntimeError(
                f"Model download failed: {exc}\n"
                "Please manually place deploy.prototxt and "
                "res10_300x300_ssd_iter_140000.caffemodel "
                f"in '{model_dir}'."
            ) from exc

        return proto, weights


# --------------------------------------------------------------------- #
# Soft-NMS (Bodla et al., 2017)
# --------------------------------------------------------------------- #
def _soft_nms(
    detections: List[Detection],
    iou_threshold: float = 0.4,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
) -> List[Detection]:
    """
    Soft-NMS: instead of hard-suppressing overlapping boxes, *decay* their
    confidence using a Gaussian penalty, then threshold.

    Advantages over hard NMS: better recall in crowd / occlusion scenarios.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    # Make mutable copies of confidences
    scores = [d.confidence for d in dets]

    kept: List[Detection] = []
    while dets:
        best_idx = int(np.argmax(scores))
        best = dets.pop(best_idx)
        best_score = scores.pop(best_idx)
        if best_score < score_threshold:
            break
        best.confidence = best_score
        kept.append(best)
        new_dets, new_scores = [], []
        for d, s in zip(dets, scores):
            iou = best.iou(d)
            # Gaussian decay
            s = s * np.exp(-(iou ** 2) / sigma)
            if s >= score_threshold:
                new_dets.append(d)
                new_scores.append(s)
        dets, scores = new_dets, new_scores

    return kept


# --------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------- #
def _download(dest: str, url: str, timeout: int = 30) -> None:
    """Download *url* to *dest* with a progress indicator."""
    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded // total_size)
            print(f"\r  {pct:3d}% ({downloaded // 1024} KB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress
