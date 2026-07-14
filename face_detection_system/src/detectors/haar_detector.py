"""
Traditional Face Detector — Haar Cascade + optional eye verification.

WHY USE THIS APPROACH
---------------------
* Zero deep-learning dependencies — runs on any CPU, including Raspberry Pi.
* Tiny footprint: the XML model is ~1 MB versus 10–100 MB for DNN models.
* Deterministic, no GPU required, easy to audit.
* Ideal for: low-power edge devices, simple demos, privacy-sensitive local use.

LIMITATIONS
-----------
* Higher false-positive rate than DNN methods.
* Sensitive to lighting, angle, and occlusion.
* Slower per-face than modern DNN detectors at equal accuracy targets.

ALGORITHM OVERVIEW (Viola–Jones, 2001)
---------------------------------------
1. Convert frame to grayscale and build an integral image (summed-area table).
2. Slide a detection window at multiple scales over the integral image.
3. At each position, evaluate a *cascade* of ~6 000 Haar-feature classifiers.
   If any stage rejects the window it is immediately discarded (fast rejection).
4. Surviving windows are candidate faces — apply NMS to merge overlapping boxes.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDetector, Detection


class HaarCascadeDetector(BaseDetector):
    """
    Multi-scale Haar Cascade face detector with optional eye-based validation.

    Parameters
    ----------
    frontal_model_path : path to haarcascade_frontalface_default.xml
    profile_model_path : optional path to haarcascade_profileface.xml
    eye_model_path     : optional path to haarcascade_eye.xml
    confidence_threshold : minimum confidence to keep a detection (mapped from
                           neighbour count normalised to [0, 1])
    scale_factor       : image pyramid scale between levels (e.g. 1.1 = 10 % shrink)
    min_neighbors      : how many overlapping candidates must agree (reduces FP rate)
    min_face_size      : smallest face bounding box to consider (pixels)
    use_eye_validation : if True, discard face candidates with no detected eyes
    """

    def __init__(
        self,
        frontal_model_path: str,
        profile_model_path: Optional[str] = None,
        eye_model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30),
        use_eye_validation: bool = False,
    ):
        super().__init__(confidence_threshold)

        # --- Load frontal cascade (required) ---
        if not os.path.isfile(frontal_model_path):
            raise FileNotFoundError(
                f"Frontal cascade not found: {frontal_model_path}\n"
                "Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades"
            )
        self._frontal = cv2.CascadeClassifier(frontal_model_path)
        if self._frontal.empty():
            raise RuntimeError(f"Failed to load cascade: {frontal_model_path}")

        # --- Load optional profile cascade ---
        self._profile: Optional[cv2.CascadeClassifier] = None
        if profile_model_path and os.path.isfile(profile_model_path):
            self._profile = cv2.CascadeClassifier(profile_model_path)
            if self._profile.empty():
                self._profile = None

        # --- Load optional eye cascade (for validation) ---
        self._eye: Optional[cv2.CascadeClassifier] = None
        if eye_model_path and os.path.isfile(eye_model_path):
            self._eye = cv2.CascadeClassifier(eye_model_path)
            if self._eye.empty():
                self._eye = None

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.use_eye_validation = use_eye_validation and (self._eye is not None)

    # ------------------------------------------------------------------ #
    # Core detection
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = self._preprocess(frame)

        # Frontal detection
        detections = self._run_cascade(self._frontal, gray, "face_frontal")

        # Profile detection (merge results)
        if self._profile is not None:
            profile_dets = self._run_cascade(self._profile, gray, "face_profile")
            # Mirror the frame and run profile cascade again (catches right-profile)
            flipped = cv2.flip(gray, 1)
            w = gray.shape[1]
            for d in self._run_cascade(self._profile, flipped, "face_profile"):
                # Mirror x-coordinate back to original space
                d.x = w - d.x - d.w
                profile_dets.append(d)
            detections.extend(profile_dets)

        # Optional eye verification — filters out many false positives
        if self.use_eye_validation:
            detections = [
                d for d in detections if self._has_eyes(frame, gray, d)
            ]

        # Final NMS across frontal + profile results
        detections = _nms(detections, iou_threshold=0.3)

        return [d for d in detections if d.confidence >= self.confidence_threshold]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale and apply histogram equalisation."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        # CLAHE: contrast-limited adaptive histogram equalisation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _run_cascade(
        self,
        cascade: cv2.CascadeClassifier,
        gray: np.ndarray,
        label: str,
    ) -> List[Detection]:
        """
        Run detectMultiScale3 which returns neighbour counts alongside boxes.
        The neighbour count is a crude proxy for confidence.
        """
        try:
            faces, reject_levels, level_weights = cascade.detectMultiScale3(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                outputRejectLevels=True,
            )
        except cv2.error:
            # Fallback for OpenCV builds that don't support detectMultiScale3
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
            )
            level_weights = None

        results: List[Detection] = []
        if len(faces) == 0:
            return results

        for i, (x, y, w, h) in enumerate(faces):
            # Map neighbour weight to [0, 1] confidence heuristically
            if level_weights is not None and len(level_weights) > i:
                raw = float(level_weights[i])
                conf = min(1.0, raw / (raw + 10.0))  # sigmoid-like squash
            else:
                conf = 0.6  # fallback when weights unavailable
            results.append(Detection(int(x), int(y), int(w), int(h), conf, label=label))

        return results

    def _has_eyes(
        self, frame: np.ndarray, gray: np.ndarray, det: Detection
    ) -> bool:
        """Return True if at least one eye is found in the upper half of the face ROI."""
        roi_gray = gray[det.y : det.y2, det.x : det.x2]
        # Look only in upper half of face (eyes are above the nose)
        upper = roi_gray[: roi_gray.shape[0] // 2, :]
        eyes = self._eye.detectMultiScale(
            upper, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
        )
        return len(eyes) >= 1


# ------------------------------------------------------------------ #
# Non-Maximum Suppression (NMS)
# ------------------------------------------------------------------ #
def _nms(detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
    """
    Greedy NMS: sort by confidence descending, suppress overlapping boxes.
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    suppressed = [False] * len(detections)
    for i, det in enumerate(detections):
        if suppressed[i]:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if not suppressed[j] and det.iou(detections[j]) > iou_threshold:
                suppressed[j] = True
    return kept
