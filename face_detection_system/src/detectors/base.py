"""
Base classes for all face detectors.

Defines the common interface that every detector must implement,
and the Detection dataclass used to pass results through the pipeline.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """
    Single face detection result.

    Attributes
    ----------
    x, y        : top-left corner of bounding box (pixels)
    w, h        : width and height of bounding box (pixels)
    confidence  : detector confidence score in [0, 1]
    landmarks   : optional list of (x, y) landmark points
    label       : optional string label (e.g. "face")
    """
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    label: str = "face"

    # ------------------------------------------------------------------ #
    # Derived geometry helpers
    # ------------------------------------------------------------------ #
    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) format."""
        return (self.x, self.y, self.w, self.h)

    def iou(self, other: "Detection") -> float:
        """Compute Intersection-over-Union with another Detection."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def crop(self, image: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """
        Crop the face region from *image*, optionally with proportional padding.

        Parameters
        ----------
        image   : BGR numpy array (H x W x 3)
        padding : fraction of bbox size to add on each side (e.g. 0.2 = 20 %)
        """
        h_img, w_img = image.shape[:2]
        pad_x = int(self.w * padding)
        pad_y = int(self.h * padding)
        x1 = max(0, self.x - pad_x)
        y1 = max(0, self.y - pad_y)
        x2 = min(w_img, self.x2 + pad_x)
        y2 = min(h_img, self.y2 + pad_y)
        return image[y1:y2, x1:x2].copy()

    def __repr__(self) -> str:
        return (
            f"Detection(bbox=({self.x},{self.y},{self.w},{self.h}), "
            f"conf={self.confidence:.3f})"
        )


class BaseDetector(ABC):
    """
    Abstract base class for every face detector.

    Subclasses must implement :meth:`detect`.
    Optional warm-up / teardown hooks: :meth:`warmup`, :meth:`close`.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._total_frames: int = 0
        self._total_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Time-tracked detection entry point."""
        t0 = time.perf_counter()
        results = self.detect(frame)
        elapsed = time.perf_counter() - t0
        self._total_frames += 1
        self._total_time += elapsed
        return results

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : numpy array of shape (H, W, 3), dtype uint8, BGR colour order

        Returns
        -------
        List[Detection] – zero or more detections above confidence_threshold
        """

    def warmup(self, dummy_shape: Tuple[int, int] = (480, 640)) -> None:
        """
        Run a single dummy inference to initialise model weights and JIT caches.
        Call once before the real-time loop.
        """
        dummy = np.zeros((*dummy_shape, 3), dtype=np.uint8)
        self(dummy)  # use __call__ so timing stats are recorded

    def close(self) -> None:
        """Release any held resources (GPU memory, file handles, …)."""

    # ------------------------------------------------------------------ #
    # Performance helpers
    # ------------------------------------------------------------------ #
    @property
    def avg_latency_ms(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return (self._total_time / self._total_frames) * 1000

    @property
    def avg_fps(self) -> float:
        if self._total_time == 0:
            return 0.0
        return self._total_frames / self._total_time

    def reset_stats(self) -> None:
        self._total_frames = 0
        self._total_time = 0.0

    def stats(self) -> dict:
        return {
            "detector": self.__class__.__name__,
            "frames_processed": self._total_frames,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_fps": round(self.avg_fps, 1),
        }
