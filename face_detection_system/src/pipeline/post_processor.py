"""
PostProcessor — filters, stabilises, and enriches raw detections.

Responsibilities
----------------
1. Size filtering      : remove implausibly small / large faces
2. Aspect ratio filter : faces are roughly square; discard extreme ratios
3. Border exclusion    : ignore detections glued to the image edge
4. Temporal smoothing  : exponential moving average on bbox coordinates
5. Minimum duration    : only emit faces seen for N consecutive frames
6. Face ID assignment  : simple IoU-based tracker for frame-to-frame association
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..detectors.base import Detection


class PostProcessor:
    """
    Stateful post-processor for a stream of Detection lists.

    Parameters
    ----------
    min_face_pixels    : smallest allowable face side length (px)
    max_face_fraction  : largest face width/height as fraction of frame dimension
    min_aspect_ratio   : minimum w/h ratio (e.g. 0.5 rejects very tall boxes)
    max_aspect_ratio   : maximum w/h ratio (e.g. 2.0 rejects very wide boxes)
    border_margin_px   : ignore faces whose bbox touches within this many px of edge
    smooth_alpha       : EMA smoothing coefficient [0=no change, 1=no smoothing]
    min_frames         : consecutive frames a face must appear before being emitted
    max_disappeared    : frames without match before a tracked face is dropped
    iou_track_threshold: minimum IoU to associate a detection with an existing track
    """

    def __init__(
        self,
        min_face_pixels: int = 20,
        max_face_fraction: float = 0.95,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        border_margin_px: int = 2,
        smooth_alpha: float = 0.6,
        min_frames: int = 2,
        max_disappeared: int = 10,
        iou_track_threshold: float = 0.2,
    ):
        self.min_face_pixels = min_face_pixels
        self.max_face_fraction = max_face_fraction
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.border_margin_px = border_margin_px
        self.smooth_alpha = smooth_alpha
        self.min_frames = min_frames
        self.max_disappeared = max_disappeared
        self.iou_track_threshold = iou_track_threshold

        # Tracker state
        self._next_id: int = 0
        self._tracks: Dict[int, _Track] = {}  # id -> Track

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],   # (H, W)
    ) -> List[Detection]:
        """
        Apply full post-processing pipeline.

        Returns stabilised, filtered detections, each annotated with a
        persistent `track_id` stored in `det.label`.
        """
        h, w = frame_shape

        # Step 1 — geometric filters
        dets = [d for d in detections if self._passes_filters(d, h, w)]

        # Step 2 — temporal tracking + smoothing
        dets = self._update_tracks(dets)

        # Step 3 — minimum duration filter
        dets = [d for d in dets if self._track_mature(d)]

        return dets

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # Geometric filters
    # ------------------------------------------------------------------ #
    def _passes_filters(self, d: Detection, h: int, w: int) -> bool:
        # Minimum size
        if d.w < self.min_face_pixels or d.h < self.min_face_pixels:
            return False
        # Maximum size
        if d.w > w * self.max_face_fraction or d.h > h * self.max_face_fraction:
            return False
        # Aspect ratio
        ratio = d.w / max(d.h, 1)
        if not (self.min_aspect_ratio <= ratio <= self.max_aspect_ratio):
            return False
        # Border exclusion
        m = self.border_margin_px
        if d.x <= m or d.y <= m or d.x2 >= w - m or d.y2 >= h - m:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def _update_tracks(self, detections: List[Detection]) -> List[Detection]:
        """IoU-based greedy tracker with EMA smoothing."""
        # Mark all existing tracks as unseen this frame
        unmatched_track_ids = set(self._tracks.keys())
        matched_det_indices = set()

        # --- Greedy match: best IoU first ---
        if detections and self._tracks:
            # Build IoU matrix
            track_ids = list(self._tracks.keys())
            iou_matrix = np.zeros((len(detections), len(track_ids)))
            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    iou_matrix[i, j] = det.iou(
                        Detection(*self._tracks[tid].smoothed_xywh, 0.0)
                    )

            while True:
                if iou_matrix.size == 0:
                    break
                r, c = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[r, c]
                if best_iou < self.iou_track_threshold:
                    break
                tid = track_ids[c]
                self._tracks[tid].update(detections[r], self.smooth_alpha)
                detections[r].label = f"face#{tid}"
                unmatched_track_ids.discard(tid)
                matched_det_indices.add(r)
                # Zero out this row and column so we don't re-match them
                iou_matrix[r, :] = 0
                iou_matrix[:, c] = 0

        # --- New tracks for unmatched detections ---
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(tid, det)
                det.label = f"face#{tid}"

        # --- Increment disappeared counter for unmatched tracks ---
        dead_ids = []
        for tid in unmatched_track_ids:
            self._tracks[tid].disappeared += 1
            if self._tracks[tid].disappeared > self.max_disappeared:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self._tracks[tid]

        # Smooth coordinates for matched detections
        for det in detections:
            tid_str = det.label  # e.g. "face#3"
            if "#" in tid_str:
                tid = int(tid_str.split("#")[1])
                if tid in self._tracks:
                    sx, sy, sw, sh = self._tracks[tid].smoothed_xywh
                    det.x, det.y, det.w, det.h = int(sx), int(sy), int(sw), int(sh)

        return detections

    def _track_mature(self, det: Detection) -> bool:
        if "#" not in det.label:
            return True  # untracked — pass through
        tid = int(det.label.split("#")[1])
        track = self._tracks.get(tid)
        if track is None:
            return False
        return track.frame_count >= self.min_frames


class _Track:
    """Internal track state for a single face identity."""

    def __init__(self, tid: int, det: Detection):
        self.tid = tid
        self.smoothed_xywh = [float(det.x), float(det.y), float(det.w), float(det.h)]
        self.frame_count = 1
        self.disappeared = 0

    def update(self, det: Detection, alpha: float) -> None:
        """EMA update: new = alpha * new + (1 - alpha) * old"""
        sx, sy, sw, sh = self.smoothed_xywh
        self.smoothed_xywh = [
            alpha * det.x + (1 - alpha) * sx,
            alpha * det.y + (1 - alpha) * sy,
            alpha * det.w + (1 - alpha) * sw,
            alpha * det.h + (1 - alpha) * sh,
        ]
        self.frame_count += 1
        self.disappeared = 0
