"""
OutputFormatter — renders detections as visual overlays and structured records.

Provides
--------
* draw_detections : overlay bounding boxes + confidence labels on a BGR frame
* DetectionRecord : dataclass for JSON / CSV serialisation
* save_crops      : save cropped face images to disk
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from ..detectors.base import Detection


# ------------------------------------------------------------------ #
# Colour palette — one colour per track ID
# ------------------------------------------------------------------ #
_PALETTE = [
    (0, 200, 255),   # amber
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (180, 0, 255),   # purple
    (0, 180, 255),   # orange
    (255, 200, 0),   # cyan
    (100, 255, 255), # yellow
    (255, 0, 180),   # pink
]


def _track_colour(label: str) -> tuple:
    if "#" in label:
        tid = int(label.split("#")[1])
        return _PALETTE[tid % len(_PALETTE)]
    return (0, 255, 0)  # default green


# ------------------------------------------------------------------ #
# Visual overlay
# ------------------------------------------------------------------ #
def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    draw_landmarks: bool = True,
    show_confidence: bool = True,
    show_track_id: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.55,
    overlay_fps: Optional[float] = None,
    overlay_count: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, confidence scores, and optional landmarks on *frame*.

    Returns a copy of the frame with annotations — never mutates the input.
    """
    canvas = frame.copy()

    for det in detections:
        colour = _track_colour(det.label)
        x1, y1, x2, y2 = det.x, det.y, det.x2, det.y2

        # --- Bounding box ---
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, line_thickness)

        # --- Label ---
        parts = []
        if show_track_id and "#" in det.label:
            parts.append(det.label)
        if show_confidence:
            parts.append(f"{det.confidence:.2f}")
        if parts:
            label_text = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            # Background chip
            chip_y1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                canvas, (x1, chip_y1), (x1 + tw + 4, y1), colour, cv2.FILLED
            )
            # Text
            cv2.putText(
                canvas,
                label_text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # black on colour chip
                1,
                cv2.LINE_AA,
            )

        # --- Landmarks (small filled circles) ---
        if draw_landmarks:
            for lx, ly in det.landmarks:
                cv2.circle(canvas, (lx, ly), 3, colour, -1)

    # --- HUD: face count + FPS ---
    hud_lines = []
    if overlay_count:
        hud_lines.append(f"Faces: {len(detections)}")
    if overlay_fps is not None:
        hud_lines.append(f"FPS: {overlay_fps:.1f}")

    for i, line in enumerate(hud_lines):
        cv2.putText(
            canvas,
            line,
            (8, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (8, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ------------------------------------------------------------------ #
# Structured record for persistence
# ------------------------------------------------------------------ #
@dataclass
class DetectionRecord:
    """Serialisable record of detections for a single frame."""
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0
    source: str = ""
    face_count: int = 0
    detections: List[dict] = field(default_factory=list)

    @classmethod
    def from_detections(
        cls,
        detections: List[Detection],
        frame_index: int = 0,
        source: str = "",
    ) -> "DetectionRecord":
        return cls(
            timestamp=time.time(),
            frame_index=frame_index,
            source=source,
            face_count=len(detections),
            detections=[
                {
                    "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                    "confidence": round(d.confidence, 4),
                    "label": d.label,
                }
                for d in detections
            ],
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class OutputFormatter:
    """
    Handles persistent output: JSON log, CSV log, and cropped face images.

    Parameters
    ----------
    output_dir   : root directory for all output artefacts
    save_crops   : save cropped face images
    save_json    : append NDJSON records to detections.ndjson
    save_csv     : append rows to detections.csv
    crop_padding : fraction of bbox to pad when cropping (0.2 = 20 %)
    """

    def __init__(
        self,
        output_dir: str = "output",
        save_crops: bool = False,
        save_json: bool = True,
        save_csv: bool = False,
        crop_padding: float = 0.15,
    ):
        self.output_dir = output_dir
        self.save_crops_flag = save_crops
        self.save_json_flag = save_json
        self.save_csv_flag = save_csv
        self.crop_padding = crop_padding

        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)

        self._json_path = os.path.join(output_dir, "detections.ndjson")
        self._csv_path = os.path.join(output_dir, "detections.csv")
        self._csv_header_written = False

    def write(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int = 0,
        source: str = "",
    ) -> DetectionRecord:
        record = DetectionRecord.from_detections(detections, frame_index, source)

        if self.save_json_flag:
            with open(self._json_path, "a") as f:
                f.write(record.to_json().replace("\n", "") + "\n")

        if self.save_csv_flag:
            self._append_csv(record)

        if self.save_crops_flag:
            self._save_crops(frame, detections, frame_index)

        return record

    def _append_csv(self, record: DetectionRecord) -> None:
        row = {
            "timestamp": record.timestamp,
            "frame_index": record.frame_index,
            "source": record.source,
            "face_count": record.face_count,
        }
        write_header = not self._csv_header_written and not os.path.isfile(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def _save_crops(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int,
    ) -> None:
        for i, det in enumerate(detections):
            crop = det.crop(frame, padding=self.crop_padding)
            if crop.size == 0:
                continue
            fname = f"frame{frame_index:06d}_face{i:02d}.jpg"
            path = os.path.join(self.output_dir, "crops", fname)
            cv2.imwrite(path, crop)
