"""
InputHandler — unified frame source abstraction.

Supports:
  • Webcam (OpenCV index or /dev/videoN path)
  • Video file (.mp4, .avi, .mkv, …)
  • Single image file (.jpg, .png, .bmp, …)
  • Directory of images (sorted glob)
  • IP / RTSP camera stream (URL string)

Usage
-----
    with InputHandler("webcam", source=0, target_fps=30) as src:
        for frame in src:
            process(frame)
"""
from __future__ import annotations

import glob
import os
import time
from enum import Enum, auto
from typing import Generator, Iterator, List, Optional, Union

import cv2
import numpy as np


class FrameSource(Enum):
    WEBCAM = auto()
    VIDEO = auto()
    IMAGE = auto()
    IMAGE_DIR = auto()
    STREAM = auto()


class InputHandler:
    """
    Unified iterator over frames from heterogeneous input sources.

    Parameters
    ----------
    source_type : FrameSource enum or string key ("webcam", "video", …)
    source      : device index (int) or path / URL (str)
    target_fps  : throttle webcam capture to this FPS (None = no cap)
    resize      : (width, height) to resize every frame, or None
    loop_video  : restart video from beginning when it ends
    max_frames  : stop after this many frames (None = infinite for cameras)
    """

    _TYPE_MAP = {
        "webcam": FrameSource.WEBCAM,
        "video": FrameSource.VIDEO,
        "image": FrameSource.IMAGE,
        "image_dir": FrameSource.IMAGE_DIR,
        "stream": FrameSource.STREAM,
    }

    def __init__(
        self,
        source_type: Union[FrameSource, str] = FrameSource.WEBCAM,
        source: Union[int, str] = 0,
        target_fps: Optional[float] = None,
        resize: Optional[tuple] = None,
        loop_video: bool = False,
        max_frames: Optional[int] = None,
    ):
        if isinstance(source_type, str):
            source_type = self._TYPE_MAP[source_type.lower()]
        self.source_type = source_type
        self.source = source
        self.target_fps = target_fps
        self.resize = resize
        self.loop_video = loop_video
        self.max_frames = max_frames

        self._cap: Optional[cv2.VideoCapture] = None
        self._image_paths: List[str] = []
        self._frame_count = 0
        self._last_capture_time = 0.0

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "InputHandler":
        self._open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self._frame_generator()

    # ------------------------------------------------------------------ #
    # Open / release
    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        if self.source_type == FrameSource.IMAGE_DIR:
            self._image_paths = self._collect_images(str(self.source))
            if not self._image_paths:
                raise FileNotFoundError(
                    f"No images found in directory: {self.source}"
                )
        elif self.source_type == FrameSource.IMAGE:
            self._image_paths = [str(self.source)]
        else:
            device = self.source if isinstance(self.source, int) else str(self.source)
            self._cap = cv2.VideoCapture(device)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open source: {self.source}")
            # Hint: request a higher buffer for RTSP streams
            if self.source_type == FrameSource.STREAM:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # Frame generator
    # ------------------------------------------------------------------ #
    def _frame_generator(self) -> Generator[np.ndarray, None, None]:
        if self.source_type in (FrameSource.IMAGE, FrameSource.IMAGE_DIR):
            yield from self._image_generator()
        else:
            yield from self._capture_generator()

    def _image_generator(self) -> Generator[np.ndarray, None, None]:
        for path in self._image_paths:
            if self.max_frames and self._frame_count >= self.max_frames:
                return
            frame = cv2.imread(path)
            if frame is None:
                print(f"[InputHandler] WARNING: Cannot read image: {path}")
                continue
            frame = self._apply_resize(frame)
            self._frame_count += 1
            yield frame

    def _capture_generator(self) -> Generator[np.ndarray, None, None]:
        assert self._cap is not None
        delay = 1.0 / self.target_fps if self.target_fps else 0.0

        while True:
            if self.max_frames and self._frame_count >= self.max_frames:
                break

            # FPS throttle
            if delay > 0:
                elapsed = time.monotonic() - self._last_capture_time
                if elapsed < delay:
                    time.sleep(delay - elapsed)

            ok, frame = self._cap.read()
            self._last_capture_time = time.monotonic()

            if not ok:
                if self.loop_video and self.source_type == FrameSource.VIDEO:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Camera disconnect / end of video
                break

            frame = self._apply_resize(frame)
            self._frame_count += 1
            yield frame

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _apply_resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
        return frame

    @staticmethod
    def _collect_images(directory: str) -> List[str]:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
        paths: List[str] = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, ext)))
            paths.extend(glob.glob(os.path.join(directory, ext.upper())))
        return sorted(set(paths))

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def native_fps(self) -> Optional[float]:
        """FPS as reported by the VideoCapture (0 if unavailable)."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or None
        return None

    @property
    def resolution(self) -> Optional[tuple]:
        """(width, height) as reported by the VideoCapture."""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return None
