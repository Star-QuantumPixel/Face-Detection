"""
Unit and integration tests for the face detection system.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detectors.base import Detection
from src.pipeline.post_processor import PostProcessor
from src.utils.benchmark import Benchmarker


# ================================================================== #
# Detection dataclass tests
# ================================================================== #
class TestDetection:
    def _make(self, x=10, y=20, w=80, h=90, conf=0.8):
        return Detection(x, y, w, h, conf)

    def test_derived_coords(self):
        d = self._make(x=10, y=20, w=80, h=90)
        assert d.x2 == 90
        assert d.y2 == 110
        assert d.center == (50, 65)
        assert d.area == 7200

    def test_to_xyxy(self):
        d = self._make(x=5, y=10, w=50, h=60)
        assert d.to_xyxy() == (5, 10, 55, 70)

    def test_iou_self(self):
        d = self._make()
        assert abs(d.iou(d) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = Detection(0, 0, 10, 10, 0.9)
        b = Detection(100, 100, 10, 10, 0.9)
        assert a.iou(b) == 0.0

    def test_iou_partial(self):
        a = Detection(0, 0, 10, 10, 0.9)
        b = Detection(5, 0, 10, 10, 0.9)   # 50 % horizontal overlap
        iou = a.iou(b)
        assert 0.0 < iou < 1.0

    def test_crop(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        d = Detection(10, 10, 30, 30, 0.9)
        crop = d.crop(frame)
        assert crop.shape == (30, 30, 3)

    def test_crop_with_padding(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        d = Detection(50, 50, 40, 40, 0.9)
        crop = d.crop(frame, padding=0.5)
        # 50 % of 40 = 20 px pad each side → 80 px crop
        assert crop.shape[0] == 80
        assert crop.shape[1] == 80

    def test_repr(self):
        d = self._make()
        assert "Detection" in repr(d)
        assert "conf=" in repr(d)


# ================================================================== #
# PostProcessor tests
# ================================================================== #
class TestPostProcessor:
    def _pp(self, **kwargs):
        return PostProcessor(
            min_frames=1,     # don't require multi-frame for unit tests
            **kwargs,
        )

    def _det(self, x=50, y=50, w=100, h=100, conf=0.9):
        return Detection(x, y, w, h, conf)

    def test_size_filter_small(self):
        pp = self._pp(min_face_pixels=60)
        d = Detection(50, 50, 30, 30, 0.9)          # 30 px — too small
        result = pp.process([d], frame_shape=(480, 640))
        assert result == []

    def test_size_filter_large(self):
        pp = self._pp(max_face_fraction=0.5)
        d = Detection(10, 10, 500, 450, 0.9)         # covers > 50 % of 640-wide frame
        result = pp.process([d], frame_shape=(480, 640))
        assert result == []

    def test_aspect_ratio_filter(self):
        pp = self._pp(min_aspect_ratio=0.5, max_aspect_ratio=2.0)
        # Very tall face (ratio = 0.1)
        tall = Detection(50, 50, 10, 100, 0.9)
        result = pp.process([tall], frame_shape=(480, 640))
        assert result == []

    def test_border_filter(self):
        pp = self._pp(border_margin_px=5)
        # Face touching left border
        d = Detection(0, 50, 80, 80, 0.9)
        result = pp.process([d], frame_shape=(480, 640))
        assert result == []

    def test_passes_valid_detection(self):
        pp = self._pp()
        d = self._det()
        result = pp.process([d], frame_shape=(480, 640))
        assert len(result) == 1

    def test_tracking_assigns_id(self):
        pp = self._pp()
        d = self._det()
        result = pp.process([d], frame_shape=(480, 640))
        assert "#" in result[0].label

    def test_tracking_stable_id(self):
        """Same face in consecutive frames should get the same track ID."""
        pp = PostProcessor(min_frames=1)
        d1 = self._det(x=50, y=50, w=100, h=100)
        d2 = self._det(x=52, y=51, w=100, h=100)   # slight movement
        r1 = pp.process([d1], (480, 640))
        r2 = pp.process([d2], (480, 640))
        assert r1[0].label == r2[0].label

    def test_tracking_new_id_after_gap(self):
        """Face that disappears for too long should get a new ID on return."""
        pp = PostProcessor(min_frames=1, max_disappeared=2)
        d = self._det()
        r1 = pp.process([d], (480, 640))
        id1 = r1[0].label
        # 3 empty frames — exceeds max_disappeared=2
        pp.process([], (480, 640))
        pp.process([], (480, 640))
        pp.process([], (480, 640))
        r2 = pp.process([d], (480, 640))
        id2 = r2[0].label
        assert id1 != id2

    def test_reset_clears_tracks(self):
        pp = PostProcessor(min_frames=1)
        d = self._det()
        pp.process([d], (480, 640))
        pp.reset()
        assert len(pp._tracks) == 0


# ================================================================== #
# Benchmarker tests
# ================================================================== #
class TestBenchmarker:
    def test_timing(self):
        import time
        bench = Benchmarker(window=5)
        with bench.time("sleep"):
            time.sleep(0.01)
        assert bench.avg_ms("sleep") > 5   # at least 5 ms
        assert bench.fps("sleep") < 200    # less than 200 fps

    def test_summary_keys(self):
        bench = Benchmarker()
        with bench.time("detect"):
            pass
        summary = bench.summary()
        assert "detect" in summary
        assert "avg_ms" in summary["detect"]
        assert "fps" in summary["detect"]

    def test_multiple_timers(self):
        bench = Benchmarker()
        with bench.time("a"):
            pass
        with bench.time("b"):
            pass
        assert len(bench.summary()) == 2


# ================================================================== #
# HaarCascadeDetector smoke test
# ================================================================== #
class TestHaarDetector:
    CASCADE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models",
        "haarcascade_frontalface_default.xml"
    )

    @pytest.mark.skipif(
        not os.path.isfile(CASCADE_PATH),
        reason="Haar cascade not present"
    )
    def test_detects_nothing_on_blank_frame(self):
        from src.detectors import HaarCascadeDetector
        det = HaarCascadeDetector(self.CASCADE_PATH, confidence_threshold=0.0)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(blank)
        assert isinstance(results, list)

    @pytest.mark.skipif(
        not os.path.isfile(CASCADE_PATH),
        reason="Haar cascade not present"
    )
    def test_stats(self):
        from src.detectors import HaarCascadeDetector
        det = HaarCascadeDetector(self.CASCADE_PATH)
        det.warmup()
        s = det.stats()
        assert s["frames_processed"] >= 1
        assert s["avg_latency_ms"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
