"""
Benchmarker — rolling-window FPS and latency tracker.

Tracks multiple named timers independently so you can measure
capture, detection, post-processing, and rendering separately.
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Optional


class _Timer:
    def __init__(self, window: int = 60):
        self._window = window
        self._times: Deque[float] = deque(maxlen=window)
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        self._t0 = None
        return elapsed

    @property
    def avg_ms(self) -> float:
        if not self._times:
            return 0.0
        return 1000 * sum(self._times) / len(self._times)

    @property
    def fps(self) -> float:
        avg = self.avg_ms
        return 1000 / avg if avg > 0 else 0.0


class Benchmarker:
    """
    Usage
    -----
    bench = Benchmarker()

    with bench.time("detect"):
        dets = detector(frame)

    print(bench.summary())
    """

    def __init__(self, window: int = 60):
        self._window = window
        self._timers: Dict[str, _Timer] = {}

    def _get(self, name: str) -> _Timer:
        if name not in self._timers:
            self._timers[name] = _Timer(self._window)
        return self._timers[name]

    @contextmanager
    def time(self, name: str):
        t = self._get(name)
        t.start()
        try:
            yield
        finally:
            t.stop()

    def avg_ms(self, name: str) -> float:
        return self._get(name).avg_ms

    def fps(self, name: str) -> float:
        return self._get(name).fps

    def summary(self) -> Dict[str, dict]:
        return {
            name: {"avg_ms": round(t.avg_ms, 2), "fps": round(t.fps, 1)}
            for name, t in self._timers.items()
        }
