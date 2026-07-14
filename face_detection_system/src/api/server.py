"""
Lightweight REST API server for face detection.

Endpoints
---------
POST /detect
    Body : multipart/form-data with field "image" (JPEG/PNG bytes)
           OR  application/json with {"image_b64": "<base64 string>"}
    Returns JSON:
    {
        "face_count": 2,
        "detections": [
            {"x": 100, "y": 50, "w": 80, "h": 90, "confidence": 0.97, "label": "face#0"},
            ...
        ],
        "latency_ms": 12.3
    }

GET /health
    Returns {"status": "ok", "detector": "DNNDetector"}

GET /stats
    Returns detector performance statistics.

Usage
-----
    from src.detectors import DNNDetector
    from src.api import FaceDetectionAPIServer

    detector = DNNDetector(model_dir="models")
    server = FaceDetectionAPIServer(detector, host="0.0.0.0", port=8080)
    server.serve_forever()

Note: This is a single-threaded server suitable for low-to-medium load.
For production use, consider FastAPI + uvicorn.
"""
from __future__ import annotations

import base64
import cgi
import io
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np

from ..detectors.base import BaseDetector
from ..pipeline.post_processor import PostProcessor
from ..detectors.recognizer import FaceRecognizer


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread to prevent blocking on concurrent connections."""
    daemon_threads = True


class FaceDetectionAPIServer:
    def __init__(
        self,
        detector: BaseDetector,
        host: str = "0.0.0.0",
        port: int = 8080,
        post_processor: Optional[PostProcessor] = None,
        recognizer: Optional[FaceRecognizer] = None,
    ):
        self.detector = detector
        self.post_processor = post_processor
        self.recognizer = recognizer
        self._host = host
        self._port = port

        # Inject references into handler via class attribute (simple DI)
        _Handler.detector = detector
        _Handler.post_processor = post_processor
        _Handler.recognizer = recognizer

    def serve_forever(self) -> None:
        server = ThreadedHTTPServer((self._host, self._port), _Handler)
        print(f"[API] Serving on http://{self._host}:{self._port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("[API] Shutting down.")
        finally:
            server.server_close()


class _Handler(BaseHTTPRequestHandler):
    detector: BaseDetector = None      # type: ignore[assignment]
    post_processor: Optional[PostProcessor] = None
    recognizer: Optional[FaceRecognizer] = None

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "detector": type(self.detector).__name__})
        elif self.path == "/stats":
            self._json(self.detector.stats())
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path == "/detect":
            self._handle_detect()
        elif self.path == "/enroll":
            self._handle_enroll()
        else:
            self._json({"error": "not found"}, status=404)

    # ------------------------------------------------------------------ #
    # Detection handler
    # ------------------------------------------------------------------ #
    def _handle_detect(self):
        content_type = self.headers.get("Content-Type", "")
        try:
            if "multipart/form-data" in content_type:
                frame = self._parse_multipart()
            elif "application/json" in content_type:
                frame = self._parse_json_b64()
            else:
                self._json({"error": "Unsupported Content-Type"}, status=415)
                return
        except Exception as e:
            self._json({"error": f"Image parse error: {e}"}, status=400)
            return

        if frame is None:
            self._json({"error": "Could not decode image"}, status=400)
            return

        t0 = time.perf_counter()
        detections = self.detector(frame)
        if self.post_processor:
            detections = self.post_processor.process(detections, frame.shape[:2])
            
        if self.recognizer:
            for det in detections:
                x, y, w, h = det.x, det.y, det.w, det.h
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    name, conf = self.recognizer.identify(crop)
                    if name != "Unknown":
                        det.label = name

        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "face_count": len(detections),
            "latency_ms": round(latency_ms, 2),
            "detections": [
                {"x": d.x, "y": d.y, "w": d.w, "h": d.h,
                 "confidence": round(d.confidence, 4), "label": d.label}
                for d in detections
            ],
        }
        self._json(result)

    def _handle_enroll(self):
        content_type = self.headers.get("Content-Type", "")
        try:
            if "application/json" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                if "image_b64" not in data or "name" not in data:
                    self._json({"error": "Missing image_b64 or name"}, status=400)
                    return
                name = data["name"]
                img_bytes = base64.b64decode(data["image_b64"])
                frame = _decode_image(img_bytes)
            else:
                self._json({"error": "Unsupported Content-Type. Use application/json"}, status=415)
                return
        except Exception as e:
            self._json({"error": f"Parse error: {e}"}, status=400)
            return

        if frame is None or not self.recognizer:
            self._json({"error": "Bad frame or recognizer not configured"}, status=400)
            return

        # Detect face first
        detections = self.detector(frame)
        if not detections:
            self._json({"error": "No face found in image"}, status=400)
            return
            
        # Get largest face
        best_det = max(detections, key=lambda d: d.w * d.h)
        x, y, w, h = best_det.x, best_det.y, best_det.w, best_det.h
        fh, fw = frame.shape[:2]
        crop = frame[max(0,y):min(fh,y+h), max(0,x):min(fw,x+w)]
        
        success = self.recognizer.enroll(name, crop)
        if success:
            self._json({"message": f"Successfully enrolled {name}"})
        else:
            self._json({"error": "Failed to extract feature"}, status=500)

    # ------------------------------------------------------------------ #
    # Parsers
    # ------------------------------------------------------------------ #
    def _parse_multipart(self) -> Optional[np.ndarray]:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        # Use cgi.FieldStorage to parse multipart
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers["Content-Type"],
            "CONTENT_LENGTH": str(length),
        }
        fs = cgi.FieldStorage(
            fp=io.BytesIO(body), headers=self.headers, environ=environ
        )
        if "image" not in fs:
            raise ValueError("No 'image' field in multipart form")
        img_bytes = fs["image"].file.read()
        return _decode_image(img_bytes)

    def _parse_json_b64(self) -> Optional[np.ndarray]:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        if "image_b64" not in data:
            raise ValueError("No 'image_b64' key in JSON body")
        img_bytes = base64.b64decode(data["image_b64"])
        return _decode_image(img_bytes)

    # ------------------------------------------------------------------ #
    # Response helper
    # ------------------------------------------------------------------ #
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Silence default HTTP logging; use structured logger instead


def _decode_image(img_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
