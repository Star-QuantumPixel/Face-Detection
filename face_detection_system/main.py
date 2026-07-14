#!/usr/bin/env python3
"""
Face Detection System — Main Entry Point
=========================================

Modes
-----
  webcam  : Real-time detection from webcam with live visualisation
  video   : Process a video file, write annotated output video
  batch   : Process a directory of images, write annotated copies + JSON log
  api     : Start REST API server (POST /detect, GET /health, GET /stats)
  bench   : Benchmark both detectors on a synthetic or real image

Quick Start
-----------
  # Real-time webcam (DNN detector, camera 0):
  python main.py

  # Traditional Haar approach:
  python main.py --detector haar

  # Process a video file:
  python main.py --mode video --source myvideo.mp4 --output output/

  # Batch process images:
  python main.py --mode batch --source ./images/ --output output/

  # Start REST API:
  python main.py --mode api --port 8080

  # Benchmark:
  python main.py --mode bench

Architecture
------------
  InputHandler  ──►  Detector  ──►  PostProcessor  ──►  OutputFormatter
      │                                                       │
      └──────────── Frame loop ──────────────────────────────┘
                                      │
                              Benchmarker (timing)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np


# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detectors import HaarCascadeDetector, DNNDetector
from src.detectors.base import BaseDetector, Detection
from src.pipeline import InputHandler, FrameSource, PostProcessor, OutputFormatter
from src.pipeline.output_formatter import draw_detections
from src.utils import get_logger, Benchmarker, load_config

logger = get_logger("main")


# ══════════════════════════════════════════════════════════════════════════ #
# Detector factory
# ══════════════════════════════════════════════════════════════════════════ #
def build_detector(detector_type: str, config: dict, model_dir: str) -> BaseDetector:
    """Instantiate the requested detector from config."""
    conf_thresh = config.get("confidence_threshold", 0.5)

    if detector_type == "haar":
        haar_cfg = config.get("haar", {})
        frontal = haar_cfg.get("frontal_model",
                               os.path.join(model_dir, "haarcascade_frontalface_default.xml"))
        profile = haar_cfg.get("profile_model",
                               os.path.join(model_dir, "haarcascade_profileface.xml"))
        eye = haar_cfg.get("eye_model",
                           os.path.join(model_dir, "haarcascade_eye.xml"))
        return HaarCascadeDetector(
            frontal_model_path=frontal,
            profile_model_path=profile if os.path.isfile(profile) else None,
            eye_model_path=eye if os.path.isfile(eye) else None,
            confidence_threshold=conf_thresh,
            scale_factor=haar_cfg.get("scale_factor", 1.1),
            min_neighbors=haar_cfg.get("min_neighbors", 5),
            min_face_size=tuple(haar_cfg.get("min_face_size", [30, 30])),
            use_eye_validation=haar_cfg.get("use_eye_validation", False),
        )

    elif detector_type == "dnn":
        dnn_cfg = config.get("dnn", {})
        return DNNDetector(
            model_dir=dnn_cfg.get("model_dir", model_dir),
            confidence_threshold=conf_thresh,
            input_size=tuple(dnn_cfg.get("input_size", [300, 300])),
            mean=tuple(dnn_cfg.get("mean", [104.0, 177.0, 123.0])),
            nms_threshold=dnn_cfg.get("nms_threshold", 0.4),
            download=True,
        )

    else:
        raise ValueError(f"Unknown detector type: '{detector_type}'. Choose 'haar' or 'dnn'.")


# ══════════════════════════════════════════════════════════════════════════ #
# Mode: webcam / video  (real-time display)
# ══════════════════════════════════════════════════════════════════════════ #
def run_realtime(
    detector: BaseDetector,
    post_processor: PostProcessor,
    output_formatter: OutputFormatter,
    source_type: str,
    source,
    config: dict,
    output_dir: str,
):
    """
    Real-time detection loop with live display.

    Threading note
    --------------
    For maximum throughput on multicore systems you can split this loop into
    three threads:
      Thread 1: capture frames  → frame_queue
      Thread 2: detect          → detection_queue
      Thread 3: display / write <- detection_queue

    Here we use a single-threaded loop for clarity and portability.
    Frame-skipping is used to maintain target FPS when detection is slow.
    """
    display_cfg = config.get("display", {})
    show_window = display_cfg.get("show_window", True)
    window_title = display_cfg.get("window_title", "Face Detection System")
    show_conf = display_cfg.get("show_confidence", True)
    show_tid = display_cfg.get("show_track_id", True)
    overlay_fps = display_cfg.get("overlay_fps", True)

    input_cfg = config.get("input", {})
    target_fps = input_cfg.get("target_fps", 30)

    bench = Benchmarker(window=30)
    frame_idx = 0
    skip_n = 0       # number of frames to skip (adaptive)
    skip_counter = 0

    fs = FrameSource.WEBCAM if source_type == "webcam" else FrameSource.VIDEO
    src_name = str(source)

    # Output video writer (for video mode)
    writer: Optional[cv2.VideoWriter] = None

    logger.info(f"Starting {source_type} detection. Press Q or Esc to quit.")

    try:
        with InputHandler(fs, source, target_fps=target_fps) as inp:
            if show_window:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

            for frame in inp:
                frame_idx += 1

                # ── Adaptive frame skipping ──────────────────────────────
                if skip_n > 0 and skip_counter < skip_n:
                    skip_counter += 1
                    continue
                skip_counter = 0

                # ── Detection ────────────────────────────────────────────
                with bench.time("detect"):
                    detections = detector(frame)

                # ── Post-processing ──────────────────────────────────────
                with bench.time("post"):
                    detections = post_processor.process(
                        detections, frame.shape[:2]
                    )

                # ── Output logging ───────────────────────────────────────
                output_formatter.write(frame, detections, frame_idx, src_name)

                # ── Adaptive skip: target 25 fps ─────────────────────────
                det_ms = bench.avg_ms("detect")
                if det_ms > 0:
                    ideal_skip = max(0, int(det_ms / (1000 / 25)) - 1)
                    skip_n = min(ideal_skip, 4)  # max skip 4 frames

                # ── Visualisation ────────────────────────────────────────
                fps_val = bench.fps("detect") if overlay_fps else None
                with bench.time("draw"):
                    annotated = draw_detections(
                        frame, detections,
                        show_confidence=show_conf,
                        show_track_id=show_tid,
                        overlay_fps=fps_val,
                    )

                # Init video writer after first frame
                if source_type == "video" and writer is None:
                    h, w = annotated.shape[:2]
                    out_path = os.path.join(output_dir, "output_annotated.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    native_fps = inp.native_fps or 25
                    writer = cv2.VideoWriter(out_path, fourcc, native_fps, (w, h))
                    logger.info(f"Writing annotated video to: {out_path}")
                if writer:
                    writer.write(annotated)

                if show_window:
                    cv2.imshow(window_title, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                        logger.info("User quit.")
                        break

                # ── Console heartbeat every 30 frames ────────────────────
                if frame_idx % 30 == 0:
                    s = bench.summary()
                    logger.info(
                        f"Frame {frame_idx} | "
                        f"detect={s['detect']['avg_ms']:.1f}ms "
                        f"({s['detect']['fps']:.0f} FPS) | "
                        f"faces={len(detections)} | "
                        f"skip={skip_n}"
                    )

    finally:
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    logger.info(f"Finished. Processed {frame_idx} frames.")
    logger.info(json.dumps(bench.summary(), indent=2))


# ══════════════════════════════════════════════════════════════════════════ #
# Mode: batch (image directory)
# ══════════════════════════════════════════════════════════════════════════ #
def run_batch(
    detector: BaseDetector,
    post_processor: PostProcessor,
    output_formatter: OutputFormatter,
    source_dir: str,
    output_dir: str,
):
    """Process all images in source_dir, save annotated copies to output_dir."""
    annotated_dir = os.path.join(output_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    bench = Benchmarker()
    total = 0
    total_faces = 0

    with InputHandler(FrameSource.IMAGE_DIR, source_dir) as inp:
        for frame_idx, frame in enumerate(inp):
            with bench.time("detect"):
                detections = detector(frame)

            # No temporal smoothing for independent images
            output_formatter.write(frame, detections, frame_idx, source_dir)

            annotated = draw_detections(frame, detections, show_track_id=False)
            out_path = os.path.join(annotated_dir, f"result_{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, annotated)

            total += 1
            total_faces += len(detections)

            if (frame_idx + 1) % 10 == 0:
                logger.info(f"Processed {frame_idx + 1} images…")

    logger.info(
        f"\nBatch complete: {total} images, {total_faces} faces detected.\n"
        f"Avg latency: {bench.avg_ms('detect'):.1f} ms/image\n"
        f"Annotated images saved to: {annotated_dir}"
    )


# ══════════════════════════════════════════════════════════════════════════ #
# Mode: api
# ══════════════════════════════════════════════════════════════════════════ #
def run_api(detector: BaseDetector, post_processor: PostProcessor, recognizer, host: str, port: int):
    from src.api import FaceDetectionAPIServer
    server = FaceDetectionAPIServer(detector, host=host, port=port,
                                    post_processor=post_processor, recognizer=recognizer)
    logger.info(f"API server starting on {host}:{port}")
    server.serve_forever()


# ══════════════════════════════════════════════════════════════════════════ #
# Mode: bench (compare both detectors)
# ══════════════════════════════════════════════════════════════════════════ #
def run_bench(model_dir: str, n_frames: int = 100):
    """
    Benchmark Haar vs DNN on random noise frames and print comparison table.

    In a real benchmark, replace synthetic frames with actual face images
    from WIDER Face or FDDB for meaningful accuracy numbers.
    """
    logger.info("=" * 60)
    logger.info("  Detector Benchmark (synthetic noise frames)")
    logger.info("=" * 60)

    # Synthetic 1080p frame
    test_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    results = {}

    # ── Haar ─────────────────────────────────────────────────────────────
    haar_path = os.path.join(model_dir, "haarcascade_frontalface_default.xml")
    if os.path.isfile(haar_path):
        logger.info("\n[1/2] Benchmarking Haar Cascade …")
        haar = HaarCascadeDetector(haar_path, confidence_threshold=0.0)
        haar.warmup()
        haar.reset_stats()

        t0 = time.perf_counter()
        for _ in range(n_frames):
            haar(test_frame)
        elapsed = time.perf_counter() - t0

        results["Haar Cascade"] = {
            "avg_latency_ms": round(haar.avg_latency_ms, 2),
            "fps": round(haar.avg_fps, 1),
            "total_s": round(elapsed, 2),
            "pros": "No DL deps, tiny model, deterministic",
            "cons": "High FP rate, pose-sensitive",
        }
        haar.close()
        logger.info(f"  avg latency: {haar.avg_latency_ms:.1f} ms  ({haar.avg_fps:.0f} FPS)")
    else:
        logger.warning("Haar cascade not found — skipping.")

    # ── DNN ──────────────────────────────────────────────────────────────
    logger.info("\n[2/2] Benchmarking DNN (ResNet-10 SSD) …")
    try:
        dnn = DNNDetector(model_dir=model_dir, confidence_threshold=0.0, download=True)
        dnn.warmup()
        dnn.reset_stats()

        t0 = time.perf_counter()
        for _ in range(n_frames):
            dnn(test_frame)
        elapsed = time.perf_counter() - t0

        results["DNN (ResNet-10 SSD)"] = {
            "avg_latency_ms": round(dnn.avg_latency_ms, 2),
            "fps": round(dnn.avg_fps, 1),
            "total_s": round(elapsed, 2),
            "pros": "High accuracy, handles poses/lighting/occlusion",
            "cons": "~10 MB model, slightly slower on old CPUs",
        }
        dnn.close()
        logger.info(f"  avg latency: {dnn.avg_latency_ms:.1f} ms  ({dnn.avg_fps:.0f} FPS)")
    except Exception as e:
        logger.error(f"DNN benchmark failed: {e}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results over {n_frames} × 720p frames")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  ▶  {name}")
        print(f"     Latency : {r['avg_latency_ms']} ms")
        print(f"     FPS     : {r['fps']}")
        print(f"     Pros    : {r['pros']}")
        print(f"     Cons    : {r['cons']}")
    print()

    print("  WHEN TO USE EACH APPROACH")
    print("  " + "-" * 56)
    print("  Haar Cascade  →  Raspberry Pi / MCU, offline edge devices,")
    print("                   simple demos, no GPU, < 1 MB memory budget")
    print("  DNN SSD       →  Desktop, server, mobile GPU, when accuracy")
    print("                   matters, varied lighting / pose / crowd")
    print()


# ══════════════════════════════════════════════════════════════════════════ #
# CLI
# ══════════════════════════════════════════════════════════════════════════ #
def parse_args():
    p = argparse.ArgumentParser(
        description="Face Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode", choices=["webcam", "video", "batch", "api", "bench"],
        default="webcam",
        help="Execution mode (default: webcam)",
    )
    p.add_argument(
        "--detector", choices=["haar", "dnn"], default="dnn",
        help="Detector backend (default: dnn)",
    )
    p.add_argument(
        "--source", default="0",
        help="Webcam index, video path, image dir, or RTSP URL (default: 0)",
    )
    p.add_argument(
        "--output", default="output",
        help="Output directory for logs, crops, annotated images/video",
    )
    p.add_argument(
        "--model-dir", default="models",
        help="Directory containing model files (default: models/)",
    )
    p.add_argument(
        "--config", default="configs/default.json",
        help="Path to JSON config file",
    )
    p.add_argument(
        "--confidence", type=float, default=None,
        help="Override confidence threshold (0–1)",
    )
    p.add_argument("--host", default="0.0.0.0", help="API server host")
    p.add_argument("--port", type=int, default=8080, help="API server port")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live window (useful in headless environments)")
    p.add_argument("--save-crops", action="store_true",
                   help="Save cropped face images")
    p.add_argument("--save-json", action="store_true",
                   help="Save detections as NDJSON log")
    p.add_argument("--bench-frames", type=int, default=50,
                   help="Number of frames for benchmark mode")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════ #
# Entry point
# ══════════════════════════════════════════════════════════════════════════ #
def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config = {}
    if os.path.isfile(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")
    else:
        logger.warning(f"Config not found: {args.config} — using defaults.")

    det_cfg = config.get("detector", {})
    pp_cfg = config.get("post_processor", {})
    out_cfg = config.get("output", {})
    disp_cfg = config.get("display", {})

    # CLI overrides
    if args.confidence is not None:
        det_cfg["confidence_threshold"] = args.confidence
    if args.no_display:
        disp_cfg["show_window"] = False
    if args.save_crops:
        out_cfg["save_crops"] = True
    if args.save_json:
        out_cfg["save_json"] = True

    # ── Benchmark mode (no detector build needed for comparison) ──────────
    if args.mode == "bench":
        run_bench(model_dir=args.model_dir, n_frames=args.bench_frames)
        return

    # ── Build components ──────────────────────────────────────────────────
    logger.info(f"Building detector: {args.detector}")
    detector = build_detector(args.detector, det_cfg, args.model_dir)
    detector.warmup()
    logger.info(f"Detector ready: {detector.__class__.__name__}")

    post_processor = PostProcessor(
        min_face_pixels=pp_cfg.get("min_face_pixels", 20),
        max_face_fraction=pp_cfg.get("max_face_fraction", 0.95),
        smooth_alpha=pp_cfg.get("smooth_alpha", 0.6),
        min_frames=pp_cfg.get("min_frames", 2),
        max_disappeared=pp_cfg.get("max_disappeared", 10),
    )

    os.makedirs(args.output, exist_ok=True)
    output_formatter = OutputFormatter(
        output_dir=args.output,
        save_crops=out_cfg.get("save_crops", False),
        save_json=out_cfg.get("save_json", False),
        save_csv=out_cfg.get("save_csv", False),
    )

    # ── Normalise source argument ─────────────────────────────────────────
    source = args.source
    try:
        source = int(source)   # webcam index
    except ValueError:
        pass  # keep as string (path / URL)

    # ── Dispatch mode ─────────────────────────────────────────────────────
    try:
        if args.mode in ("webcam", "video"):
            config["display"] = disp_cfg
            run_realtime(
                detector, post_processor, output_formatter,
                source_type=args.mode,
                source=source,
                config=config,
                output_dir=args.output,
            )

        elif args.mode == "batch":
            run_batch(
                detector, post_processor, output_formatter,
                source_dir=str(source),
                output_dir=args.output,
            )

        elif args.mode == "api":
            from src.detectors.recognizer import FaceRecognizer
            try:
                recognizer = FaceRecognizer(model_dir=args.model_dir)
            except Exception as e:
                logger.warning(f"Could not load face recognizer: {e}")
                recognizer = None
                
            run_api(detector, post_processor, recognizer, host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        detector.close()
        logger.info("Detector closed. Goodbye.")


if __name__ == "__main__":
    main()
