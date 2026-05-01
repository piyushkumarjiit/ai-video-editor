"""
FILE: yolo_tracker.py
ROLE: Inference Worker (Detection + Re-Identification)
-------------------------------------------------------------------------
DESCRIPTION:
Performs object detection using YOLOv8x (TensorRT Engine) and 
temporal tracking via StrongSORT with OSNet ReID. Optimized for 
high-resolution (1088px) inference on Pascal-architecture GPUs.

FEATURES:
- Dynamic OS-level stderr suppression for TensorRT/C++ noise.
- Visual heartbeats (raw) and detection previews (annotated).
- Automated JSON trajectory exporting for downstream RAG/AI pipelines.

UPDATES:
- Replaced CPU VideoCapture with cv2.cudacodec for 1080 Ti hardware decoding.
- Optimized for high-resolution (1088px) inference on Pascal-architecture GPUs.
-------------------------------------------------------------------------
"""

import os
import sys
import warnings
import logging

# ─── SILENCE BEFORE ANY IMPORTS ───────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRT_LOGGER_SEVERITY'] = '3'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger('boxmot').setLevel(logging.CRITICAL)
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
logging.getLogger('torch').setLevel(logging.CRITICAL)

import argparse
import contextlib
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.tracker_zoo import create_tracker

MODEL_CONFIG = {"imgsz": 1088, "half": False}
SCRIPT_ROOT  = Path(__file__).parent.resolve()
REID_PATH    = SCRIPT_ROOT / "models" / "TensorModels" / "osnet_x0_25_msmt17.pt"
YOLO_PATH    = SCRIPT_ROOT / "models" / "TensorModels" / "yolov8x.engine"


@contextlib.contextmanager
def suppress_stderr_fd():
    """Redirect OS-level fd 2 to /dev/null. Catches C++/TRT stderr noise."""
    null_fd  = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)


def process_video(video_path, args, detector, device):
    # ─── STREAM INITIALIZATION WITH FALLBACK ───────────────────────────
    use_gpu_decode = True
    try:
        # Primary Attempt: 1080 Ti Hardware Decoder
        reader = cv2.cudacodec.createVideoReader(str(video_path))
        
        # Metadata check (Hardware readers often need a warm-up or side-car for count)
        _temp_cap = cv2.VideoCapture(str(video_path))
        total_frames = int(_temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _temp_cap.release()
        print(json.dumps({"type": "info", "msg": "Using 1080 Ti NVDEC for decoding."}))
    
    except Exception as e:
        # Secondary Attempt: CPU Fallback
        use_gpu_decode = False
        print(json.dumps({"type": "info", "msg": f"GPU Decoder failed ({e}). Falling back to CPU."}))
        reader = cv2.VideoCapture(str(video_path))
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    heartbeat = preview_dir / "latest_heartbeat.jpg"
    detection_path = preview_dir / "latest_detection.jpg"

    with suppress_stderr_fd():
        tracker = create_tracker(
            tracker_type="strongsort",
            reid_weights=REID_PATH,
            device=device,
            half=MODEL_CONFIG["half"],
        )

    if hasattr(tracker, "tracker"):
        tracker.tracker.max_age = 400
        tracker.tracker.n_init  = 2

    entities  = {}
    frame_idx = 0

    print(json.dumps({"type": "init", "total": total_frames}), flush=True)

    while True:
        # ─── UNIFIED FRAME GRABBING ────────────────────────────────────
        if use_gpu_decode:
            ret, gpu_frame = reader.nextFrame()
            if not ret: break
            frame = gpu_frame.download() # Convert GpuMat to NumPy
        else:
            ret, frame = reader.read()
            if not ret: break
            
        frame_idx += 1

        # ─── INFERENCE & TRACKING ──────────────────────────────────────
        results = detector.predict(
            frame, conf=args.conf, classes=[0, 2, 3, 5, 7],
            device=device, verbose=False, imgsz=MODEL_CONFIG["imgsz"],
        )

        annotated_frame = results[0].plot()
        dets = (results[0].boxes.data.cpu().numpy()
                if len(results[0].boxes) > 0 else np.empty((0, 6)))

        try:
            tracks = tracker.update(dets, frame)

            if args.show_thumbnails and frame_idx % args.thumbnail_interval == 0:
                cv2.imwrite(str(heartbeat), frame)
                cv2.imwrite(str(detection_path), annotated_frame)

            if tracks is not None:
                for t in tracks:
                    tid = str(int(t[4]))
                    if tid not in entities:
                        entities[tid] = {"trajectory": []}
                    entities[tid]["trajectory"].append({
                        "f": frame_idx,
                        "b": [int(t[0]), int(t[1]), int(t[2]), int(t[3])],
                    })
        except Exception:
            pass

        if frame_idx % 5 == 0:
            print(json.dumps({"type": "progress", "frame": frame_idx}), flush=True)

    # Cleanup based on reader type
    if not use_gpu_decode:
        reader.release()

    with open(out_dir / f"{video_path.stem}_tracking.json", "w") as f:
        json.dump(entities, f)

    print(json.dumps({"type": "done"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",             type=str,   required=True)
    parser.add_argument("--output_dir",         type=str,   required=True)
    parser.add_argument("--conf",               type=float, default=0.45)
    parser.add_argument("--worker_id",          type=int,   default=0)
    parser.add_argument("--show_thumbnails",    action="store_true")
    parser.add_argument("--thumbnail_interval", type=int,   default=30)
    args = parser.parse_args()

    with suppress_stderr_fd():
        detector = YOLO(str(YOLO_PATH), task="detect")

    process_video(Path(args.source), args, detector, "cuda:0")

if __name__ == "__main__":
    main()