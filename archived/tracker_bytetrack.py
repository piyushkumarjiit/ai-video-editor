"""
---- Legacy : No longer used ---
FILE: tracker_bytetrack.py
ROLE: Temporal Entity Tracking (Object Persistence)
-------------------------------------------------------------------------
DESCRIPTION:
Batch processes videos using ByteTrack. Optimized for 1080 Ti.
Updated: Suppresses NumPy subnormal warnings and YOLO console noise.
-------------------------------------------------------------------------
"""

import cv2
import json
import os
import torch
import logging
import warnings
from ultralytics import YOLO
from pathlib import Path

# --- SILENCE CONSOLE NOISE ---
# 1. Suppress the NumPy subnormal warnings you're seeing
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
# 2. Suppress Ultralytics internal logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# --- CONFIGURATION ---
INPUT_DIR = "samples/sanitized"
OUTPUT_DIR = "tracking"
#MODEL_VERSION = "models/yolov8x.pt"
#MODEL_PATH = os.path.join("models", "yolov8x.pt")

MODEL_DIR = "models"
MODEL_FILENAME = "yolov8x.pt" # or yolov8n-face.pt
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_DIR, MODEL_FILENAME)

# # 2. Safety Check: Verify the file exists so Ultralytics doesn't auto-download to root
# if not os.path.exists(MODEL_PATH):
#     print(f"❌ CRITICAL ERROR: Model not found at {MODEL_PATH}")
#     print(f"Please ensure {MODEL_FILENAME} is moved into the '{MODEL_DIR}' folder.")
#     exit(1)

def batch_track_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not detected. Running on CPU.")
        device = "cpu"
    else:
        device = 0 
        print(f"🚀 GPU ENGINE: {torch.cuda.get_device_name(0)}")
        print("💡 Tracking active. Check nvtop for encoder/decoder spikes.")

    # verbose=False here prevents the YOLO startup banner
    model = YOLO(MODEL_PATH, verbose=False)
    print(f"✅ Loaded {MODEL_FILENAME} from {MODEL_PATH}")
    
    video_extensions = ('.mp4', '.mkv', '.mov', '.avi')
    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(video_extensions)]
    
    if not videos:
        print(f"📭 No videos found in {INPUT_DIR}.")
        return

    for video_file in videos:
        video_path = os.path.join(INPUT_DIR, video_file)
        video_stem = Path(video_file).stem
        output_json = os.path.join(OUTPUT_DIR, f"{video_stem}_tracking.json")

        if os.path.exists(output_json):
            print(f" Already processed file. Skipping: {video_file}...")
            continue

        print(f"🎬 Processing: {video_file}...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # verbose=False here removes the frame-by-frame text output
            results = model.track(
                source=video_path, 
                tracker="bytetrack.yaml", 
                stream=True, 
                persist=True,
                device=device,
                verbose=False 
            )
            
            tracks = {}
            for frame_idx, r in enumerate(results):
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    
                    for obj_id, box in zip(ids, xyxy):
                        str_id = str(obj_id)
                        if str_id not in tracks:
                            tracks[str_id] = {"first_frame": frame_idx, "trajectory": []}
                        
                        tracks[str_id]["last_frame"] = frame_idx
                        tracks[str_id]["trajectory"].append({
                            "frame": frame_idx,
                            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                        })

            with open(output_json, "w") as f:
                json.dump({"source_video": video_file, "fps": fps, "entities": tracks}, f, indent=2)
            
            print(f"✅ Success: {video_file} ({len(tracks)} tracks generated).")

        except Exception as e:
            print(f"❌ Error processing {video_file}: {e}")

if __name__ == "__main__":
    batch_track_videos()