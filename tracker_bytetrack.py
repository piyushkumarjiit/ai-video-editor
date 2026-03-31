# tracker_bytetrack.py
import cv2
import json
import os
from ultralytics import YOLO

def track_entities(video_path, output_json, model_version="yolov8n.pt"):
    print(f"[INFO] Initializing YOLO Tracker on {video_path}...")
    model = YOLO(model_version) # Automatically uses CUDA if available
    
    # Run built-in ByteTrack
    results = model.track(source=video_path, tracker="bytetrack.yaml", stream=True, persist=True)
    
    tracks = {}
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            
            for obj_id, box in zip(ids, xyxy):
                if obj_id not in tracks:
                    tracks[obj_id] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "trajectory": []
                    }
                
                tracks[obj_id]["last_frame"] = frame_idx
                tracks[obj_id]["trajectory"].append({
                    "frame": frame_idx,
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                })

    print(f"[SUCCESS] Tracked {len(tracks)} unique entities.")
    with open(output_json, "w") as f:
        json.dump({"fps": fps, "entities": tracks}, f, indent=2)
    
    return tracks

if __name__ == "__main__":
    # Quick test execution
    track_entities("input.mp4", "tracking_data.json")