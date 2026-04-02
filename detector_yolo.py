import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURATION ---
# 'yolov8n.pt' is extremely fast; 'yolov8m.pt' is more accurate for faces
MODEL_VARIANT = 'yolov8n-face.pt' # Or 'yolov8n.pt' for general objects
KEYFRAMES_DIR = 'keyframes'
OUTPUT_FILE = 'yolo_detections.json'
CONFIDENCE_THRESHOLD = 0.4

# 1. Load the model once (Keep it in VRAM)
# Note: You may need to download a face-specific weights file or use standard yolov8
model = YOLO('yolov8n.pt') 

def run_yolo_scan():
    analysis_results = {}

    # Get folders (videos) inside keyframes directory
    video_folders = [f for f in os.listdir(KEYFRAMES_DIR) if os.path.isdir(os.path.join(KEYFRAMES_DIR, f))]

    for video_name in video_folders:
        print(f"\n🚀 YOLO High-Speed Scan: {video_name}")
        analysis_results[video_name] = []
        
        folder_path = os.path.join(KEYFRAMES_DIR, video_name)
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])

        for img_name in tqdm(images, desc="Processing Frames"):
            img_path = os.path.join(folder_path, img_name)
            
            # Perform Detection
            # stream=True handles memory better for large folders
            results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)
            
            frame_detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get coordinates in [x1, y1, x2, y2]
                    # We convert to your [ymin, xmin, ymax, xmax] 0-1000 format
                    b = box.xyxyn[0].tolist() # Normalized 0.0 to 1.0
                    
                    # YOLO gives [xmin, ymin, xmax, ymax] -> We swap for your JSON format
                    ymin, xmin, ymax, xmax = b[1] * 1000, b[0] * 1000, b[3] * 1000, b[2] * 1000
                    
                    label_id = int(box.cls[0])
                    label_name = model.names[label_id]

                    # Filter for things we care about (Person/Face)
                    if label_name in ['person', 'face']:
                        frame_detections.append({
                            "id": f"{label_name}_{len(frame_detections)+1}",
                            "label": label_name,
                            "bbox_2d": [int(ymin), int(xmin), int(ymax), int(xmax)]
                        })

            analysis_results[video_name].append({
                "frame": img_name,
                "detections": frame_detections
            })

    # Save to the same JSON format we used for Qwen
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(analysis_results, f, indent=4)

    print(f"\n✨ YOLO Scan Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_yolo_scan()