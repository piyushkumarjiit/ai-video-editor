"""
FILE: post_yolo_draw_labels.py
ROLE: VLM Visual Prompt Preparation (Phase 1.5)
-------------------------------------------------------------------------
DESCRIPTION:
Takes raw frames and YOLO coordinates to produce 'boxed' images in the 
'qwen_input' directory. This 'visual prompting' technique is what 
allows the VLM to perform specific role assignment (e.g., 'Target 1 is 
the Officer').
-------------------------------------------------------------------------
"""

import cv2
import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
YOLO_JSON = 'yolo_detections.json'  # Your YOLO output
SOURCE_DIR = 'keyframes'
TARGET_DIR = 'qwen_input'          # Where the "labeled" images will go

def prepare_labeled_frames():
    if not os.path.exists(YOLO_JSON):
        print(f"❌ Error: {YOLO_JSON} not found. Run YOLO detector first.")
        return

    with open(YOLO_JSON, 'r') as f:
        video_data = json.load(f)

    os.makedirs(TARGET_DIR, exist_ok=True)

    for video_name, frames in video_data.items():
        print(f"\n🎨 Drawing Labels for: {video_name}")
        output_video_dir = os.path.join(TARGET_DIR, video_name)
        os.makedirs(output_video_dir, exist_ok=True)

        for frame_entry in tqdm(frames, desc="Processing Frames"):
            frame_name = frame_entry['frame']
            detections = frame_entry['detections']
            
            img_path = os.path.join(SOURCE_DIR, video_name, frame_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            h, w, _ = img.shape

            for i, det in enumerate(detections):
                ymin, xmin, ymax, xmax = det['bbox_2d']
                
                # Convert 0-1000 scale to pixel coordinates
                start_point = (int(xmin * w / 1000), int(ymin * h / 1000))
                end_point = (int(xmax * w / 1000), int(ymax * h / 1000))

                # Draw high-visibility Green Box
                cv2.rectangle(img, start_point, end_point, (0, 255, 0), 3)
                
                # Draw ID Tag (TARGET_0, TARGET_1, etc.)
                label = f"TARGET_{i}"
                label_pos = (start_point[0], max(30, start_point[1] - 10))
                
                # Black background for text for better readability
                cv2.rectangle(img, (label_pos[0], label_pos[1]-25), 
                              (label_pos[0]+130, label_pos[1]+5), (0,0,0), -1)
                cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)

            # Save the "Pre-Processed" frame
            cv2.imwrite(os.path.join(output_video_dir, frame_name), img)

    print(f"\n✨ Frames prepared in '{TARGET_DIR}'. Now run vision_analyze.py on this folder.")

if __name__ == "__main__":
    prepare_labeled_frames()