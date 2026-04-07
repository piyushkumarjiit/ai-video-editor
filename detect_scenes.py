"""
FILE: detect_scenes.py
STATUS: DEPRECATED / LEGACY
-------------------------------------------------------------------------
DESCRIPTION:
An early-stage utility for segmenting video into 10-second 'scenes' based 
on a fixed frame count. 

REASON FOR DEPRECATION:
Superseded by more efficient frame extraction methods that support 
variable FPS and targeted incident detection. Fixed-interval sampling 
(300 frames) often misses the exact start of critical incidents.
-------------------------------------------------------------------------
"""

import cv2
import os
import json

def extract_scenes(video_path, output_dir="samples/scenes", threshold=30.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scene_count = 0
    start_frame = 0
    current_frame = 0
    
    print(f"🎬 Processing Video: {width}x{height} @ {fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple Logic: Save metadata every 300 frames (approx 10s) 
        # or use a real scene-change detection threshold here.
        if current_frame % 300 == 0 and current_frame > 0:
            scene_id = f"scene_{scene_count:03d}"
            scene_path = os.path.join(output_dir, scene_id)
            if not os.path.exists(scene_path): os.makedirs(scene_path)

            # SAVE METADATA WITH GLOBAL OFFSET
            metadata = {
                "scene_id": scene_id,
                "global_start_frame": start_frame,
                "global_end_frame": current_frame,
                "resolution": {"w": width, "h": height},
                "fps": fps
            }
            
            with open(os.path.join(scene_path, "details.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Save a reference keyframe for the AI to look at
            cv2.imwrite(os.path.join(scene_path, "keyframe.jpg"), frame)
            
            print(f"✅ Scene {scene_id} recorded (Start: {start_frame})")
            start_frame = current_frame
            scene_count += 1
            
        current_frame += 1

    cap.release()
    print(f"🚀 Total Scenes Extracted: {scene_count}")

if __name__ == "__main__":
    # Update this path to your sanitized video on the R720
    extract_scenes("samples/sanitized/input_video.mp4")