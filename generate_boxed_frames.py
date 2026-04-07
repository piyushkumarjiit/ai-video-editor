"""
FILE: generate_boxed_frames.py
STATUS: DEPRECATED / LEGACY
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

def draw_boxes_for_qwen(video_name, frame_data):
    img_path = f"keyframes/{video_name}/{frame_data['frame']}"
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    for i, det in enumerate(frame_data['detections']):
        ymin, xmin, ymax, xmax = det['bbox_2d']
        start = (int(xmin * w / 1000), int(ymin * h / 1000))
        end = (int(xmax * w / 1000), int(ymax * h / 1000))
        
        # Draw a bright box and a large ID label
        cv2.rectangle(img, start, end, (0, 255, 0), 2)
        cv2.putText(img, f"TARGET_{i}", (start[0], start[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save to a temporary "labeled" folder for Qwen to look at
    os.makedirs("qwen_input", exist_ok=True)
    cv2.imwrite(f"qwen_input/{frame_data['frame']}", img)