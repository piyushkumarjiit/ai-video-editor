import cv2
import json
import os
import numpy as np

# --- CONFIGURATION ---
ANALYSIS_FILE = 'video_analysis.json'
DEBUG_OUTPUT = 'debug_grounding.jpg'

def run_quick_test():
    # 1. Load the standardized JSON
    if not os.path.exists(ANALYSIS_FILE):
        print(f"❌ Error: {ANALYSIS_FILE} not found.")
        return

    with open(ANALYSIS_FILE, 'r') as f:
        data = json.load(f)

    # 2. Pick the first video and frame that actually has detections
    # This prevents testing on an empty frame (like Frame 001 in some videos)
    target_video = None
    target_frame_data = None

    for video, frames in data.items():
        for frame in frames:
            if frame.get('detections'): # Find the first frame with AI data
                target_video = video
                target_frame_data = frame
                break
        if target_video: break

    if not target_video:
        print("⚠️ No detections found in any frames to test.")
        return

    img_name = target_frame_data['frame']
    img_path = os.path.join("keyframes", target_video, img_name)
    
    print(f"📂 Testing Video: {target_video}")
    print(f"🖼️ Testing Frame: {img_name}")

    # 3. Robust Image Load (Byte-level for Linux R720 path compatibility)
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return

    file_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        print("❌ Error: OpenCV could not decode image.")
        return

    h, w, _ = img.shape

    # 4. Draw Boxes [ymin, xmin, ymax, xmax]
    for det in target_frame_data['detections']:
        bbox = det['bbox_2d']
        label = det['id']
        
        # Mapping Qwen-VL [ymin, xmin, ymax, xmax] to pixels
        ymin, xmin, ymax, xmax = bbox
        
        start_point = (int(xmin * w / 1000), int(ymin * h / 1000))
        end_point = (int(xmax * w / 1000), int(ymax * h / 1000))

        # Draw green box
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 3)
        
        # Draw label
        cv2.putText(img, label, (start_point[0], start_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. Save and Finish
    cv2.imwrite(DEBUG_OUTPUT, img)
    print(f"✨ Success! Verification image created: {DEBUG_OUTPUT}")

if __name__ == "__main__":
    run_quick_test()