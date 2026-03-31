import cv2
import numpy as np
import json
import os

def create_test_resources():
    # 1. Create a Synthetic Video (5 seconds, 30fps)
    video_path = "samples/test_motion.mp4"
    os.makedirs("samples", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    print(f"Generating synthetic video: {video_path}")
    for i in range(150): # 150 frames = 5 seconds
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Move a white square diagonally
        pos = i * 3
        cv2.rectangle(frame, (pos, pos), (pos + 50, pos + 50), (255, 255, 255), -1)
        out.write(frame)
    out.release()

    # 2. Create a Sparse "UI Manifest" (Simulating AI detection)
    # We only provide the first and last positions.
    manifest_path = "ui_manifest_test.json"
    test_data = {
        "entity_001": [
            {"frame_number": 0, "x": 0, "y": 0, "w": 50, "h": 50},
            {"frame_number": 149, "x": 447, "y": 447, "w": 50, "h": 50}
        ]
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"Generated sparse manifest: {manifest_path}")

if __name__ == "__main__":
    create_test_resources()