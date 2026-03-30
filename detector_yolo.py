import cv2
import os
from ultralytics import YOLO

# Load the face detection model
model_path = os.path.join("models", "yolov8n-face.pt")
model = YOLO(model_path)

# Ensure the model uses the 1080 Ti
model.to('cuda')

def run_test_detection(video_path):
    print(f"--- Processing {video_path} on GPU ---")
    results = model.predict(source=video_path, save=False, stream=True, conf=0.25)
    
    for result in results:
        # This will show you how many faces it found in the current frame
        if len(result.boxes) > 0:
            print(f"Detected {len(result.boxes)} face(s)")

if __name__ == "__main__":
    # Test on your sample motion video
    run_test_detection("samples/test_motion.mp4")