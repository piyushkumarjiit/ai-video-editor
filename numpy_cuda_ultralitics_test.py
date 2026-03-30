import numpy as np
import cv2
from ultralytics import YOLO

print(f"NumPy Version: {np.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
try:
    # Use the face model since it's already verified
    model = YOLO('models/yolov8n-face.pt')
    print("✅ Ultralytics can load models with this NumPy version.")
except Exception as e:
    print(f"❌ Still failing: {e}")