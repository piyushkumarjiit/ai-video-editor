FILE: cv_cuda_test.py
STATUS: DEPRECATED / LEGACY
import cv2
count = cv2.cuda.getCudaEnabledDeviceCount()
if count > 0:
    print(f"Success! Found {count} CUDA-capable device(s).")
else:
    print("No CUDA support detected.")