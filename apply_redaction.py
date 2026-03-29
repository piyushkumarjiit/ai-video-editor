import cv2
import numpy as np

def redact_frame_gpu(frame, detections):
    # 1. Upload frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # 2. Create the blur filter (reusable for performance)
    # ksize is the blur intensity
    cuda_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (51, 51), 0)

    for det in detections:
        # Get coordinates (ensure they are translated to pixels first)
        x1, y1, x2, y2 = det['box']
        
        # Define the ROI (Region of Interest) on GPU
        roi_gpu = cv2.cuda_GpuMat(gpu_frame, (y1, y2 - y1, x1, x2 - x1))
        
        # Apply blur to the ROI directly in GPU memory
        blurred_roi = cuda_blur.apply(roi_gpu)
        blurred_roi.copyTo(roi_gpu)

    # 3. Download the fully processed frame back to CPU
    return gpu_frame.download()