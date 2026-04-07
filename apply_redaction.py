"""
FILE: apply_redaction.py
ROLE: CUDA-Accelerated Video Redaction (Final Phase)
-------------------------------------------------------------------------
DESCRIPTION:
The final production script. It reads the tracking manifest and applies 
feathered Gaussian blurs to specified entities using NVIDIA CUDA 
acceleration. 

HARDWARE COMPATIBILITY:
- REQUIRES: OpenCV compiled with CUDA support.
- Performance: Highly optimized for Pascal/Turing/Ampere GPUs.
-------------------------------------------------------------------------
"""

import cv2
import numpy as np

def apply_feathered_blur_cuda(gpu_frame, bbox, blur_strength=51, feather_amount=0.3):
    """
    Uses NVIDIA CUDA to apply a feathered Gaussian blur.
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    
    # 1. Define ROI on the GPU
    # Note: CUDA ROIs are handled via GpuMat headers
    gpu_roi = cv2.cuda_GpuMat(gpu_frame, (y, y+h), (x, x+w))
    
    # 2. Create Gaussian Filter on GPU
    k_size = (blur_strength // 2 * 2 + 1)
    cuda_blur_filter = cv2.cuda.createGaussianFilter(gpu_roi.type(), gpu_roi.type(), (k_size, k_size), 0)
    
    # 3. Apply Blur
    gpu_blurred_roi = cuda_blur_filter.apply(gpu_roi)
    
    # 4. Generate the Feathered Mask (Created once on CPU, then uploaded)
    # Optimization: In a production loop, cache these masks by (w, h) dimensions
    mask = np.zeros((h, w), dtype=np.float32)
    inner_w, inner_h = int(w * (1 - feather_amount)), int(h * (1 - feather_amount))
    pad_w, pad_h = (w - inner_w) // 2, (h - inner_h) // 2
    cv2.rectangle(mask, (pad_w, pad_h), (pad_w + inner_w, pad_h + inner_h), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    
    gpu_mask = cv2.cuda_GpuMat()
    gpu_mask.upload(cv2.merge([mask, mask, mask])) # Upload 3-channel mask
    
    # 5. GPU Alpha Blending: Result = (Blur * Mask) + (Original * (1 - Mask))
    # We use cuda.multiply and cuda.add for high-speed arithmetic
    term1 = cv2.cuda.multiply(gpu_blurred_roi.convertTo(cv2.CV_32F), gpu_mask)
    
    inv_mask = cv2.cuda.subtract(cv2.cuda_GpuMat(gpu_mask.size(), gpu_mask.type(), (1,1,1)), gpu_mask)
    term2 = cv2.cuda.multiply(gpu_roi.convertTo(cv2.CV_32F), inv_mask)
    
    gpu_blended = cv2.cuda.add(term1, term2).convertTo(cv2.CV_8U)
    
    # 6. Copy blended ROI back to the original GPU frame
    gpu_blended.copyTo(gpu_roi)
    
    return gpu_frame

def process_video_cuda(video_in, video_out, tracked_manifest):
    cap = cv2.VideoCapture(video_in)
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 
                             cap.get(cv2.CAP_PROP_FPS), 
                             (int(cap.get(3)), int(cap.get(4))))

    gpu_frame = cv2.cuda_GpuMat()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Upload current frame to GPU
        gpu_frame.upload(frame)
        
        str_idx = str(frame_idx)
        if str_idx in tracked_manifest:
            for item in tracked_manifest[str_idx]:
                gpu_frame = apply_feathered_blur_cuda(gpu_frame, item['bbox'])
        
        # Download result back to CPU for writing
        final_frame = gpu_frame.download()
        writer.write(final_frame)
        frame_idx += 1

    cap.release()
    writer.release()