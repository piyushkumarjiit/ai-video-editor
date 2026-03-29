import cv2
import subprocess
import numpy as np

def get_nvenc_writer(output_path, width, height, fps):
    """Initializes an FFmpeg process configured for NVIDIA Hardware Encoding."""
    command = [
        'ffmpeg',
        '-y',                 # Overwrite output file
        '-f', 'rawvideo',      # Input is raw pixels
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',   # OpenCV default pixel format
        '-r', str(fps),
        '-i', '-',            # Read from stdin (the pipe)
        '-c:v', 'h264_nvenc', # Use NVIDIA Hardware Encoder
        '-preset', 'p4',      # P1 (fastest) to P7 (slowest/best)
        '-cq', '24',          # Constant Quality (lower is better)
        '-pix_fmt', 'yuv420p',# Standard format for player compatibility
        output_path
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

def process_video_gpu(input_path, output_path, detections_by_frame):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the GPU-accelerated FFmpeg writer
    writer_proc = get_nvenc_writer(output_path, width, height, fps)

    # Pre-create the CUDA Blur Filter (Reusing it saves memory allocation time)
    # (51, 51) is the kernel size; increase for more "unreadable" redaction
    cuda_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (51, 51), 0)
    
    # GPU Memory allocation
    gpu_frame = cv2.cuda_GpuMat()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Upload to GPU
        gpu_frame.upload(frame)

        # 2. Apply Redaction if detections exist for this frame
        if frame_idx in detections_by_frame:
            for box in detections_by_frame[frame_idx]:
                # SCALE COORDS: x1, y1, x2, y2 = scale_coords(box, width, height)
                x1, y1, x2, y2 = box 
                
                # Create a View (ROI) of the box in GPU memory
                # Note: OpenCV uses (x, y, width, height) for ROI rectangles
                roi = cv2.cuda_GpuMat(gpu_frame, (x1, y1, x2-x1, y2-y1))
                
                # Apply blur directly to the GPU ROI
                cuda_filter.apply(roi, roi)

        # 3. Download and Pipe to NVENC
        # We download back to CPU because FFmpeg pipe expects a byte stream
        final_frame = gpu_frame.download()
        writer_proc.stdin.write(final_frame.tobytes())
        
        frame_idx += 1

    # Cleanup
    cap.release()
    writer_proc.stdin.close()
    writer_proc.wait()