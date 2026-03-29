import cv2
import numpy as np

def detect_scenes_gpu(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    scene_changes = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height * 3 # 3 channels (BGR)

    # Initialize GPU Mats
    gpu_prev_frame = cv2.cuda_GpuMat()
    gpu_curr_frame = cv2.cuda_GpuMat()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Upload current frame to GPU
        gpu_curr_frame.upload(frame)
        
        # 2. Convert to Grayscale or stay BGR (Grayscale is faster for diffs)
        # For simplicity and accuracy in color-based cuts, we stay in BGR here
        
        if frame_count > 0:
            # 3. Calculate Absolute Difference on GPU
            # This is the "Motion/Change" map
            gpu_diff = cv2.cuda.absdiff(gpu_curr_frame, gpu_prev_frame)
            
            # 4. Sum all pixel differences on GPU
            # Returns a Scalar (sum of B, G, R channels)
            diff_sum = cv2.cuda.sum(gpu_diff)
            
            # 5. Calculate average change per pixel
            avg_diff = sum(diff_sum) / total_pixels
            
            # If the change exceeds threshold, mark a scene cut
            if avg_diff > threshold:
                timestamp = frame_count / fps
                scene_changes.append((frame_count, timestamp))
                print(f"Scene change detected at frame {frame_count} ({timestamp:.2f}s) - Score: {avg_diff:.2f}")

        # Move current to previous for next iteration
        gpu_curr_frame.copyTo(gpu_prev_frame)
        frame_count += 1

    cap.release()
    return scene_changes

if __name__ == "__main__":
    # Lower threshold is more sensitive; 30-40 is standard for hard cuts
    scenes = detect_scenes_gpu("input_video.mp4", threshold=35.0)
    print(f"Total scenes found: {len(scenes)}")