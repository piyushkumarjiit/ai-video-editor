# Quick test for checking if using HW cuda for frame encoding/deconding would work

import cv2
import os

# Absolute path to your video
VIDEO_PATH = "/home/pk/ai-video-editor/samples/sanitized/Karen_Fights_Real_Estate_Agent_Over_House.mp4"

try:
    print("🚀 Attempting to initialize CUDA Video Reader...")
    
    # This creates a hardware decoder on Device 0
    gpu_reader = cv2.cudacodec.createVideoReader(VIDEO_PATH)
    
    # Read a frame directly into a GpuMat (GPU-side memory)
    ret, gpu_frame = gpu_reader.nextFrame()
    
    if ret:
        print("✅ SUCCESS! 1080 Ti is decoding video.")
        print(f"   - Frame type: {type(gpu_frame)}")
        print(f"   - Dimensions: {gpu_frame.size()}")
        
        # To actually see it, we have to download it to CPU
        cpu_frame = gpu_frame.download()
        print("✅ Successfully downloaded GPU frame to Host RAM.")
    else:
        print("❌ Reader initialized but failed to pull a frame.")

except AttributeError:
    print("❌ Your OpenCV build has 'cudacodec' but the Python bindings are missing 'createVideoReader'.")
except Exception as e:
    print(f"❌ CUDA Decoder Error: {e}")