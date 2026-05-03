import cv2
import numpy as np

def test_cuda_video():
    print(f"OpenCV Version: {cv2.__version__}")
    
    # 1. Check for CUDA Devices
    gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
    if gpu_count == 0:
        print("❌ No CUDA devices found. Check your drivers.")
        return
    
    device_info = cv2.cuda.getDevice()
    print(f"✅ Found {gpu_count} GPU(s). Using: {cv2.cuda.printShortCudaDeviceInfo(device_info)}")

    # 2. Test Hardware Decoding (NVCUVID)
    # We create a reader. Even without a file, we check if the constructor exists.
    try:
        # Initializing with an empty string or dummy path to check for constructor access
        dummy_reader = cv2.cudacodec.createVideoReader("")
        print("✅ NVCUVID (Decoder) initialized successfully.")
    except cv2.error as e:
        if "empty" in str(e).lower() or "can't open" in str(e).lower():
            print("✅ NVCUVID (Decoder) is linked and functional.")
        else:
            print(f"❌ NVCUVID Error: {e}")

def test_nvenc():
    print(f"Testing OpenCV {cv2.__version__} NVENC...")
    
    # Create a dummy image (Black frame)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    try:
        # We try to create the writer directly. 
        # Using H264 codec (NVIDIA hardware accelerated)
        writer = cv2.cudacodec.createVideoWriter(
            "test_output.mp4", 
            (640, 480), 
            cv2.cudacodec.H264
        )
        
        # Try to write one frame to see if it actually executes
        writer.write(gpu_frame)
        print("✅ NVCUVENC (Encoder) successfully initialized and wrote a frame!")
        
    except AttributeError:
        print("❌ Error: 'cudacodec' exists but 'createVideoWriter' attribute is missing.")
        print("Checking available attributes in cudacodec...")
        print(dir(cv2.cudacodec))
    except Exception as e:
        print(f"❌ NVCUVENC linkage failed: {e}")

if __name__ == "__main__":
    test_cuda_video()
    test_nvenc()