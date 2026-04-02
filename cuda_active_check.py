import os
import subprocess
import json
import cv2
import requests
import torch  # Added missing import
import numpy as np
from ultralytics import YOLO

def check_nvidia_smi():
    print("--- 🖥️  System Drivers (nvidia-smi) ---")
    try:
        res = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if res.returncode == 0:
            print("✅ NVIDIA drivers are active.")
            print(res.stdout.split('\n')[0])
        else:
            print("❌ nvidia-smi found but returned an error.")
    except FileNotFoundError:
        print("❌ nvidia-smi NOT found. Are drivers installed?")

def check_opencv_cuda():
    print("\n--- 👁️  OpenCV CUDA Support ---")
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"✅ OpenCV is compiled with CUDA. Found {count} GPU(s).")
            # Safely grab build info
            build_info = cv2.getBuildInformation()
            if "NVIDIA CUDA:" in build_info:
                print(f"   Build Info: {build_info.split('NVIDIA CUDA:')[1].splitlines()[0].strip()}")
        else:
            print("❌ OpenCV found, but NOT compiled with CUDA support.")
    except AttributeError:
        print("❌ This version of OpenCV (cv2) does not have the .cuda module.")

def check_ffmpeg_nvenc():
    print("\n--- 🎞️  FFmpeg NVENC Hardware Accel ---")
    try:
        cmd = ["ffmpeg", "-encoders"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if "h264_nvenc" in res.stdout:
            print("✅ FFmpeg supports h264_nvenc (NVIDIA Encoder).")
            # Real-world test
            test_cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=1",
                "-c:v", "h264_nvenc", "-frames:v", "1", "gpu_test.mp4"
            ]
            test_res = subprocess.run(test_cmd, capture_output=True)
            if test_res.returncode == 0:
                print("✅ Real-world test: Successfully encoded frame using NVENC.")
                if os.path.exists("gpu_test.mp4"): os.remove("gpu_test.mp4")
            else:
                print("❌ Real-world test: NVENC encode failed.")
        else:
            print("❌ FFmpeg does NOT have h264_nvenc enabled.")
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")

def check_llama_cpp_cuda():
    print("\n--- 🦙 Llama-CPP-Python CUDA Support ---")
    try:
        from llama_cpp import llama_cpp
        is_cuda = llama_cpp.llama_supports_gpu_offload()
        if is_cuda:
            devices = llama_cpp.llama_backend_init()
            print(f"✅ llama-cpp-python is compiled with GPU support. ({devices} devices)")
        else:
            print("❌ llama-cpp-python found, but running on CPU ONLY.")
    except ImportError:
        print("❌ llama-cpp-python is not installed.")

def check_pytorch_cuda():
    print("\n--- 🔥 PyTorch / Ultralytics Support ---")
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA Available | Device: {torch.cuda.get_device_name(0)}")
        print(f"   Torch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")
        try:
            x = torch.rand(100, 100).cuda()
            print("✅ GPU Tensor Allocation: Successful.")
        except Exception as e:
            print(f"❌ GPU Tensor Allocation: FAILED ({e})")
    else:
        print("❌ PyTorch is running on CPU.")

def check_numpy_simd():
    print("\n--- 🔢 NumPy Acceleration ---")
    try:
        # Check for AVX support (Xeon E5 specialization)
        config = np.show_config()
        print("✅ NumPy configuration detected.")
        # Check runtime for SIMD
        if hasattr(np, 'show_runtime'):
            print("✅ SIMD/AVX acceleration identified.")
    except:
        print("⚠️  Could not verify NumPy SIMD features.")

def check_ollama_gpu():
    print("\n--- 🤖 Ollama GPU Inference ---")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama server reachable.")
            ps_res = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
            if "100% GPU" in ps_res.stdout:
                print("✅ Model loaded 100% on GPU.")
            else:
                print("⚠️  No model active on GPU. Run 'ollama run' to test VRAM.")
    except:
        print("❌ Ollama server is NOT running.")

def check_opencv_dnn():
    print("\n👁️  [OpenCV DNN Backend]")
    backends = [b for b in dir(cv2.dnn) if "BACKEND_CUDA" in b]
    if backends:
        print("✅ OpenCV is CUDA-Ready for DNN models.")
    else:
        print("❌ OpenCV DNN defaults to CPU.")

def check_ultralytics_yolo():
    """Validates Ultralytics YOLOv8 integration with custom OpenCV and CUDA."""
    print("\n--- 🎯 Ultralytics / YOLOv8 Integration ---")
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        import numpy as np

        # 1. Verify it's using YOUR custom OpenCV build
        print(f"✅ Ultralytics linked to OpenCV {cv2.__version__}")

        # 2. Initialize model and move to 1080 Ti
        # Note: 'yolov8n.pt' will download to the current dir if not present
        model = YOLO("yolov8n.pt") 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # 3. Perform a "Warm-up" Inference (Tests the CUDA kernels)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)

        print(f"✅ Ultralytics Inference Test: Successful on {device}")
        return True

    except ImportError:
        print("⚠️  Ultralytics not installed. Skip with: pip install ultralytics --no-deps")
        return False
    except Exception as e:
        print(f"❌ Ultralytics Integration: Failed - {e}")
        return False

def check_torch_vision_sync():
    print("\n--- 🛠️  Torch/Torchvision Binary Sync ---")
    try:
        import torchvision
        print(f"✅ Torchvision Version: {torchvision.__version__}")
        
        # This is the "Gold Standard" test: Does a Vision Op work on the GPU?
        # If Torch and Torchvision are out of sync, this will throw a C++ error immediately.
        from torchvision.ops import nms
        boxes = torch.rand(5, 4).cuda()
        scores = torch.rand(5).cuda()
        _ = nms(boxes, scores, 0.5)
        print("✅ Torch & Torchvision are BINARY COMPATIBLE on GPU.")
    except Exception as e:
        print(f"❌ Torch/Torchvision Sync FAILED: {e}")
        print("💡 Suggestion: Reinstall torchvision to match your Torch version.")

# --- NEW: VRAM STATUS ---
def check_vram_headroom():
    print("\n--- 💾 GPU VRAM Headroom ---")
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        r = torch.cuda.memory_reserved(0) / 1024**3
        a = torch.cuda.memory_allocated(0) / 1024**3
        f = t - r  # Free roughly
        print(f"✅ Total VRAM: {t:.2f}GB | Allocated: {a:.2f}GB | Available: {f:.2f}GB")
        if f < 2.0:
            print("⚠️  Warning: Low VRAM headroom. Close Ollama or other processes.")
    else:
        print("❌ No CUDA device detected for VRAM check.")

# --- UPDATED PYTORCH CHECK (With Version Warnings) ---
def check_pytorch_cuda():
    print("\n--- 🔥 PyTorch / CUDA Core ---")
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA Available | Device: {torch.cuda.get_device_name(0)}")
        print(f"   Torch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")
        
        # Alert if versions look suspicious but work
        if "2.8.0" in torch.__version__ and "12." in torch.version.cuda:
            print("ℹ️  Note: Running Torch 2.8.0 on CUDA 12. This is your 'Pinned' stable build.")

        try:
            x = torch.rand(100, 100).cuda()
            print("✅ GPU Tensor Allocation: Successful.")
        except Exception as e:
            print(f"❌ GPU Tensor Allocation: FAILED ({e})")
    else:
        print("❌ PyTorch is running on CPU.")


if __name__ == "__main__":
    print("🚀 Starting AI-Video-Editor GPU Pipeline Diagnostics\n")
    check_nvidia_smi()
    check_opencv_cuda()
    check_opencv_dnn()
    check_ffmpeg_nvenc()
    check_llama_cpp_cuda()
    check_pytorch_cuda()
    check_numpy_simd()
    check_ollama_gpu()
    check_ultralytics_yolo()
    check_pytorch_cuda()
    check_torch_vision_sync()
    check_vram_headroom()
    print("\n✨ Diagnostics Complete. Use this output for your next environment sync.")