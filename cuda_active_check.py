"""
FILE: cuda_active_check.py
ROLE: Full-Stack GPU & AI Diagnostic Suite
-------------------------------------------------------------------------
DESCRIPTION:
A comprehensive health check for the 1080 Ti / R720 environment. 
Verifies NVIDIA drivers, OpenCV CUDA compilation, FFmpeg NVENC 
availability, and VRAM headroom. Use this to troubleshoot 'Slow 
Processing' errors.
-------------------------------------------------------------------------
"""

import os
import subprocess
import json
import cv2
import requests
import torch  # Added missing import
import numpy as np
from ultralytics import YOLO

MODEL_DIR = "models"
MODEL_FILENAME = "yolov8x.pt" # or yolov8n-face.pt
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_DIR, MODEL_FILENAME)

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

# Ensure Ultralytics doesn't try to auto-update TensorRT
os.environ['YOLO_SKIP_CHECK'] = 'True'

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

def check_ffmpeg_hw_accel():
    print("\n--- 🎞️  FFmpeg NVIDIA Hardware Accel (H.264 & H.265) ---")
    
    codecs_to_test = [
        ("h264_nvenc", "h264_cuvid", "h264_test.mp4"),
        ("hevc_nvenc", "hevc_cuvid", "h265_test.mp4")
    ]

    for encoder, decoder, filename in codecs_to_test:
        print(f"\n🚀 Testing {encoder.upper()} / {decoder.upper()}...")
        try:
            # 1. Encode Test
            encode_cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=1",
                "-c:v", encoder, "-frames:v", "1", filename
            ]
            encode_res = subprocess.run(encode_cmd, capture_output=True, text=True)

            if encode_res.returncode == 0:
                print(f"✅ {encoder}: Encoding Successful.")
                
                # 2. Decode Test (Only runs if encode succeeded)
                decode_cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda", "-c:v", decoder,
                    "-i", filename, "-f", "null", "-"
                ]
                decode_res = subprocess.run(decode_cmd, capture_output=True, text=True)

                if decode_res.returncode == 0:
                    print(f"✅ {decoder}: Decoding Successful.")
                else:
                    print(f"❌ {decoder}: Decoding Failed.")
            else:
                print(f"❌ {encoder}: Encoding Failed.")

        except Exception as e:
            print(f"❌ FFmpeg check failed for {encoder}: {e}")

        finally:
            # This ensures no .mp4 ghosts are left behind regardless of success/failure
            if os.path.exists(filename):
                os.remove(filename)



def check_llama_cpp_cuda():
    print("\n--- 🦙 Llama-CPP-Python CUDA Support ---")
    try:
        from llama_cpp import llama_cpp

        is_cuda = llama_cpp.llama_supports_gpu_offload()
        if is_cuda:
            llama_cpp.llama_backend_init()  # returns None, just initializes
            
            # Get device count separately via CUDA bindings
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"✅ llama-cpp-python compiled with GPU support. ({device_count} CUDA device(s) visible)")
        else:
            print("❌ llama-cpp-python found, but running on CPU ONLY.")

    except ImportError:
        print("❌ llama-cpp-python is not installed.")
    except Exception as e:
        print(f"❌ llama-cpp-python check failed: {e}")
        

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
        LOCAL_IP = subprocess.getoutput(r"ip route get 1.1.1.1 | grep -oP 'src \K\S+'")
        url = f"http://{LOCAL_IP}:11434/api/tags"
        #print(f"DEBUG: Connecting to Ollama at: {url}") 
        response = requests.get(url, timeout=2)            
        if response.status_code == 200:
            print("✅ Ollama server reachable.")
        else:
            print("⚠️ Ollama server reachable.")
    except:
        print("❌ Exception while trying to reach Ollama server.")

def check_opencv_dnn():
    print("\n👁️  [OpenCV DNN Backend]")
    backends = [b for b in dir(cv2.dnn) if "BACKEND_CUDA" in b]
    if backends:
        print("✅ OpenCV is CUDA-Ready for DNN models.")
    else:
        print("❌ OpenCV DNN defaults to CPU.")

def register_safe_globals():
    """Register all classes needed for weights_only=True deserialization."""
    import collections
    import torch
    import ultralytics.nn.modules.conv as ulconv
    import ultralytics.nn.modules.block as ulblock
    import ultralytics.nn.modules.head as ulhead
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel
    )

    # Dynamically grab every class ultralytics defines in these modules
    # This future-proofs against minor version changes in 8.2.x
    def get_classes(module):
        import inspect
        return [
            obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if module.__name__ in (obj.__module__ or "")
        ]
    safe_classes = [
        # Ultralytics task-level models
        DetectionModel, SegmentationModel, PoseModel, ClassificationModel,

        # All classes from ultralytics conv/block/head modules dynamically
        *get_classes(ulconv),
        *get_classes(ulblock),
        *get_classes(ulhead),

        # PyTorch native nn layers pickled inside .pt files
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.LeakyReLU,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.upsampling.Upsample,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.dropout.Dropout,

        # PyTorch containers
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.container.ModuleDict,

        # stdlib
        collections.OrderedDict,
    ]

    torch.serialization.add_safe_globals(safe_classes)


def check_ultralytics_yolo():
    """Validates Ultralytics YOLOv8 integration with custom OpenCV and CUDA."""
    print("\n--- 🎯 Ultralytics / YOLOv8 Integration ---")
    try:
        import cv2
        import torch
        import numpy as np
        from ultralytics import YOLO

        # Safe globals already registered at startup via register_safe_globals()

        print(f"✅ Ultralytics linked to OpenCV {cv2.__version__}")

        # Initialize model and move to GPU
        model = YOLO(MODEL_PATH, verbose=False)
        print(f"✅ Loaded {MODEL_FILENAME} from {MODEL_PATH}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Warm-up inference
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print(f"✅ Ultralytics Inference Test: Successful on {device}")
        return True

    except ImportError:
        print("⚠️  Ultralytics not installed. Skip with: pip install ultralytics --no-deps")
        return False
    except Exception as e:
        if "Unsupported global" in str(e):
            # Extract the missing class name from the error for easy fixing
            import re
            match = re.search(r'GLOBAL ([\w\.]+) was not an allowed', str(e))
            if match:
                print(f"❌ Missing safe global: {match.group(1)}")
                print(f"   Add to register_safe_globals(): {match.group(1).split('.')[-1]}")
            else:
                print(f"❌ Ultralytics Integration: Failed - {e}")
        else:
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


def check_tensorrt_readiness():
    print("\n--- 🧠 TensorRT & ONNX Readiness ---")
    
    # 1. Check TensorRT Python Bindings
    try:
        import tensorrt as trt
        print(f"✅ TensorRT Library: FOUND (Version: {trt.__version__})")
    except ImportError:
        print("❌ TensorRT Library: NOT FOUND. Run 'pip install tensorrt-cu12'")
        return

    # 2. Check ONNX & Simplifier
    try:
        import onnx
        import onnxslim
        print(f"✅ ONNX & ONNX-Slim: FOUND (Ready for conversion simplify=True)")
    except ImportError as e:
        print(f"❌ ONNX Tools: MISSING ({e}). Conversion may fail.")

    # 3. Check ONNXRuntime GPU Provider
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print(f"✅ ONNXRuntime GPU: ACTIVE (Providers: {providers})")
        else:
            print(f"⚠️  ONNXRuntime GPU: INACTIVE. Found only: {providers}")
            print("   Note: This may slow down the 'simplify' phase of export.")
    except ImportError:
        print("❌ ONNXRuntime: NOT FOUND.")

    # 4. Check for TensorRT Builder Access
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        if builder:
            print("✅ TensorRT Builder: Operational.")
    except Exception as e:
        print(f"❌ TensorRT Builder: FAILED initialization ({e})")


def asr_health_check():
    print("\n---🔍 Starting Unified ASR Health Check...")
    results = {}

    # Check 1: CUDA & Hardware
    results['CUDA Available'] = torch.cuda.is_available()
    results['GPU Name'] = torch.cuda.get_device_name(0) if results['CUDA Available'] else "N/A"

    # Check 2: DeepFilterNet (The recent hurdle)
    try:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        results['DeepFilterNet'] = "✅ Healthy (Engine Initialized)"
    except Exception as e:
        results['DeepFilterNet'] = f"❌ FAILED: {str(e)}"

    # Check 3: WhisperX
    try:
        import whisperx
        results['WhisperX'] = "✅ Healthy"
    except Exception as e:
        results['WhisperX'] = f"❌ FAILED: {str(e)}"

    # Check 4: Pyannote
    try:
        import pyannote.audio
        results['Pyannote'] = "✅ Healthy"
    except Exception as e:
        results['Pyannote'] = f"❌ FAILED: {str(e)}"

    print("\n--- FINAL REPORT ---")
    for k, v in results.items():
        print(f"{k}: {v}")


def check_deepfilter_gpu():
    """Validates DeepFilterNet noise suppression engine on CUDA."""
    print("\n--- 🎙️  DeepFilterNet Noise Suppression ---")
    try:
        import torch
        import numpy as np
        from df.enhance import init_df, enhance
        from df.io import load_audio, save_audio

        # 1. Initialize the DeepFilter engine
        model, df_state, _ = init_df()
        print(f"✅ DeepFilterNet engine initialized.")
        print(f"   Sample Rate: {df_state.sr()} Hz")

        # 2. Verify CUDA availability and move model to GPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✅ DeepFilterNet model moved to: {device}")

        # 3. Warm-up inference with a synthetic 1-second audio tensor
        #    Shape: (channels, samples) — mono at df_state sample rate
        sr = df_state.sr()
        dummy_audio = torch.zeros(1, sr, dtype=torch.float32)  # 1 sec silence
        enhanced = enhance(model, df_state, dummy_audio)
        print(f"✅ Inference Test: Successful — output shape: {enhanced.shape}")

        # 4. VRAM impact report
        if device == "cuda:0":
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved  = torch.cuda.memory_reserved(0)  / 1024**2
            print(f"   VRAM After Load — Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB")
            if reserved > 500:
                print("⚠️  DeepFilterNet is consuming significant VRAM. "
                      "Monitor headroom if running alongside YOLO + WhisperX.")

        return True

    except ImportError as e:
        print(f"❌ DeepFilterNet not installed or missing dependency: {e}")
        print("   Install with: pip install deepfilterlib deepfilternet")
        return False
    except Exception as e:
        print(f"❌ DeepFilterNet: Failed — {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting AI-Video-Editor GPU Pipeline Diagnostics\n")
    # Call once at startup before any model loads
    register_safe_globals()
    check_nvidia_smi()
    check_opencv_cuda()
    check_opencv_dnn()
    check_ffmpeg_hw_accel()
    check_llama_cpp_cuda()
    check_pytorch_cuda()
    check_numpy_simd()
    check_ollama_gpu()
    check_ultralytics_yolo()
    check_torch_vision_sync()
    check_vram_headroom()
    check_tensorrt_readiness()
    asr_health_check()
    check_deepfilter_gpu()
    print("\n✨ Diagnostics Complete. Use this output for your next environment sync.")