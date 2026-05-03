# -------------------------------------------------------------------------
# FILE: recreate_env_tensor.sh
# ROLE: Rapid Environment Recovery & Deployment
#
# DESCRIPTION:
# A streamlined script for rebuilding the Python environment for use by TensorRT. It 
# enforces a hardware-optimized installation order to ensure Torch 
# components are correctly linked to CUDA 12.6 along with Audio and Vision Pipeline suite.
#
# HARDWARE COMPATIBILITY:
# - Forces Torch installation via the cu126 index for 1080 Ti support.
# - Configures Ultralytics,Pynote, DNN with --no-deps to prevent driver issues.
#
# #./recreate_env_tensor.sh
# -------------------------------------------------------------------------

#!/bin/bash
unset PYTHONPATH
BUILD_OPENCV=true

# Force correct library loading for the current session & health check
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export MKL_SERVICE_FORCE_INTEL=1

VENV_PATH=${1:-"$HOME/.virtualenvs/ai-video-tensor"}
TARGET_PYTHON=${2:-"python3.11"}
ROLE=${3:-"denoise"}
FORCE_REBUILD=${4:-false} # Optional 4th argument

if [[ -z "$VENV_PATH" || -z "$TARGET_PYTHON" || -z "$FORCE_REBUILD" ]]; then
    echo "❌ Usage: $0 [venv_name] [python_version] [denoise|asr|diarize] (force_rebuild)"
    echo "Proceeding with defaults VENV_PATH: $VENV_PATH, TARGET_PYTHON: $TARGET_PYTHON and ROLE: $ROLE"
fi

# 0. Prerequisites
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# install rust as it is used by deepfilternet
if ! command -v cargo &> /dev/null; then
    echo "🦀 Installing Rust for DeepFilterNet..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Check if the specific Python version is even installed
if ! command -v $TARGET_PYTHON &> /dev/null; then
    echo "❌ $TARGET_PYTHON not found. Installing via deadsnakes..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install $TARGET_PYTHON $TARGET_PYTHON-venv $TARGET_PYTHON-dev -y
fi

# Delete old venv if it exists
if [[ -d "$VENV_PATH" ]]; then
    echo "🗑️ Cleaning up old environment if it exists..."
    rm -rf "$VENV_PATH"
else
    echo "🗑️ No need to clean up old environment as it does not exists..."
fi

# 1. Create the environment using that exact version
echo "🐍 Creating Virtual Env with $($TARGET_PYTHON --version)..."
$TARGET_PYTHON -m venv $VENV_PATH

# 2. Activate and verify
echo "🔌 Activating environment..."
source "$VENV_PATH/bin/activate"

# 4. Base Tooling
pip install --upgrade pip==26.1 setuptools==82.0.1 wheel==0.47.0

# 5. Layer 0: Foundations & Networking
echo "📦 Installing Foundations & Networking..."
pip install --no-deps \
    attrs==26.1.0 distro==1.9.0 filelock==3.25.2 h11==0.16.0 \
    ml-dtypes==0.4.1 numpy==2.0.2 packaging==26.0 platformdirs==4.3.6 \
    psutil==7.2.2 python-dateutil==2.9.0.post0 pytz==2026.1.post1 \
    six==1.17.0 sniffio==1.3.1 typing-inspection==0.4.2 \
    typing_extensions==4.15.0 tzdata==2026.1 \
    cffi==2.0.0 pycparser==3.0 PySocks==1.7.1 \
    cryptography==46.0.7 python-dotenv==1.2.2 wcwidth==0.6.0

# 6. Layer 1: Communication & Google Stack
echo "📦 Installing Communication & Google Auth Stack..."
pip install --no-deps \
    aiohappyeyeballs==2.6.1 aiohttp==3.13.5 aiosignal==1.4.0 \
    certifi==2026.2.25 charset-normalizer==3.4.7 frozenlist==1.8.0 \
    google-api-core==2.30.3 google-api-python-client==2.194.0 \
    google-auth==2.49.2 google-auth-httplib2==0.3.1 \
    google-auth-oauthlib==1.3.1 googleapis-common-protos==1.74.0 \
    httpcore==1.0.9 httplib2==0.31.2 httpx==0.28.1 idna==3.11 \
    multidict==6.7.1 oauthlib==3.3.1 propcache==0.4.1 \
    proto-plus==1.27.2 protobuf==7.34.1 pyasn1==0.6.3 \
    pyasn1_modules==0.4.2 requests==2.33.1 requests-oauthlib==2.0.0 \
    rsa==4.9 uritemplate==4.2.0 urllib3==2.6.3 yarl==1.23.0 \
    fsspec==2026.2.0 jiter==0.14.0

# 7. Layer 2: Core AI Stack (Torch 2.7.1)
echo "🔥 Installing Torch 2.7.1 components for CUDA 12.6..."
pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 \
    torchaudio==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    triton==3.3.1

# 8. Layer 3: CUDA Dependecies (cu126)
echo "📦 Installing NVIDIA CUDA 12.6 Runtimes..."
pip install --no-deps \
    cuda-bindings==12.9.4 cuda-pathfinder==1.2.2 cuda-toolkit==12.6.3 \
    nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-cupti-cu12==12.6.80 \
    nvidia-cuda-nvrtc-cu12==12.6.77 nvidia-cuda-runtime-cu12==12.6.77 \
    nvidia-cudnn-cu12==9.5.1.17 nvidia-cufft-cu12==11.3.0.4 \
    nvidia-cufile-cu12==1.11.1.6 nvidia-curand-cu12==10.3.7.77 \
    nvidia-cusolver-cu12==11.7.1.2 nvidia-cusparse-cu12==12.5.4.2 \
    nvidia-cusparselt-cu12==0.6.3 nvidia-nccl-cu12==2.26.2 \
    nvidia-nvjitlink-cu12==12.6.85 nvidia-nvshmem-cu12==3.4.5 \
    nvidia-nvtx-cu12==12.6.77

# 9. Layer 4: Math, Data & Logic
echo "📦 Installing Math, Data & Analytics..."
pip install --no-deps \
    contourpy==1.3.3 cycler==0.12.1 fonttools==4.62.1 joblib==1.5.3 \
    kiwisolver==1.5.0 matplotlib==3.10.8 mpmath==1.3.0 networkx==3.6.1 \
    pandas==2.2.3 pyparsing==3.3.2 scikit-learn==1.8.0 scipy==1.17.1 \
    sympy==1.14.0 threadpoolctl==3.6.0 soxr==1.0.0 \
    nltk==3.9.4 optuna==4.8.0 primePy==1.3

# 10. Layer 5: Vision & ONNX
echo "📦 Installing Vision & Tracking..."
pip install --no-deps \
    ImageIO==2.37.3 boxmot==18.0.0 filterpy==1.4.5 flatbuffers==25.12.19 \
    lap==0.5.13 lapx==0.9.4 onnxruntime-gpu==1.20.0 onnxslim==0.1.71 onnx==1.17.0 \
    imageio-ffmpeg==0.6.0 pillow==11.3.0 screeninfo==0.8.1 \
    ultralytics==8.2.50 clip-anytorch==2.6.0 yacs==0.1.8

# 11. Layer 6: Transformers & Audio
echo "📦 Installing Transformers & Audio..."
pip install --no-deps \
    accelerate==1.13.0 einops==0.8.2 ftfy==6.3.1 huggingface_hub==1.10.1 \
    regex==2026.4.4 safetensors==0.7.0 sentencepiece==0.2.1 tokenizers==0.22.2 \
    tqdm==4.67.3 transformers==5.5.3 \
    asteroid-filterbanks==0.4.0 ctranslate2==4.7.1 faster-whisper==1.2.1 \
    HyperPyYAML==1.2.3 julius==0.2.7 pyannote.audio==3.4.0 \
    pyannote.core==5.0.0 pyannote.database==5.1.3 pyannote.metrics==3.2.1 \
    pyannote.pipeline==3.0.1 soundfile==0.13.1 speechbrain==1.1.0 \
    torch-audiomentations==0.12.0 torch_pitch_shift==1.2.5 \
    torchmetrics==1.9.0 whisperx==3.7.2

# 11.A Layer 6: Tensor RT for exporting .engine model. This is specific to the graphics card in question 1080TI
pip install --no-deps --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
    tensorrt==8.6.1.post1 tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1

# 12. Layer 7: Final Utilities
echo "📦 Installing Final Utilities..."
pip install --no-deps \
    Jinja2==3.1.6 Mako==1.3.10 MarkupSafe==3.0.3 Pygments==2.20.0 \
    SQLAlchemy==2.0.49 alembic==1.18.4 annotated-doc==0.0.4 \
    annotated-types==0.7.0 antlr4-python3-runtime==4.9.3 anyio==4.13.0 \
    av==15.1.0 beautifulsoup4==4.14.3 click==8.3.2 colorlog==6.10.1 \
    decorator==5.2.1 diskcache==5.6.3 docopt==0.6.2 gdown==6.0.0 omegaconf==2.3.0 \
    greenlet==3.4.0 hf-xet==1.4.3 lightning==2.6.1 lightning-utilities==0.15.3 \
    loguru==0.7.3 markdown-it-py==4.0.0 mdurl==0.1.2 moviepy==2.2.1 ollama==0.6.1 \
    openai==2.31.0 proglog==0.1.12 pydantic==2.13.0 pydantic_core==2.46.0 \
    pytorch-lightning==2.6.1 pyyaml==6.0.3 rich==15.0.0 ruamel.yaml==0.18.17 \
    ruamel.yaml.clib==0.2.15 semver==3.0.4 shellingham==1.5.4 typer==0.24.1 \
    sortedcontainers==2.4.0 soupsieve==2.8.3 tabulate==0.10.0 yt-dlp==2026.3.17 \
    tensorboardX==2.6.5 toml==0.10.2 pytorch-metric-learning==2.9.0    

# 12.A Layer 7: Llama CPP with Cuda
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61" pip install llama_cpp_python==0.3.20 --no-cache-dir

# 13. Conditional OpenCV Build
if [ "$BUILD_OPENCV" = true ]; then
    # Use the absolute path to the venv python to avoid path confusion
    TARGET_VENV_PYTHON="$VENV_PATH/bin/python3"
    VENV_SITE_PACKAGES=$("$TARGET_VENV_PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    
    CLEAN_CV_PATH="$HOME/.local/lib/opencv_cuda/cv2"
    REBUILD_NEEDED=false

    if [ "$FORCE_REBUILD" = true ]; then
        echo "⚠️ Force rebuild flag detected."
        REBUILD_NEEDED=true
    fi

    if [ -d "$CLEAN_CV_PATH" ] && [ "$REBUILD_NEEDED" = false ]; then
        echo "🔍 Found existing build vault. Testing compatibility..."
        ln -sf "$CLEAN_CV_PATH" "$VENV_SITE_PACKAGES/"
        if "$TARGET_VENV_PYTHON" -c "import cv2; import numpy; print('✅ Compatibility Passed')" 2>/dev/null; then
            echo "✅ Vault is healthy. Skipping rebuild."
        else
            echo "❌ Compatibility Failed. Triggering rebuild..."
            REBUILD_NEEDED=true
        fi
    else
        REBUILD_NEEDED=true
    fi

    if [ "$REBUILD_NEEDED" = true ]; then
        if [ -f "./install_cv_cuda.sh" ]; then
            echo "🔨 Building OpenCV with CUDA..."
            chmod +x install_cv_cuda.sh
            source ./install_cv_cuda.sh "$VENV_PATH"
        else
            echo "❌ Error: install_cv_cuda.sh missing."
            exit 1
        fi
    fi

    # Final Link Reinforcement
    if [ -d "$CLEAN_CV_PATH" ]; then
        # 1. Clean existing folder/link to prevent "nested" link errors
        if [ -d "$VENV_SITE_PACKAGES/cv2" ]; then
            sudo rm -rf "$VENV_SITE_PACKAGES/cv2"
        fi
        
        # 2. Create a proper directory instead of a single symlink
        mkdir -p "$VENV_SITE_PACKAGES/cv2"
        
        # 3. Link the entire contents of the Vault (Binary + Configs)
        echo "🔗 Linking full OpenCV Suite (Binary + Configs)..."
        ln -sf "$CLEAN_CV_PATH/"* "$VENV_SITE_PACKAGES/cv2/"
        
        # 4. Optional: Create an __init__.py if the vault doesn't have it
        if [ ! -f "$VENV_SITE_PACKAGES/cv2/__init__.py" ]; then
            echo "from .cv2 import *" > "$VENV_SITE_PACKAGES/cv2/__init__.py"
        fi
    fi
fi

# 14. Installing DNN related binaries and modules
# Check if libsndfile1 is present in the system. Install if missing.
if ! command -v dpkg -s libsndfile1 &> /dev/null; then
    echo "❌ libsndfile1 not found. Installing ..."
    sudo apt update
    sudo apt-get install -y libsndfile1
fi

# Install rest of the DNN stack
pip install --no-deps --no-cache-dir soundfile==0.13.1 librosa==0.10.1 \
    audioread==3.0.1 DeepFilterNet==0.5.6 DeepFilterLib==0.5.6

pip install --no-deps --no-cache-dir lazy_loader==0.4 acoustics==0.2.6 \
    pydub==0.25.1 appdirs==1.4.4 numba==0.59.1

# Set ownership back to the current user for the entire venv tree
echo "🔐 Adjusting permissions for $VENV_PATH..."
sudo chown -R $USER:$USER "$VENV_PATH"

# Clear any build-time path leaks
export PYTHONPATH=""

# 6. Verification Block
echo "------------------------------------------------"
echo "🔍 Running Final Health Check..."

python << EOF
import sys
import torch

try:
    print("Checking DeepFilterNet import...")
    from df.enhance import init_df
    print("✅ DeepFilterNet module: FOUND")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA Available: YES (Device: {torch.cuda.get_device_name(0)})")
    else:
        print("❌ CUDA Available: NO")

    print("⏳ Testing model initialization...")
    model, df_state, _ = init_df()
    print("✅ Model Loading: SUCCESS")

except Exception as e:
    print(f"\n❌ HEALTH CHECK FAILED")
    print(f"Error Type: {type(e).__name__}")
    print(f"Message: {e}")
    # This prints the specific line that failed inside the library
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n🎉 Environment {sys.prefix} is 100% healthy!")
EOF

echo "✅ Env setup complete. To use it, run: source $VENV_PATH/bin/activate . Run python3 cuda_active_check.py to verify."