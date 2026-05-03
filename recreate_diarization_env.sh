#!/bin/bash

# Define the target version and path
TARGET_PYTHON="python3.10"
VENV_PATH="$HOME/.virtualenvs/ai-video-diarize"

BUILD_OPENCV=true
FORCE_REBUILD=true

echo "🗑️ Cleaning up old environment if it exists..."
rm -rf "$VENV_PATH"

echo "🛠️ Installing system-level FFmpeg headers..."
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev

# 1. Create the environment and upgrade pip
echo "🐍 Creating Virtual Env..."
# 1. Check if the specific Python version is even installed
if ! command -v $TARGET_PYTHON &> /dev/null; then
    echo "❌ $TARGET_PYTHON not found. Installing via deadsnakes..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install $TARGET_PYTHON $TARGET_PYTHON-venv $TARGET_PYTHON-dev -y
fi

# 2. Create the venv using that exact version
echo "Creating venv with $($TARGET_PYTHON --version)..."
$TARGET_PYTHON -m venv $VENV_PATH

# 3. Activate and verify
echo "🔌 Activating environment..."
source "$VENV_PATH/bin/activate"

echo "📦 Upgrading pip and setting up build tools..."
"$TARGET_PYTHON" -m pip install --upgrade pip setuptools<70.0.0 wheel<0.45.0

echo "🧪 Installing PyTorch and Diarization Stack for CUDA 12.6..."

# Install the 'Bridge' dependencies to lock the versions
"$TARGET_PYTHON" -m pip install numpy==1.26.4 transformers==4.37.2 tokenizers==0.15.2

echo "🚀 Installing authentic WhisperX from GitHub..."

# 2. Install from the official source (v3.1.1 is highly stable for 1080 Ti)
"$TARGET_PYTHON" -m pip install git+https://github.com/m-bain/whisperX.git@v3.1.1

# We use the explicit cu126 index to ensure your 1080 Ti is utilized
"$TARGET_PYTHON" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    python-dotenv==1.0.1 faster-whisper==1.0.1 ctranslate2==4.3.1 \
    pyannote.audio==3.1.1 pyannote.core==5.0.0 speechbrain==1.0.0 \
    huggingface_hub==0.24.7 matplotlib tqdm==4.66.4 pandas==2.2.2 \
    scipy==1.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-cache-dir

# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# echo "🧬 Installing WhisperX and Pyannote Stack..."
# # Pinned versions to ensure 'use_auth_token' remains functional
# pip install whisperx==3.7.2 pyannote.audio==3.3.2 speechbrain==1.1.0

# echo "🛡️ Fixing 'Dependency Hell' with Legacy Pins..."
# # These specific versions prevent the 'unexpected keyword argument' crash
# pip install "huggingface_hub<0.25.0" "transformers<=4.48.0" "tokenizers<0.20.0"

# echo "🔢 Finalizing Foundation..."
# # Pinning Numpy to 1.26.4 to ensure compatibility with WhisperX/Pyannote C-extensions
# pip install numpy
# #pip install pandas scipy


# 5. Conditional OpenCV Build
# Flags: $BUILD_OPENCV (true/false), $FORCE_REBUILD (true/false)
if [ "$BUILD_OPENCV" = true ]; then
    # Use the specific venv python to get the correct site-packages path
    TARGET_VENV_PYTHON="$VENV_PATH/bin/python3"
    VENV_SITE_PACKAGES=$("$TARGET_VENV_PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    
    CLEAN_CV_PATH="$HOME/.local/lib/opencv_cuda/cv2"
    REBUILD_NEEDED=false

    if [ "$FORCE_REBUILD" = true ]; then
        echo "⚠️ Force rebuild flag detected."
        REBUILD_NEEDED=true
    fi

    # Check the Permanent Vault instead of the build folder
    if [ -d "$CLEAN_CV_PATH" ] && [ "$REBUILD_NEEDED" = false ]; then
        echo "🔍 Found existing build vault. Testing compatibility..."
        ln -sf "$CLEAN_CV_PATH" "$VENV_SITE_PACKAGES/"
        
        if "$TARGET_VENV_PYTHON" -c "import cv2; import numpy; print('✅ Compatibility Passed')" 2>/dev/null; then
            echo "✅ Vault is healthy. Skipping rebuild."
        else
            echo "❌ Compatibility Failed (likely NumPy mismatch). Triggering rebuild..."
            REBUILD_NEEDED=true
        fi
    else
        REBUILD_NEEDED=true
    fi

    if [ "$REBUILD_NEEDED" = true ]; then
        if [ -f "./install_cv_cuda.sh" ]; then
            echo "🔨 Building OpenCV with CUDA (This will update the Vault)..."
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

echo "------------------------------------------------"
echo "🔍 Running Final Health Check..."
"$VENV_PATH/bin/python3" << EOF
import torch
import cv2
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ OpenCV CUDA Enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
EOF

echo "✅ Diarization Environment is ready!"
echo "To use it, run: source $VENV_PATH/bin/activate"