#!/bin/bash

# Define the environment path
VENV_PATH="$HOME/.virtualenvs/ai-video-diarize"
BUILD_OPENCV=true
FORCE_REBUILD=false

echo "🗑️ Cleaning up old environment if it exists..."
rm -rf "$VENV_PATH"

echo "📂 Creating new Diarization VENV..."
python3 -m venv "$VENV_PATH"

echo "🔌 Activating environment..."
source "$VENV_PATH/bin/activate"

echo "📦 Upgrading pip and setting up build tools..."
pip install --upgrade pip setuptools "wheel<0.45.0"

echo "🧪 Installing PyTorch and Diarization Stack for CUDA 12.6..."

# Uninstall standard onnxruntime just in case a dependency pulled it in
pip uninstall -y onnxruntime

# We use the explicit cu126 index to ensure your 1080 Ti is utilized
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    whisperx pyannote.audio onnxruntime-gpu speechbrain \
    transformers tokenizers numpy==1.26.4 python-dotenv \
    huggingface_hub matplotlib tqdm pandas scipy librosa \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
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

    # Final Link Reinforcement (Ensuring the symlink exists in this specific venv)
    if [ -d "$CLEAN_CV_PATH" ]; then
        ln -sf "$CLEAN_CV_PATH" "$VENV_SITE_PACKAGES/"
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