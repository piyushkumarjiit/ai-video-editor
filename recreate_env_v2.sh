# -------------------------------------------------------------------------
# FILE: recreate_env_v2.sh
# ROLE: Rapid Environment Recovery & Deployment
#
# DESCRIPTION:
# A streamlined script for rebuilding the Python environment. It 
# enforces a hardware-optimized installation order to ensure Torch 
# components are correctly linked to CUDA 12.6.
#
# HARDWARE COMPATIBILITY:
# - Forces Torch installation via the cu126 index for 1080 Ti support.
# - Configures Ultralytics with --no-deps to prevent driver issues.
# -------------------------------------------------------------------------

#!/bin/bash

# Force correct library loading for the current session & health check
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MKL_SERVICE_FORCE_INTEL=1

unset PYTHONPATH

# Define the environment path
VENV_PATH="$HOME/.virtualenvs/ai-video-denoise"

BUILD_OPENCV=true

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

echo "🗑️ Cleaning up old environment if it exists..."
sudo rm -rf "$VENV_PATH"

# 1. Create the environment and upgrade pip
echo "🐍 Creating Virtual Env..."
python3 -m venv "$VENV_PATH"
echo "🔌 Activating environment..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip

# 2. Install Torch and Audio first (The heavy lifting). Unified Installation (Prevents Resolver Collisions)
echo "🔥 Installing Torch components for CUDA 12.1. Installing all components in a single pass..."
pip install --upgrade pip setuptools "wheel<0.45.0"

# Torch AND the requirements file in one command as this prevents pip from 'fixing' dependencies later and breaking torchaudio
pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    deepfilternet \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

# 3. Install the rest of the requirements
# if [ -f "requirements-denoise.txt" ]; then
#     echo "📦 Installing requirements-denoise.txt..."
#     pip install -r requirements-denoise.txt
# else
#     echo "⚠️ Warning: requirements-denoise.txt not found, skipping."
# fi

# 4. The Critical "Manual" Step for Ultralytics (if needed in this new environment)
#pip install ultralytics --no-deps --no-cache-dir

# 5. Conditional OpenCV Build
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
        if [ -d "$VENV_SITE_PACKAGES/cv2" ] && [ ! -L "$VENV_SITE_PACKAGES/cv2" ]; then
             sudo rm -rf "$VENV_SITE_PACKAGES/cv2"
        fi
        ln -sf "$CLEAN_CV_PATH" "$VENV_SITE_PACKAGES/"
    fi
fi

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

print("\n🎉 Environment 'ai-video-v2' is 100% healthy!")
EOF

echo "✅ Env 2 setup complete. To use it, run: source $VENV_PATH/bin/activate . Run cuda_active_check.py to verify."