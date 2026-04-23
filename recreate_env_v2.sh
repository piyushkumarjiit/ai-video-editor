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

# 1. Create the environment and upgrade pip
echo "🐍 Creating Virtual Env..."
python3 -m venv ~/.virtualenvs/ai-video-v2
source ~/.virtualenvs/ai-video-v2/bin/activate
pip install --upgrade pip

# 2. Install Torch and Audio first (The heavy lifting). Unified Installation (Prevents Resolver Collisions)
echo "🔥 Installing Torch components for CUDA 12.6. Installing all components in a single pass..."
pip install --upgrade pip setuptools wheel

# Torch AND the requirements file in one command as this prevents pip from 'fixing' dependencies later and breaking torchaudio
pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    deepfilternet \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
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

# 5. Copy your custom OpenCV-CUDA build (if not in site-packages)
# ln -s /path/to/opencv/build/cv2.so ~/.virtualenvs/ai-video-v2/lib/python3.12/site-packages/
echo "🔗 Linking OpenCV-CUDA build..."
OPENCV_SO=$(find "$HOME" -name "cv2*.so" -type f | head -n 1)
if [ -f "$OPENCV_SO" ]; then
    VENV_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    #ln -sf "$OPENCV_SO" "$VENV_SITE_PACKAGES/cv2.so"
    cp -f "$OPENCV_SO" "$VENV_SITE_PACKAGES/cv2.so"
    echo "✅ Success: OpenCV linked to $VENV_SITE_PACKAGES"
else
    echo "❌ Error: Compiled OpenCV .so file not found in $HOME"
fi


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

echo "✅ Env 2 setup complete. Run cuda_active_check.py to verify."