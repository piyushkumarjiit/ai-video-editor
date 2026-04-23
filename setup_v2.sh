#!/bin/bash
# -------------------------------------------------------------------------
# ROLE: Primary Environment & Dependency Orchestrator (Idempotent Version)
# -------------------------------------------------------------------------

ENV_PATH="$HOME/.virtualenvs/ai-video-env"
BUILD_OPENCV=true
# --- OLAH CONFIGURATION VARIABLES ---
# --- CONFIGURATION VARIABLES ---
NFS_SERVER="192.168.1.50"
NFS_SHARE="/mnt/pool0/ai_data"
MOUNT_POINT="/mnt/ai-models"  # Consistent with your Ollama path

# 1. Automated NFS Mounting
sudo mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
    echo "🔗 Mounting NFS share..."
    sudo mount -t nfs "$NFS_SERVER:$NFS_SHARE" "$MOUNT_POINT"
    
    if ! grep -q "$MOUNT_POINT" /etc/fstab; then
        echo "$NFS_SERVER:$NFS_SHARE $MOUNT_POINT nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
    fi
fi

# 2. Safety Check
if ! mountpoint -q "$MOUNT_POINT"; then
    echo "❌ ERROR: External storage failed to mount at $MOUNT_POINT"
    exit 1
fi

# 3. Ollama Configuration
# Set the Environment variable only if it doesn't exist
if ! grep -q "OLLAMA_MODELS=" /etc/environment; then
    echo "OLLAMA_MODELS=\"$MOUNT_POINT\"" | sudo tee -a /etc/environment
fi

# Update the Systemd unit
SERVICE_FILE="/etc/systemd/system/ollama.service"
if [ -f "$SERVICE_FILE" ]; then
    if ! grep -q "OLLAMA_MODELS=$MOUNT_POINT" "$SERVICE_FILE"; then
        echo "⚙️ Injecting OLLAMA_MODELS path into Systemd..."
        sudo sed -i "/\[Service\]/a Environment=\"OLLAMA_MODELS=$MOUNT_POINT\"" "$SERVICE_FILE"
        sudo systemctl daemon-reload
        sudo systemctl restart ollama
    fi
else
    echo "⚠️ Ollama service file not found at $SERVICE_FILE. Skipping service restart."
fi

echo "🚀 Starting AI Environment Setup..."

# 0. Permission Guard (Fixes the 'share/man' issue before it happens)


# 1. System Dependencies (Non-destructive)
# We check for nvcc to avoid re-adding repos or triggering driver mismatches
if ! command -v nvcc &> /dev/null; then
    echo "📦 Installing CUDA Toolkit (Development headers only)..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    # --no-install-recommends is critical to avoid overwriting existing drivers
    sudo apt install -y --no-install-recommends \
        build-essential cmake python3-venv python3-dev \
        nvidia-cuda-toolkit ffmpeg libcudnn8 libcudnn8-dev
else
    echo "✅ CUDA Toolkit already present."
fi

# 2. Path Setup (Dynamic)
export PATH=$(dirname $(command -v nvcc)):$PATH
export LD_LIBRARY_PATH=$(dirname $(dirname $(command -v nvcc)))/lib64:$LD_LIBRARY_PATH

# 3. Venv Creation
sudo mkdir -p "$HOME/.virtualenvs"
# This finds the actual person logged in, even if the script is running as root
REAL_USER=${SUDO_USER:-$USER}
sudo chown -R $REAL_USER:$REAL_USER "$HOME/.virtualenvs"
if [ ! -d "$ENV_PATH" ]; then
    echo "🛠️ Creating Virtual Environment..."
    python3 -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

# 4. GPU-Optimized Compilation (Llama-CPP)
# We check if it's already installed with CUDA to save time
if ! python3 -c "import llama_cpp" &> /dev/null; then
    echo "🏎️ Building llama-cpp with CUDA Support..."
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install llama-cpp-python --no-cache-dir
else
    echo "✅ llama-cpp-python already installed."
fi

# 5. Conditional OpenCV Build
if [ "$BUILD_OPENCV" = true ]; then
    echo "Flag --build-opencv detected."
    # Search for our custom build first
    OPENCV_SO=$(find "$HOME/opencv_build" -name "cv2*.so" -type f 2>/dev/null | head -n 1)    
    if [ -n "$OPENCV_SO" ]; then
        echo "✅ Existing OpenCV build found at $OPENCV_SO. Skipping build."
        VENV_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        ln -sf "$OPENCV_SO" "$VENV_SITE_PACKAGES/cv2.so"
        echo "✅ Success: OpenCV linked to $VENV_SITE_PACKAGES"
    else
        echo "Error: Compiled .so file not found."
    fi
elif [ -f "./install_cv_cuda.sh" ]; then
        echo "🔨 Building OpenCV with CUDA (This will take 30+ mins)..."
        chmod +x install_cv_cuda.sh
        # Pass the number of cores to the sub-script to speed it up
        export MAKEFLAGS="-j$(nproc)"
        source ./install_cv_cuda.sh "$ENV_PATH"
        OPENCV_SO=$(find "$HOME/opencv_build" -name "cv2*.so" -type f | head -n 1)
else
        echo "❌ Error: install_cv_cuda.sh missing and no previous build found."
fi

# 6. Install Python Requirements
if [ -f "requirements.txt" ]; then
    echo "📚 Installing project dependencies..."
    # We use --upgrade-strategy only-if-needed to protect our CUDA-compiled packages
    pip install -r requirements.txt --upgrade-strategy only-if-needed
fi

# 7. Final Verification Suite
echo "🔍 Verifying GPU Handshake..."
python3 <<EOF
import sys, torch, cv2
from llama_cpp import Llama
try:
    # Check Torch
    t_cuda = torch.cuda.is_available()
    print(f"--- Torch CUDA: {'✅' if t_cuda else '❌'}")
    
    # Check OpenCV
    cv_cuda = "NVIDIA CUDA:                   YES" in cv2.getBuildInformation()
    print(f"--- OpenCV CUDA: {'✅' if cv_cuda else '❌'}")

    # Check Llama (API Friendly check)
    Llama(model_path="", n_gpu_layers=-1, verbose=False)
    print("--- Llama-CPP CUDA: ✅")
except Exception as e:
    if "No such file" in str(e): print("--- Llama-CPP CUDA: ✅")
    else: print(f"--- Llama-CPP CUDA: ❌ ({e})")
EOF

# 7. Download Sample Data (Optional)
if [ -f "sample_videos.txt" ]; then
    echo "ðŸ“¥ Downloading samples from sample_videos.txt..."
    # Download as MP4
    yt-dlp -f "bestvideo[vcodec^=avc1][height<=720]+bestaudio[ext=m4a]/best[vcodec^=avc1][height<=720]/best" \
    -a sample_videos.txt \
    -P "./samples" \
    -o "%(title)s.%(ext)s" \
    --no-mtime \
    --restrict-filenames \
    --merge-output-format mp4
fi

# Create models direcotry
mkdir -p models
# Download into models folder
if [ ! -f "models/yolov8n-face.pt" ]; then
    wget -O models/yolov8n-face.pt https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt
fi

echo "âœ… Setup Complete! Run 'source $ENV_PATH/bin/activate' to start."