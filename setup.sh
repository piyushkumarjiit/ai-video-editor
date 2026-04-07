# -------------------------------------------------------------------------
# FILE: setup.sh
# ROLE: Primary Environment & Dependency Orchestrator
#
# DESCRIPTION:
# The main project initialization script. It automates system dependency 
# installation, venv creation, and the GPU-optimized compilation of 
# llama-cpp-python. Includes a validation suite to verify CUDA support.
#
# HARDWARE COMPATIBILITY:
# - Configures GGML_CUDA=on for NVIDIA GPU offloading.
# - Dynamically locates nvcc and sets critical LD_LIBRARY_PATHs.
# -------------------------------------------------------------------------

#!/bin/bash
ENV_PATH="$HOME/.virtualenvs/ai-video-env"
# Set it to True if you want to build opencv that uses CUDA and GPU but it will tkae a long time 30+ mins
BUILD_OPENCV=false

echo "🚀 Starting AI Environment Setup..."

# Add NVIDIA repo and download the keyring for NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 1. System Dependencies
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev nvidia-cuda-toolkit ffmpeg libcudnn8 libcudnn8-dev

# 2. Path Setup (Critical for 1080 Ti detection)
# This dynamically finds where nvcc was installed
export PATH=$(command -v nvcc | xargs dirname):$PATH
export LD_LIBRARY_PATH=$(dirname $(command -v nvcc) | xargs dirname)/lib64:$LD_LIBRARY_PATH

# 3. Venv Creation
mkdir -p "$HOME/.virtualenvs"
if [ ! -d "$ENV_PATH" ]; then
    python3 -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"
pip install --upgrade pip

# 4. GPU-Optimized Compilation
echo "🏎️ Building llama-cpp with 1080 Ti (CUDA) Support..."
# Added FORCE_CMAKE=1 and GGML_CUDA=on for the 1080 Ti
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
pip install llama-cpp-python --no-cache-dir --force-reinstall

# --- 3. Conditional OpenCV Build ---
if [ "$BUILD_OPENCV" = true ]; then
    echo "Flag --build-opencv detected. Calling independent build script..."
    
    # Check if the script exists before calling
    if [ -f "./install_cv_cuda.sh" ]; then
        chmod +x install_cv_cuda.sh        
        source ./install_cv_cuda.sh "$ENV_PATH"

        # if eveyrhing succeeded ink OpenCV_CUDA to venv
        echo "Linking build to active virtual environment..."
        VENV_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        OPENCV_SO=$(find /usr/local/lib/python3* -name "cv2*.so" | head -n 1)

        if [ -f "$OPENCV_SO" ]; then
            ln -sf "$OPENCV_SO" "$VENV_SITE_PACKAGES/cv2.so"
            echo "Success: OpenCV linked to $VENV_SITE_PACKAGES"
        else
            echo "Error: Compiled .so file not found."
        fi
    else
        echo "Error: install_cv_cuda.sh not found in current directory."
    fi
fi

# 5. Install the rest
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --upgrade
fi

# 6. Verify GPU Acceleration
echo "🔍 Verifying GPU Handshake..."
python3 <<EOF
try:
    import llama_cpp
    # Check if the compiled backend supports CUDA
    from llama_cpp import llama_cpp as low_level
    lib = low_level.load_shared_library('llama')
    gpu_supported = bool(lib.llama_supports_gpu_offload())
    
    if gpu_supported:
        print("\n✅ SUCCESS: llama-cpp-python is compiled with GPU support!")
    else:
        print("\n❌ FAILURE: llama-cpp-python is using CPU only.")
except Exception as e:
    print(f"\n⚠️ Error during verification: {e}")
EOF

# 7. Download Sample Data (Optional)
if [ -f "sample_videos.txt" ]; then
    echo "📥 Downloading samples from sample_videos.txt..."
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
wget -O models/yolov8n-face.pt https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt

echo "✅ Setup Complete! Run 'source $ENV_PATH/bin/activate' to start."