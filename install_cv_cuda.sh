# -------------------------------------------------------------------------
# FILE: install_cv_cuda.sh
# ROLE: Hardware-Specific OpenCV & PyTorch Builder
#
# DESCRIPTION:
# Compiles OpenCV from source with CUDA/cuDNN support specifically 
# tailored for the 1080 Ti (Pascal). It also synchronizes PyTorch 
# installations using the cu126 index for legacy hardware support.
#
# HARDWARE COMPATIBILITY:
# - Targets Compute Capability 6.1 (NVIDIA Pascal / 1080 Ti).
# - Requires gcc-12/g++-12 for OpenCV-CUDA source compilation.
# -------------------------------------------------------------------------

#!/bin/bash

# --- PRE-FLIGHT CHECKS ---

if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo. Please enter your password:"
    sudo -v || exit 1
fi
# there is also dependecy on Numpy 1.X so ensure that your venv uses 1.X

# Add NVIDIA repo and download the keyring for NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
KEYRING_FILE="cuda-keyring_1.1-1_all.deb"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$KEYRING_FILE"

echo "🌐 Downloading CUDA keyring..."
# -q for quiet, -O to ensure it saves with the name we expect
wget -q "$KEYRING_URL" -O "$KEYRING_FILE"
echo "📦 Installing keyring..."
sudo dpkg -i "$KEYRING_FILE"
sudo apt update
# Delete the file immediately after installation
echo "🧹 Cleaning up $KEYRING_FILE..."
rm -f "$KEYRING_FILE"
sudo apt install gcc-12 g++-12 ninja-build -y # needed for opencv-cuda compilation as later versions are not supported yet
sudo apt install libcudnn8 libcudnn8-dev

# Download NVIDIA Video Decoder SDK and set below variable to the unzipped folder path, preferably inside opencv_build
VIDEO_SDK_DIR="$HOME/opencv_build/Video_Codec_SDK_13.0.37"

# 1. Get Compute Capability (e.g., 6.1 or 8.9)
CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1)
echo "🚀 Detected GPU Compute Capability: $CUDA_CAP"
if [ -z "$CUDA_CAP" ]; then
    echo "CRITICAL: No NVIDIA GPU detected via nvidia-smi."
    echo "If you are on Hypervisor, ensure PCIe Passthrough is enabled for the 1080 Ti."
    exit 1
fi
echo "Found GPU with Compute Capability: $CUDA_CAP"

# Check if a virtual environment is active and determine Target Location
# Priority 1: Argument passed from parent script ($1)
# Priority 2: Currently active VIRTUAL_ENV variable
# Priority 3: System-wide fallback
if [ -n "$1" ]; then
    ENV_PATH="$1"
    TARGET_PYTHON="$ENV_PATH/bin/python3"
    echo "🎯 Using provided argument path: $ENV_PATH"
    # Check what version this path actually reports
    DETECTED_VER=$($TARGET_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    
    if [[ "$DETECTED_VER" != "3.10" && "$DETECTED_VER" != "3.11" ]]; then
        echo "❌ Version Mismatch! Found Python $DETECTED_VER, but this script supports 3.10 or 3.11."
        echo "   Please use: python3.11 -m venv $ENV_PATH"
        exit 1
    fi
    echo "🎯 Validated Python $DETECTED_VER at: $ENV_PATH"
elif [ -n "$VIRTUAL_ENV" ]; then
    ENV_PATH="$VIRTUAL_ENV"
    TARGET_PYTHON="$VIRTUAL_ENV/bin/python3"
    echo "✅ Active venv detected: $VIRTUAL_ENV"
else
    echo "⚠️  No venv detected. Defaulting to system-wide installation."
    ENV_PATH="/usr/local"
    TARGET_PYTHON=$(which python3)
fi

# Final Check: Ensure the Python binary exists
if [ ! -f "$TARGET_PYTHON" ]; then
    echo "❌ Error: Python executable not found at $TARGET_PYTHON"
    exit 1
fi

# 2. Check for Old OpenCV
if "$TARGET_PYTHON" -c "import cv2" &> /dev/null; then
    OLD_VER=$("$TARGET_PYTHON" -c "import cv2; print(cv2.__version__)")
    echo "WARNING: Existing OpenCV ($OLD_VER) detected."
    echo "It is highly recommended to run 'pip uninstall opencv-python' before continuing."
    read -p "Press Enter to continue anyway, or Ctrl+C to stop..."
fi

# --- INSTALLATION START ---
sudo apt-get install -y build-essential cmake git pkg-config libjpeg-dev libtiff-dev libpng-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
libgtk-3-dev libatlas-base-dev gfortran python3-dev python3-numpy

# Dynamically pull the paths required by CMake
PYTHON_INCLUDE_DIR=$( "$TARGET_PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))" )
PYTHON_LIBRARY=$("$TARGET_PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_PACKAGES_PATH=$( "$TARGET_PYTHON" -c "import sysconfig; print(sysconfig.get_path('platlib'))" )
NUMPY_INCLUDE_DIR=$( "$TARGET_PYTHON" -c "import numpy; print(numpy.get_include())" )

# Setup directories
mkdir -p ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git --depth 1
git clone https://github.com/opencv/opencv_contrib.git --depth 1
cd opencv && mkdir -p build && cd build

# Navigate to your build directory (where you keep your source code)
cd ~/opencv_build

# Clone the NVIDIA Video Codec headers
# git clone https://github.com/FFmpeg/nv-codec-headers.git
# cd nv-codec-headers
# make
# sudo make install
# sudo ln -sf /usr/local/include/ffnvcodec/*.h /usr/local/cuda/include/
# sudo ln -sf /usr/local/lib/pkgconfig/ffnvcodec.pc /usr/lib/x86_64-linux-gnu/pkgconfig/
# sudo ln -sf /usr/local/include/ffnvcodec/*.h /usr/local/cuda/include/

# Download the Nvidia SDK
# extract it and keep inside opencv_build
# set LIB DIR PATH
# run below command to create symlinks
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvcuvid.h" /usr/local/cuda/include/nvcuvid.h
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/cuviddec.h" /usr/local/cuda/include/cuviddec.h
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvEncodeAPI.h" /usr/local/cuda/include/nvEncodeAPI.h

cd ~/opencv_build/opencv/build

# Purge previous build cache
echo "🧹 Purging stale CMake and loader cache to prevent version ghosting..."
rm -rf python_loader
rm -f CMakeCache.txt


# Configure (Tailored for 1080 Ti / Compute 6.1)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX="$ENV_PATH" \
      -D CMAKE_C_COMPILER=gcc-12 \
      -D CMAKE_CXX_COMPILER=g++-12 \
      -D WITH_CUDA=ON \
      -D CUDA_HOST_COMPILER=/usr/bin/gcc-12 \
      -D CUDA_ARCH_BIN=$CUDA_CAP \
      -D CUDA_ARCH_PTX=$CUDA_CAP \
      -D OPENCV_CUDA_FORCE_PTX_EDITION=$CUDA_CAP \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_opencv_python3=ON \
      -D HAVE_opencv_python3=ON \
      -D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_build/opencv_contrib/modules \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D ENABLE_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D VIDEOIO_INCLUDE_DIRS="$VIDEO_SDK_DIR/Interface" \
      -D VIDEOIO_LIBRARIES="$VIDEO_SDK_DIR/Lib/linux/stubs/x86_64" \
      -D WITH_NVCUVID=ON \
      -D NVCUVID_INCLUDE_DIR="$VIDEO_SDK_DIR/Interface" \
      -D NVCUVID_LIBRARY="$VIDEO_SDK_DIR/Lib/linux/stubs/x86_64/libnvcuvid.so" \
      -D WITH_NVCUVENC=ON \
      -D NVCUVENC_INCLUDE_DIR="$VIDEO_SDK_DIR/Interface" \
      -D NVCUVENC_LIBRARY="$VIDEO_SDK_DIR/Lib/linux/stubs/x86_64/libnvidia-encode.so" \
      -D PYTHON3_EXECUTABLE="$TARGET_PYTHON" \
      -D PYTHON3_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
      -D PYTHON3_LIBRARY=$PYTHON_LIBRARY \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$NUMPY_INCLUDE_DIR \
      -D PYTHON3_PACKAGES_PATH=$PYTHON_PACKAGES_PATH \
      -G Ninja ..


# Build & Install
CPU_CORES=$(nproc)
echo "Building with $CPU_CORES cores (Dynamic detection)..."
#make -j"$CPU_CORES"
ninja

#ninja install
#sudo ldconfig

# No longer doing torch things as that should be handled by the main script
# Determine the correct Torch Index URL
# # Pascal (6.x) and older needs the cu126 or cu124 branch for CC 6.1 support
# if (( $(echo "$CUDA_CAP < 7.0" | bc -l) )); then
#     echo "⚠️ Legacy GPU detected (Pascal/Maxwell). Forcing compatibility build..."
#     INDEX_URL="https://download.pytorch.org/whl/cu126"
# else
#     echo "✅ Modern GPU detected. Using standard high-performance build..."
#     INDEX_URL="https://download.pytorch.org/whl/cu130"
# fi

# # Execute the Clean Reinstall
# pip uninstall -y torch torchvision torchaudio
# pip install torch==2.4.0 \
#     torchvision==0.19.0 \
#     torchaudio==2.4.0 \
#     --extra-index-url https://download.pytorch.org/whl/cu121 \
#     --no-cache-dir

# echo "✨ PyTorch installation synchronized with hardware capability."

# --- 1. Define Permanent Destination ---
CLEAN_CV_PATH="$HOME/.local/lib/opencv_cuda"
mkdir -p "$CLEAN_CV_PATH"

# --- 2. Stage the Move ---
echo "🚚 Staging build for move..."
TEMP_VAULT="$HOME/.local/lib/opencv_cuda_temp"
rm -rf "$TEMP_VAULT" && mkdir -p "$TEMP_VAULT/cv2"

# A: Copy wrappers
if [ -d "$(pwd)/python_loader/cv2" ]; then
    cp -r "$(pwd)/python_loader/cv2/"* "$TEMP_VAULT/cv2/"
fi

# B: Find and Copy actual binary
REAL_SO=$(find "$(pwd)/lib" -name "cv2*.so" | head -n 1)

if [ -n "$REAL_SO" ] && [ -f "$REAL_SO" ]; then
    cp "$REAL_SO" "$TEMP_VAULT/cv2/"
    echo "💎 Verified Binary: $(basename "$REAL_SO")"
    
    # --- 3. THE SAFETY GATE ---
    echo "🛡️ Final Integrity Check..."
    if [ -f "$TEMP_VAULT/cv2/$(basename "$REAL_SO")" ]; then
        echo "✅ Integrity Verified. Finalizing Vault..."
        rm -rf "$CLEAN_CV_PATH/cv2"
        mv "$TEMP_VAULT/cv2" "$CLEAN_CV_PATH/"
        rm -rf "$TEMP_VAULT"
        
        echo "🧹 Cleaning up build debris (Safe to delete now)..."
        cd ~
        # rm -rf "$HOME/opencv_build"  # Uncomment this once you trust the script 100%
    else
        echo "❌ ERROR: Binary transfer failed integrity check!"
        echo "⚠️  ABORTING CLEANUP. Your build is still safe at: $(pwd)"
        exit 1
    fi
else
    echo "❌ ERROR: Could not find compiled cv2*.so in $(pwd)/lib"
    echo "⚠️  ABORTING CLEANUP. Check the build directory manually."
    exit 1
fi

# --- 4. Final Linking ---
VENV_SITE_PACKAGES=$("$TARGET_PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
sudo rm -rf "$VENV_SITE_PACKAGES/cv2"
ln -sf "$CLEAN_CV_PATH" "$VENV_SITE_PACKAGES/"
sudo ldconfig

echo "✨ Move Process Complete."

"$TARGET_PYTHON" -c 'import cv2; print(f"OpenCV Version: {cv2.__version__}"); print(f"CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")'
echo "Installation complete. Verify with: python3 -c 'import cv2; print(cv2.getBuildInformation())'"