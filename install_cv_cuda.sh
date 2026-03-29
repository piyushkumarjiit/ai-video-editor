#!/bin/bash

# --- PRE-FLIGHT CHECKS ---
# Add NVIDIA repo and download the keyring for NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install gcc-12 g++-12 libcudnn8 libcudnn8-dev

# 1. Check for GPU
CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1)
if [ -z "$CUDA_CAP" ]; then
    echo "CRITICAL: No NVIDIA GPU detected via nvidia-smi."
    echo "Since you are on ESXi, ensure PCIe Passthrough is enabled for the 1080 Ti."
    exit 1
fi
echo "Found GPU with Compute Capability: $CUDA_CAP"

# 2. Check for Old OpenCV
if python3 -c "import cv2" &> /dev/null; then
    OLD_VER=$(python3 -c "import cv2; print(cv2.__version__)")
    echo "WARNING: Existing OpenCV ($OLD_VER) detected."
    echo "It is highly recommended to run 'pip uninstall opencv-python' before continuing."
    read -p "Press Enter to continue anyway, or Ctrl+C to stop..."
fi

# --- INSTALLATION START ---

sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config libjpeg-dev libtiff-dev libpng-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
libgtk-3-dev libatlas-base-dev gfortran python3-dev python3-numpy

# Setup directories
mkdir -p ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git --depth 1
git clone https://github.com/opencv/opencv_contrib.git --depth 1
cd opencv && mkdir -p build && cd build

# Configure (Tailored for 1080 Ti / Compute 6.1)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D CMAKE_C_COMPILER=gcc-12 \
      -D CMAKE_CXX_COMPILER=g++-12 \
      -D CUDA_HOST_COMPILER=/usr/bin/gcc-12 \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_ARCH_BIN=$CUDA_CAP \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_opencv_python3=ON \
      -D HAVE_opencv_python3=ON ..

# Build & Install
echo "Building with 8 cores (ESXi limit)..."
make -j8
sudo make install
sudo ldconfig

echo "Installation complete. Verify with: python3 -c 'import cv2; print(cv2.getBuildInformation())'"