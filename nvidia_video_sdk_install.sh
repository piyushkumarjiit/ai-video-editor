# --- Configuration ---
SDK_VERSION="13.0.37"
SDK_ZIP="Video_Codec_SDK_${SDK_VERSION}.zip"
DOWNLOADS_DIR="$HOME/Downloads"
OPENCV_BUILD_DIR="$HOME/opencv_build"
VIDEO_SDK_DIR="$OPENCV_BUILD_DIR/Video_Codec_SDK_${SDK_VERSION}"



# --- Logic ---
# Download the Nvidia SDK
# extract it and keep inside opencv_build
# set LIB DIR PATH
# run below command to create symlinks
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvcuvid.h" /usr/local/cuda/include/nvcuvid.h
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/cuviddec.h" /usr/local/cuda/include/cuviddec.h
#sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvEncodeAPI.h" /usr/local/cuda/include/nvEncodeAPI.h

#--- logic ends ---
echo "🔍 Checking for NVIDIA Video Codec SDK..."

if [ ! -d "$VIDEO_SDK_DIR" ]; then
    if [ -f "$DOWNLOADS_DIR/$SDK_ZIP" ]; then
        echo "📦 Found SDK zip. Extracting to $OPENCV_BUILD_DIR..."
        unzip -q "$DOWNLOADS_DIR/$SDK_ZIP" -d "$OPENCV_BUILD_DIR/"
        
        # Optional: Remove Samples and Binaries to save space (since we only need Interface and Lib)
        echo "🧹 Trimming SDK to essential components..."
        rm -rf "$VIDEO_SDK_DIR/Samples"
        rm -rf "$VIDEO_SDK_DIR/Doc"
        rm -rf "$VIDEO_SDK_DIR/Bin"
    else
        echo "❌ Error: $SDK_ZIP not found in $DOWNLOADS_DIR."
        echo "Please download the SDK from https://developer.nvidia.com/video-codec-sdk and place it in your Downloads folder."
        exit 1
    fi
else
    echo "✅ SDK already extracted at $VIDEO_SDK_DIR"
fi

# --- Symlink Automation (The 'Secret Sauce') ---
echo "🔗 Creating hardware-accelerated symlinks..."
sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvcuvid.h" /usr/local/cuda/include/nvcuvid.h
sudo ln -sf "$VIDEO_SDK_DIR/Interface/cuviddec.h" /usr/local/cuda/include/cuviddec.h
sudo ln -sf "$VIDEO_SDK_DIR/Interface/nvEncodeAPI.h" /usr/local/cuda/include/nvEncodeAPI.h

# Verify the links
if [ -L "/usr/local/cuda/include/nvcuvid.h" ]; then
    echo "✨ Symlinks verified and active."
else
    echo "⚠️ Warning: Symlink creation failed. You may need to run this part as sudo."
fi