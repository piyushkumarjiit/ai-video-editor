#!/bin/bash
ENV_PATH="$HOME/.virtualenvs/ai-video-env"

echo "🚀 Starting R720 AI Environment Setup..."

# 1. System Dependencies
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev nvidia-cuda-toolkit

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

echo "✅ Setup Complete! Run 'source $ENV_PATH/bin/activate' to start."