#!/bin/bash
# -------------------------------------------------------------------------
# FILE: recreate_env_split.sh
# ROLE: Multi-Role AI Environment Builder (Denoise, ASR, Diarize)
# USAGE: ./recreate_env_split.sh [venv_name] [python_version] [role]
# ROLES: denoise | asr | diarize
# DESCRIPTION:
# A streamlined script for rebuilding the Python environment. It 
# enforces a hardware-optimized installation order to ensure Torch 
# components are correctly linked to CUDA 11.8 for Pascal (1080 Ti) support.
#
# HARDWARE COMPATIBILITY:
# - Forces Torch installation via the cu118 index for 1080 Ti stability.
# - Optimized for NVIDIA Driver 535+ (CUDA 12.2).

# 1. Setup the Denoising Environment
#chmod +x recreate_env_split.sh
#./recreate_env_split.sh ai-video-denoise python3.11 denoise

# 2. Setup the Transcription (ASR) Environment
#./recreate_env_split.sh ai-video-asr python3.11 asr

# 3. Setup the Speaker Identification (Diarize) Environment
#./recreate_env_split.sh ai-video-diarize python3.11 diarize
# -------------------------------------------------------------------------

VENV_NAME=$1
TARGET_PYTHON=$2
ROLE=$3
FORCE_REBUILD=${4:-false} # Optional 4th argument

if [[ -z "$VENV_NAME" || -z "$TARGET_PYTHON" || -z "$ROLE" ]]; then
    echo "❌ Usage: $0 [venv_name] [python_version] [denoise|asr|diarize] (force_rebuild)"
    exit 1
fi

VENV_PATH="$HOME/.virtualenvs/$VENV_NAME"
BUILD_OPENCV=true

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MKL_SERVICE_FORCE_INTEL=1
unset PYTHONPATH

echo "🗑️ Cleaning up $VENV_PATH..."
sudo rm -rf "$VENV_PATH"

# 1. Prerequisites (Rust for Denoise)
if [ "$ROLE" == "denoise" ]; then
    if ! command -v cargo &> /dev/null; then
        echo "🦀 Installing Rust for DeepFilterNet..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
fi

# 2. Install Target Python via Deadsnakes
if ! command -v $TARGET_PYTHON &> /dev/null; then
    echo "❌ $TARGET_PYTHON not found. Installing via PPA..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install $TARGET_PYTHON $TARGET_PYTHON-venv $TARGET_PYTHON-dev -y
fi

# 3. Create Venv
echo "🐍 Creating venv at $VENV_PATH..."
$TARGET_PYTHON -m venv $VENV_PATH
source "$VENV_PATH/bin/activate"
# Quote the version constraints to prevent shell redirection errors
$TARGET_PYTHON -m pip install --upgrade pip "setuptools<70.0.0" "wheel<0.45.0"

# 4. Hardware-Optimized Installation
echo "🔥 Installing $ROLE-specific stack for 1080 Ti..."

# Common base for all (NumPy 1.26.4 is the bridge)
#BASE_LIBS="torch==2.1.2 torchaudio==2.1.2 numpy==1.26.4 pandas<2.2.0 scipy<1.13.0 --extra-index-url https://download.pytorch.org/whl/cu118"

if [ "$ROLE" == "denoise" ]; then
    # Denoise Stack: DeepFilterNet 0.5.6
    # $TARGET_PYTHON -m pip install torch==2.1.2 torchaudio==2.1.2 numpy==1.26.4 pandas==2.1.4 scipy==1.11.4 deepfilternet==0.5.6 --extra-index-url https://download.pytorch.org/whl/cu124
    $TARGET_PYTHON -m pip install deepfilternet==0.5.6 numpy==1.26.4
    $TARGET_PYTHON -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    
elif [ "$ROLE" == "asr" ]; then
    # ASR Stack: WhisperX (Manual layering to protect pins)
    #$TARGET_PYTHON -m pip install torch==2.1.2 torchaudio==2.1.2 numpy==1.26.4 ctranslate2==4.3.1 --extra-index-url https://download.pytorch.org/whl/cu118
    #$TARGET_PYTHON -m pip install git+https://github.com/m-bain/whisperX.git@v3.1.1 --no-deps
    #$TARGET_PYTHON -m pip install torch==2.1.2 torchaudio==2.1.2 numpy==1.26.4 ctranslate2==4.3.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
    #$TARGET_PYTHON -m pip install faster-whisper==1.0.3 transformers==4.37.2 nltk setfst srt pandas==2.1.4 scipy==1.11.4
    # Claude
    #$TARGET_PYTHON -m pip install whisperx --no-deps
    #$TARGET_PYTHON -m pip install pyannote.audio==3.3.2 --no-deps
    #$TARGET_PYTHON -m pip install pyannote.core pyannote.pipeline pyannote.metrics speechbrain asteroid-filterbanks omegaconf einops
    
    # --- PHASE 1: CORE FOUNDATION (Hardware & Version Bridge) ---
    $TARGET_PYTHON -m pip install "setuptools==69.5.1" "wheel==0.43.0" "packaging==24.0" "fsspec[http]==2024.3.1" "protobuf==3.20.3" --no-deps
    $TARGET_PYTHON -m pip install torch==2.5.1 torchaudio==2.5.1 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cu124 --no-deps

    # --- PHASE 2: THE PLUMBING ---
    # 0.A Torch & NVIDIA Stack (Added mpmath here so sympy works immediately)
    $TARGET_PYTHON -m pip install "mpmath==1.3.0" "sympy==1.13.1" "jinja2==3.1.3" "networkx==3.2.1" "triton==3.1.0" --no-deps
    $TARGET_PYTHON -m pip install \
        "nvidia-cublas-cu12==12.4.5.8" "nvidia-cuda-cupti-cu12==12.4.127" \
        "nvidia-cuda-nvrtc-cu12==12.4.127" "nvidia-cuda-runtime-cu12==12.4.127" \
        "nvidia-cudnn-cu12==9.1.0.70" "nvidia-cufft-cu12==11.2.1.3" \
        "nvidia-curand-cu12==10.3.5.147" "nvidia-cusolver-cu12==11.6.1.9" \
        "nvidia-cusparse-cu12==12.3.1.170" "nvidia-nccl-cu12==2.21.5" \
        "nvidia-nvjitlink-cu12==12.4.127" "nvidia-nvtx-cu12==12.4.127" --no-deps

    # 0.B-G Audio, Math, & Data Helpers (Added six here)
    $TARGET_PYTHON -m pip install "pyannote.database==5.0.1" "rich==13.7.1" "soundfile==0.12.1" "docopt==0.6.2" "optuna==3.6.1" \
        "scikit-learn==1.4.2" "sortedcontainers==2.4.0" "julius==0.2.7" "librosa==0.10.1" "torch-pitch-shift==1.2.4" --no-deps
    $TARGET_PYTHON -m pip install "hyperpyyaml==1.2.2" "sentencepiece==0.2.0" "lightning-utilities==0.11.2" \
        "pytorch-lightning==2.1.4" "antlr4-python3-runtime==4.9.3" --no-deps
    $TARGET_PYTHON -m pip install "annotated-types==0.6.0" "pydantic-core==2.18.2" "typing-inspection==0.4.2" "httpcore==1.0.5" --no-deps
    $TARGET_PYTHON -m pip install "audioread==3.0.1" "decorator==5.1.1" "lazy_loader==0.4" "msgpack==1.0.8" "pooch==1.8.1" "soxr==0.3.7" "numba==0.59.1" --no-deps
    $TARGET_PYTHON -m pip install "threadpoolctl==3.4.0" "alembic==1.13.1" "colorlog==6.8.2" "sqlalchemy==2.0.29" "ruamel.yaml==0.18.6" "primePy==1.3" --no-deps
    $TARGET_PYTHON -m pip install "anyio==4.3.0" "distro==1.9.0" "httpx==0.27.0" "pydantic==2.7.1" "sniffio==1.3.1" --no-deps
    $TARGET_PYTHON -m pip install "greenlet==3.0.3" "Mako==1.3.2" "typer==0.12.3" "shellingham==1.5.4" "h11==0.14.0" "matplotlib==3.8.4" "tabulate==0.9.0" --no-deps
    $TARGET_PYTHON -m pip install "pytz==2024.1" "python-dateutil==2.9.0.post0" "six==1.16.0" --no-deps

    # --- PHASE 3: THE AI MODELS ---
    # 1. Install WhisperX first (it's the most likely to cause drift)
    $TARGET_PYTHON -m pip install git+https://github.com/m-bain/whisperX.git@v3.3.1 --no-build-isolation --no-deps
    
    # 2. IMMEDIATE RE-ASSERTION (Fixes the drift WhisperX just caused)
    $TARGET_PYTHON -m pip install --force-reinstall --no-deps "faster-whisper==1.1.0" "transformers==4.40.2" "ctranslate2==4.4.0" \
        "nltk==3.8.1" "pandas==2.2.2" "tqdm==4.66.4" "scipy==1.13.1" "huggingface-hub==0.23.0"

    # 3. Install Pyannote Stack
    $TARGET_PYTHON -m pip install "pyannote.core==5.0.0" "pyannote.pipeline==3.0.1" "pyannote.metrics==3.2.1" "speechbrain==1.0.0" "asteroid-filterbanks==0.4.0" "omegaconf==2.3.0" "einops==0.7.0" --no-deps
    $TARGET_PYTHON -m pip install "pyannote.audio==3.1.1" "lightning==2.1.4" "torchmetrics==1.2.1" "pytorch-metric-learning==2.3.0" "tensorboardX==2.6" "semver==3.0.2" "torch-audiomentations==0.11.0" --no-deps
    $TARGET_PYTHON -m pip install "openai==1.30.1" "python-dotenv==1.0.1" "requests==2.31.0" --no-deps
    # 1. Restore the missing typing foundations required by Torch
    $TARGET_PYTHON -m pip install "typing-extensions==4.12.2" --no-deps
    # 2. Fix the PIL/Pillow issue found in the matplotlib trace
    $TARGET_PYTHON -m pip install "Pillow==10.3.0" --no-deps
    # 3. Final Guardrail: Ensure NumPy hasn't drifted to 2.x
    $TARGET_PYTHON -m pip install "numpy==1.26.4" --force-reinstall --no-deps
    
    # --- PHASE 4: FINAL INFRASTRUCTURE REPAIR ---
    
    # 1. Foundation & Networking Fixes
    $TARGET_PYTHON -m pip install "urllib3==2.2.1" "idna==3.7" "certifi==2024.2.2" "charset-normalizer==3.3.2" "requests==2.31.0" --no-deps
    $TARGET_PYTHON -m pip install "PyYAML==6.0.1" "filelock==3.13.1" "regex==2023.12.25" "six==1.16.0" "packaging==24.0" "pyparsing==3.1.2" --no-deps

    # 2. AI & Data Model Bridges (Crucial for Transformers/WhisperX)
    # NOTE: We removed tokenizers 0.15.2 from here to avoid the version crash
    $TARGET_PYTHON -m pip install "huggingface-hub==0.23.0" "safetensors==0.4.3" "joblib==1.4.2" "tokenizers==0.19.1" --force-reinstall --no-deps

    # 3. Audio & C-Interface Layer
    $TARGET_PYTHON -m pip install "av==12.1.0" "cffi==1.16.0" "pycparser==2.22" "soundfile==0.12.1" --no-deps

    # 4. Matplotlib & UI Layer Fixes
    $TARGET_PYTHON -m pip install "kiwisolver==1.4.5" "fonttools==4.51.0" "python-dateutil==2.9.0" "cycler==0.12.1" "contourpy==1.2.1" "Pillow==10.3.0" --no-deps

    # 5. The Hardware Anchor (Final lock on NumPy)
    $TARGET_PYTHON -m pip install "numpy==1.26.4" "typing-extensions==4.12.2" --force-reinstall --no-deps


# 2. Install the necessary diarization/NLP helpers that WhisperX needs
#~/.virtualenvs/ai-video-asr/bin/python -m pip install faster-whisper==1.0.3 transformers==4.37.2 nltk
elif [ "$ROLE" == "diarize" ]; then
    # Diarization Stack: Pyannote (Manual layering)
    #$TARGET_PYTHON -m pip install torch==2.1.2 torchaudio==2.1.2 numpy==1.26.4 --extra-index-url https://download.pytorch.org/whl/cu118
    #$TARGET_PYTHON -m pip install speechbrain==1.0.0
    # $TARGET_PYTHON -m pip install pyannote.audio==3.1.1
    #$TARGET_PYTHON -m pip install huggingface_hub omegaconf pytorch-lightning>=2.0 pyyaml pandas==2.1.4 scipy==1.11.4
    # Cluade
    $TARGET_PYTHON -m pip install torch==2.5.1 torchaudio==2.5.1  --index-url https://download.pytorch.org/whl/cu124  --no-deps
    $TARGET_PYTHON -m pip install pyannote.audio==3.3.2 --no-deps
    $TARGET_PYTHON -m pip install pyannote.core pyannote.pipeline pyannote.metrics speechbrain asteroid-filterbanks omegaconf einops
    $TARGET_PYTHON -m pip install openai python-dotenv requests
fi

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

# Set ownership back to the current user for the entire venv tree
echo "🔐 Adjusting permissions for $VENV_PATH..."
sudo chown -R $USER:$USER "$VENV_PATH"

# Clear any build-time path leaks
export PYTHONPATH=""

# 6. Verification Block
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

$TARGET_PYTHON -c "import numpy; import torch; import whisperx; import pyannote.audio; \
import openai; print(f'--- FINAL VERIFICATION ---'); print(f'Numpy: {numpy.__version__} \
(Target: 1.26.4)'); print(f'GPU: {torch.cuda.get_device_name(0)}'); \
print(f'ASR: WhisperX {whisperx.__version__}'); \
print(f'Diarize: Pyannote {pyannote.audio.__version__}'); \
print('🚀 ENVIRONMENT FULLY STABILIZED')"

echo "✅ Env setup complete. To use it, run: source $VENV_PATH/bin/activate . Run cuda_active_check.py to verify."