#!/bin/bash

# Configuration
VENV_PATH="$HOME/.virtualenvs/ai-video-tensor"
PYTHON_BIN="python3.11"

echo "🚀 Starting 1:1 Environment Sync ($PYTHON_BIN + CUDA 12.6)..."

# 0. Delete the old problematic VENV
echo "🗑️ Cleaning up old environment if it exists..."
rm -rf "$VENV_PATH"


# 1. Fresh VENV Creation
$PYTHON_BIN -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# 2. Base Tooling
pip install --upgrade pip==26.1 setuptools==82.0.1 wheel==0.47.0

# 3. Layer 0: Foundations & Networking
echo "📦 Installing Foundations & Networking..."
pip install --no-deps \
    attrs==26.1.0 distro==1.9.0 filelock==3.25.2 h11==0.16.0 \
    ml-dtypes==0.4.1 numpy==2.0.2 packaging==26.0 platformdirs==4.3.6 \
    psutil==7.2.2 python-dateutil==2.9.0.post0 pytz==2026.1.post1 \
    six==1.17.0 sniffio==1.3.1 typing-inspection==0.4.2 \
    typing_extensions==4.15.0 tzdata==2026.1 \
    cffi==2.0.0 pycparser==3.0 PySocks==1.7.1 \
    cryptography==46.0.7 python-dotenv==1.2.2 wcwidth==0.6.0

# 4. Layer 1: Communication & Google Stack
echo "📦 Installing Communication & Google Auth Stack..."
pip install --no-deps \
    aiohappyeyeballs==2.6.1 aiohttp==3.13.5 aiosignal==1.4.0 \
    certifi==2026.2.25 charset-normalizer==3.4.7 frozenlist==1.8.0 \
    google-api-core==2.30.3 google-api-python-client==2.194.0 \
    google-auth==2.49.2 google-auth-httplib2==0.3.1 \
    google-auth-oauthlib==1.3.1 googleapis-common-protos==1.74.0 \
    httpcore==1.0.9 httplib2==0.31.2 httpx==0.28.1 idna==3.11 \
    multidict==6.7.1 oauthlib==3.3.1 propcache==0.4.1 \
    proto-plus==1.27.2 protobuf==7.34.1 pyasn1==0.6.3 \
    pyasn1_modules==0.4.2 requests==2.33.1 requests-oauthlib==2.0.0 \
    rsa==4.9 uritemplate==4.2.0 urllib3==2.6.3 yarl==1.23.0 \
    fsspec==2026.2.0 jiter==0.14.0

# 5. Layer 2: Core AI Stack (Torch 2.7.1 + cu126)
echo "📦 Installing Torch 2.7.1 ..."
pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 \
    torchaudio==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    triton==3.3.1

# 5. Layer 2.5: Core AI Stack (cu126)
echo "📦 Installing NVIDIA CUDA 12.6 Runtimes..."
pip install --no-deps \
    cuda-bindings==12.9.4 cuda-pathfinder==1.2.2 cuda-toolkit==12.6.3 \
    nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-cupti-cu12==12.6.80 \
    nvidia-cuda-nvrtc-cu12==12.6.77 nvidia-cuda-runtime-cu12==12.6.77 \
    nvidia-cudnn-cu12==9.5.1.17 nvidia-cufft-cu12==11.3.0.4 \
    nvidia-cufile-cu12==1.11.1.6 nvidia-curand-cu12==10.3.7.77 \
    nvidia-cusolver-cu12==11.7.1.2 nvidia-cusparse-cu12==12.5.4.2 \
    nvidia-cusparselt-cu12==0.6.3 nvidia-nccl-cu12==2.26.2 \
    nvidia-nvjitlink-cu12==12.6.85 nvidia-nvshmem-cu12==3.4.5 \
    nvidia-nvtx-cu12==12.6.77

# 6. Layer 3: Math, Data & Logic
echo "📦 Installing Math, Data & Analytics..."
pip install --no-deps \
    contourpy==1.3.3 cycler==0.12.1 fonttools==4.62.1 joblib==1.5.3 \
    kiwisolver==1.5.0 matplotlib==3.10.8 mpmath==1.3.0 networkx==3.6.1 \
    pandas==2.2.3 pyparsing==3.3.2 scikit-learn==1.8.0 scipy==1.17.1 \
    sympy==1.14.0 threadpoolctl==3.6.0 \
    nltk==3.9.4 optuna==4.8.0 primePy==1.3

# 7. Layer 4: Vision & ONNX
echo "📦 Installing Vision & Tracking..."
pip install --no-deps \
    ImageIO==2.37.3 boxmot==18.0.0 filterpy==1.4.5 flatbuffers==25.12.19 \
    imageio-ffmpeg==0.6.0 lap==0.5.13 lapx==0.9.4 onnxruntime==1.24.4 \
    pillow==11.3.0 screeninfo==0.8.1 \
    ultralytics==8.4.37 clip-anytorch==2.6.0 yacs==0.1.8

# 8. Layer 5: Transformers & Audio
echo "📦 Installing Transformers & Audio..."
pip install --no-deps \
    accelerate==1.13.0 einops==0.8.2 ftfy==6.3.1 huggingface_hub==1.10.1 \
    regex==2026.4.4 safetensors==0.7.0 sentencepiece==0.2.1 tokenizers==0.22.2 \
    tqdm==4.67.3 transformers==5.5.3 \
    asteroid-filterbanks==0.4.0 ctranslate2==4.7.1 faster-whisper==1.2.1 \
    HyperPyYAML==1.2.3 julius==0.2.7 pyannote.audio==3.4.0 \
    pyannote.core==5.0.0 pyannote.database==5.1.3 pyannote.metrics==3.2.1 \
    pyannote.pipeline==3.0.1 soundfile==0.13.1 speechbrain==1.1.0 \
    torch-audiomentations==0.12.0 torch_pitch_shift==1.2.5 \
    torchmetrics==1.9.0 whisperx==3.7.2

# 9. Layer 6: Final Utilities
echo "📦 Installing Final Utilities..."
pip install --no-deps \
    Jinja2==3.1.6 Mako==1.3.10 MarkupSafe==3.0.3 Pygments==2.20.0 \
    SQLAlchemy==2.0.49 alembic==1.18.4 annotated-doc==0.0.4 \
    annotated-types==0.7.0 antlr4-python3-runtime==4.9.3 anyio==4.13.0 \
    av==15.1.0 beautifulsoup4==4.14.3 click==8.3.2 colorlog==6.10.1 \
    decorator==5.2.1 diskcache==5.6.3 docopt==0.6.2 gdown==6.0.0 \
    greenlet==3.4.0 hf-xet==1.4.3 lightning==2.6.1 lightning-utilities==0.15.3 \
    llama_cpp_python==0.3.20 loguru==0.7.3 markdown-it-py==4.0.0 \
    mdurl==0.1.2 moviepy==2.2.1 ollama==0.6.1 omegaconf==2.3.0 \
    openai==2.31.0 proglog==0.1.12 pydantic==2.13.0 pydantic_core==2.46.0 \
    pytorch-lightning==2.6.1 pyyaml==6.0.3 rich==15.0.0 ruamel.yaml==0.18.17 \
    ruamel.yaml.clib==0.2.15 semver==3.0.4 shellingham==1.5.4 \
    sortedcontainers==2.4.0 soupsieve==2.8.3 tabulate==0.10.0 \
    tensorboardX==2.6.5 toml==0.10.2 typer==0.24.1 yt-dlp==2026.3.17 \
    pytorch-metric-learning==2.9.0

echo "✅ Script complete. Verify with 'pip freeze'."