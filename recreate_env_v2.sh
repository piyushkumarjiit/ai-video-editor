#!/bin/bash
# 1. Create the environment
python3 -m venv ~/.virtualenvs/ai-video-v2
source ~/.virtualenvs/ai-video-v2/bin/activate

# 2. Install Torch and Audio first (The heavy lifting)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir

# 3. Install the rest of the requirements
pip install -r requirements.txt

# 4. The Critical "Manual" Step for Ultralytics
pip install ultralytics --no-deps --no-cache-dir

# 5. Link your custom OpenCV-CUDA build (if not in site-packages)
# ln -s /path/to/opencv/build/cv2.so ~/.virtualenvs/ai-video-v2/lib/python3.12/site-packages/

echo "✅ Env 2 is ready. Run cuda_active_check.py to verify."