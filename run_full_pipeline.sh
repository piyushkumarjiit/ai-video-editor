# -------------------------------------------------------------------------
# FILE: run_full_pipeline.sh
# ROLE: End-to-End Automation & Execution Wrapper
#
# DESCRIPTION:
# Orchestrates the entire video processing lifecycle. Manages the 
# sequence of Sanitization (NVENC), AI Grounding (Qwen-VL), and 
# Final Redaction (CUDA), with a pause for manual user selection.
#
# HARDWARE COMPATIBILITY:
# - Uses 1080 Ti for stage 1 (NVENC) and stage 5 (CUDA Redaction).
# - Manages OLLAMA_HOST environment variables for Dockerized AI.
# -------------------------------------------------------------------------

#!/bin/bash

# --- DOCKER & NETWORK CONFIG ---
# This tells your Python scripts to look at your Docker IP
export OLLAMA_HOST="http://192.168.2.229:11434"

# --- CONFIGURATION ---
INPUT_VIDEO="samples/raw/my_video.mp4"
SANITIZED_VIDEO="samples/sanitized/Karen_Fights_Real_Estate_Agent_Over_House.mp4"
UI_MANIFEST="ui_manifest.json"

# 1. Start Environment
echo "🔧 Activating AI Environment..."
source ai-video-env/bin/activate

# 2. Sanitize & Standardize (NVENC)
echo "🎥 Stage 1: Sanitizing video using 1080 Ti NVENC..."
#python3 sanitize_videos.py --input "$INPUT_VIDEO" --output "$SANITIZED_VIDEO"

# 3. Extract Scenes & Global Offsets
echo "🎞️ Stage 2: Extracting scenes and recording offsets..."
#python3 detect_scenes.py

# 4. AI Vision Grounding (Qwen-VL)
# Note: This will now automatically use the OLLAMA_HOST variable
echo "🧠 Stage 3: Running AI Vision Analysis via Docker (Qwen3-VL)..."
python3 vision_analyze.py

# 5. Build UI Manifest
echo "📝 Stage 4: Generating User Selection Manifest..."
python3 get_ui_manifest.py

# --- THE CRITICAL PAUSE ---
echo "--------------------------------------------------------"
echo "✋ STOP! AI analysis is complete."
echo "Please open '$UI_MANIFEST' in VS Code."
echo "Change 'selected': false to 'selected': true for IDs you want to blur."
echo "--------------------------------------------------------"
read -p "Press [Enter] once you have saved your selections to start the GPU Render..."

# 6. Final GPU Redaction (Interpolation + CUDA Blur)
echo "🚀 Stage 5: Finalizing Redaction on 1080 Ti..."
python3 run_redaction_pipeline.py

echo "✨ DONE! Your redacted video is in the 'output/' folder."