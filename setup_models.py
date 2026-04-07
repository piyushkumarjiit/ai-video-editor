"""
FILE: setup_models.py
ROLE: Centralized AI Model Synchronization & Asset Management
-------------------------------------------------------------------------
DESCRIPTION:
An initialization utility that automates the downloading and 
verification of YOLO and Pyannote/WhisperX models from Hugging Face. 
It ensures all weights are stored in a local directory to support 
offline inference and validates model integrity before deployment.

HARDWARE COMPATIBILITY:
- Optimized for environments using the NVIDIA 1080 Ti for inference.
- Requires a valid Hugging Face Token (HF_TOKEN) for repository access.
-------------------------------------------------------------------------
"""

from dotenv import load_dotenv
import os
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from ultralytics import YOLO

# --- CONFIGURATION ---
load_dotenv() # This loads the variables from the .env file
HF_TOKEN = os.getenv("HF_TOKEN")
# CRITICAL: Point this to your large data drive to save root space!
TARGET_DIR = "models" 

# Registry for YOLO (.pt files)
YOLO_REGISTRY = [
    ("yolov8n.pt", "ultralytics/yolov8", "yolov8n.pt"), # Base model for testing
    ("yolov8n-face.pt", "arnabdhar/YOLOv8-Face-Detection", "model.pt"),
    ("license_plate_detector.pt", "koushik-ai/yolov8-license-plate-detection", "best.pt"),
]

# Registry for WhisperX/Diarization (Full Repositories)
REPO_REGISTRY = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "guillaumekln/faster-whisper-medium", # Centralize the actual Whisper weights
    "SAI-Sreeram/whisperx-vad-segmentation" 
]

def verify_yolo(path):
    """Ensures the YOLO model is valid and loadable on the 1080 Ti."""
    if not os.path.exists(path) or os.path.getsize(path) < 10000: return False
    try:
        # Load in CPU mode just for verification to save VRAM during sync
        model = YOLO(path)
        return True
    except: return False

def sync():

    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable not found.")
        print("💡 Run: export HF_TOKEN='your_token_here' or add it to ~/.bashrc")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Set HF_HOME globally for this session so libraries find these models
    os.environ["HF_HOME"] = TARGET_DIR

    print(f"📂 Centralizing models in: {TARGET_DIR}")

    # 1. Sync YOLO Models
    for local_name, repo, remote_name in YOLO_REGISTRY:
        dest = os.path.join(TARGET_DIR, local_name)
        if not verify_yolo(dest):
            print(f"✅ Using token from environment to sync to {TARGET_DIR}")
            print(f"📥 Downloading YOLO: {local_name}...")
            hf_hub_download(
                repo_id=repo, 
                filename=remote_name, 
                local_dir=TARGET_DIR,
                token=HF_TOKEN
            )
            # Handle renaming if remote name differs from our local preference
            downloaded_path = os.path.join(TARGET_DIR, remote_name)
            if os.path.exists(downloaded_path) and remote_name != local_name:
                os.rename(downloaded_path, dest)

    # 2. Sync Full Repos (WhisperX / Pyannote)
    for repo_id in REPO_REGISTRY:
        print(f"📥 Syncing Repository: {repo_id}...")
        snapshot_download(
            repo_id=repo_id, 
            local_dir=os.path.join(TARGET_DIR, "hub", repo_id.replace("/", "--")), 
            token=HF_TOKEN,
            # This ensures we don't download .bin if .safetensors exist (saves space)
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"] 
        )

    print("\n✅ All models synced. Ready for offline inference.")

if __name__ == "__main__":
    sync()