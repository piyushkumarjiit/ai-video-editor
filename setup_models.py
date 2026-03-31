import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# --- DYNAMIC REGISTRY ---
# Switched to Koushik-ai which uses standard LFS (no Xet/Pointer issues)
MODELS_REGISTRY = [
    ("yolov8n-face.pt", "arnabdhar/YOLOv8-Face-Detection", "model.pt"),
    ("license_plate_detector.pt", "koushik-ai/yolov8-license-plate-detection", "best.pt"),
]
TARGET_DIR = "models"

def verify_model(path):
    if not os.path.exists(path): return False
    # Check if it's a tiny pointer file (Xet/LFS issue)
    if os.path.getsize(path) < 10000: 
        print(f"⚠️  {os.path.basename(path)} is too small. Likely a pointer file.")
        return False

    print(f"🔍 Verifying {os.path.basename(path)}...")
    try:
        model = YOLO(path)
        _ = model.names
        print(f"✅ Verified: {path}")
        return True
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
        return False

def sync():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for local_name, repo, remote_name in MODELS_REGISTRY:
        dest = os.path.join(TARGET_DIR, local_name)
        
        if verify_model(dest):
            continue

        success = False
        for r_type in ["model", "dataset"]:
            try:
                print(f"📥 Downloading {remote_name} from {repo}...")
                # We use local_dir_use_symlinks=False to ensure we get the real file
                hf_hub_download(
                    repo_id=repo, 
                    filename=remote_name, 
                    local_dir=TARGET_DIR,
                    local_dir_use_symlinks=False,
                    repo_type=r_type,
                    force_download=True 
                )
                
                temp_path = os.path.join(TARGET_DIR, remote_name)
                
                # Verify BEFORE Renaming
                if verify_model(temp_path):
                    if os.path.exists(dest): os.remove(dest)
                    # Only rename if names are different
                    if os.path.abspath(temp_path) != os.path.abspath(dest):
                        os.rename(temp_path, dest)
                        print(f"🏷️  Renamed {remote_name} -> {local_name}")
                    success = True
                    break
                else:
                    if os.path.exists(temp_path): os.remove(temp_path)
            except Exception as e:
                print(f"❌ {r_type} attempt failed: {e}")

        if not success:
            print(f"❌ CRITICAL: Failed to sync {local_name}")

if __name__ == "__main__":
    sync()