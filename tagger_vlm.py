import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# --- MODEL PATH CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.environ['HF_HOME'] = MODELS_DIR
# This stops it from checking the internet every time you run it
os.environ['TRANSFORMERS_OFFLINE'] = "1" 

from transformers import CLIPProcessor, CLIPModel

# --- CONFIG ---
MANIFEST_DIR = "tracking"
OUTPUT_DIR = "tracking/manifests_cleaned"
VIDEO_DIR = "samples/sanitized"
MODEL_ID = "openai/clip-vit-base-patch32"

def run_reid():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 GPU ENGINE: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"🎨 Loading CLIP model...")
    
    try:
        model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception as e:
        print("🌐 Local model not found, attempting one last download...")
        os.environ['TRANSFORMERS_OFFLINE'] = "0"
        model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)

    manifests = [f for f in os.listdir(MANIFEST_DIR) if f.endswith("_tracking.json")]
    
    for m_file in manifests:
        print(f"\n🔄 De-duplicating: {m_file}")
        with open(os.path.join(MANIFEST_DIR, m_file), "r") as f:
            data = json.load(f)
        
        video_filename = data.get("source_video")
        video_path = os.path.join(VIDEO_DIR, video_filename)
        cap = cv2.VideoCapture(video_path)
        
        valid_ids = []
        embeddings = []
        entities = data["entities"]
        
        print(f" 📸 Extracting fingerprints for {len(entities)} IDs...")
        
        for eid, info in entities.items():
            target_frame = info.get("best_frame", info.get("first_frame", 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                traj = info.get("trajectory", [])
                frame_data = next((item for item in traj if item.get("frame") == target_frame), traj[0])
                box = frame_data.get("bbox")
                
                if box:
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    
                    if crop.size > 0:
                        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            # FIX: Use vision_model to get the pooled output directly
                            vision_outputs = model.get_image_features(**inputs)
                            # FIX: vision_outputs IS the tensor, so we detach it directly
                            emb = vision_outputs.detach().cpu().numpy().flatten()
                            embeddings.append(emb)
                            valid_ids.append(eid)

        if not embeddings:
            continue

        # --- CLUSTERING ---
        embeddings = normalize(np.array(embeddings))
        clustering = DBSCAN(eps=0.35, min_samples=1, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        new_entities = {}
        cluster_map = {}

        for i, cluster_id in enumerate(labels):
            original_eid = valid_ids[i]
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = original_eid
                new_entities[original_eid] = entities[original_eid]
                new_entities[original_eid]["merged_ids"] = [original_eid]
            else:
                master_id = cluster_map[cluster_id]
                new_entities[master_id]["trajectory"].extend(entities[original_eid]["trajectory"])
                new_entities[master_id]["merged_ids"].append(original_eid)

        # Sort trajectories
        for eid in new_entities:
            new_entities[eid]["trajectory"] = sorted(new_entities[eid]["trajectory"], key=lambda x: x["frame"])

        data["entities"] = new_entities
        out_path = os.path.join(OUTPUT_DIR, m_file.replace("_tracking.json", "_reid_cleaned.json"))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f" ✨ SUCCESS: {len(entities)} IDs -> {len(new_entities)} Identities.")
        cap.release()

if __name__ == "__main__":
    run_reid()