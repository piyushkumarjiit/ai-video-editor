import os
import sys
import warnings
import numpy as np
import json
import torch
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel, logging as hf_logging

# --- TOP LEVEL SILENCING ---
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

# --- CONFIG ---
MANIFEST_DIR = "tracking"
OUTPUT_DIR = "tracking/manifests_cleaned"
VIDEO_DIR = "samples/sanitized"
MODEL_ID = "openai/clip-vit-base-patch32"

# Debugging & Visualization
VISUALIZE = True       
DEBUG_DIR = "debug"

# 🚀 OSNet-Style Tuning
EPS = 0.16             # Strict visual matching
BATCH_SIZE = 64        # Stabilized for 1080 Ti
SAMPLES_PER_ID = 5     # Frames to average per tracklet
OVERLAP_LIMIT = 45     # Max frame overlap for "handoff" merges
DIST_THRESHOLD = 550   # Spatial movement tolerance
MIN_TRACK_LEN = 15     # Purge blips shorter than 0.5s (prevents object flicker)
INWARD_CROP = 0.20     # 🚀 Aggressive 20% crop to focus on face/torso only

load_dotenv()

def extract_tensor(outputs):
    """Robustly extracts 2D tensor from various CLIP output types."""
    if torch.is_tensor(outputs):
        emb = outputs
    elif hasattr(outputs, "image_embeds"):
        emb = outputs.image_embeds
    elif hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output
    else:
        emb = outputs[0]
    if len(emb.shape) == 3:
        emb = emb[:, 0, :] 
    return emb

def letterbox_crop(image, expected_size=(224, 224)):
    """Resizes and pads image to square to prevent CLIP distortion and handles alpha channels."""
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    ih, iw = image.shape[:2]
    w, h = expected_size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    image_resized = cv2.resize(image, (nw, nh))
    
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    dy, dx = (h - nh) // 2, (w - nw) // 2
    new_image[dy:dy+nh, dx:dx+nw, :] = image_resized
    return new_image

def get_centroid(bbox):
    """Returns (x, y) center of a bounding box [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def run_reid():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if VISUALIZE:
        os.makedirs(f"{DEBUG_DIR}/visual_clusters", exist_ok=True)
        os.makedirs(f"{DEBUG_DIR}/final_identities", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 GPU ENGINE: {torch.cuda.get_device_name(0)}")
    
    model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    for m_file in [f for f in os.listdir(MANIFEST_DIR) if f.endswith("_tracking.json")]:
        print(f"\n🔄 Processing: {m_file}")
        with open(os.path.join(MANIFEST_DIR, m_file), "r") as f:
            data = json.load(f)
        
        video_path = os.path.join(VIDEO_DIR, data.get("source_video"))
        
        # 🚀 FILTER: Keep only 'person' class and long-duration tracks
        entities = {k: v for k, v in data["entities"].items() 
                   if v.get("label", "person") == "person" and len(v.get("trajectory", [])) >= MIN_TRACK_LEN}
        
        if not entities: continue

        request_list = []
        for eid, info in entities.items():
            traj = info.get("trajectory", [])
            indices = np.linspace(0, len(traj) - 1, min(len(traj), SAMPLES_PER_ID), dtype=int)
            for idx in indices:
                request_list.append({"eid": eid, "frame": traj[idx]["frame"], "bbox": traj[idx]["bbox"]})
        request_list = sorted(request_list, key=lambda x: x["frame"])

        try:
            reader = cv2.cudacodec.createVideoReader(video_path)
            use_cuda = True
            print(" 📼 Using CUDA Hardware Decoder (NVDEC)...")
        except:
            cap = cv2.VideoCapture(video_path)
            use_cuda = False
            print(" ⚠️ NVDEC failed, using CPU.")

        id_to_embeddings = {eid: [] for eid in entities.keys()}
        id_to_best_crop = {} 
        batch_imgs, batch_eids = [], []
        current_frame = -1

        print(f" 📸 Extracting fingerprints from {len(request_list)} samples...")

        for req in request_list:
            frame = None
            if use_cuda:
                while current_frame < req["frame"]:
                    ret, gpu_mat = reader.nextFrame()
                    if not ret: break
                    current_frame += 1
                if current_frame == req["frame"]: frame = gpu_mat.download()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, req["frame"])
                ret, frame = cap.read()

            if frame is not None and req["bbox"]:
                x1, y1, x2, y2 = map(int, req["bbox"])
                bw, bh = x2 - x1, y2 - y1
                
                # 🚀 Filter non-human shaped boxes
                if bw < 40 or bh < 80: continue

                # 🚀 Aggressive Inward Crop
                hp, wp = int(bh * INWARD_CROP), int(bw * INWARD_CROP)
                crop = frame[max(0, y1+hp):min(frame.shape[0], y2-hp), 
                             max(0, x1+wp):min(frame.shape[1], x2-wp)]
                
                if crop.size > 0:
                    sq_crop = letterbox_crop(crop)
                    batch_imgs.append(Image.fromarray(cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)))
                    batch_eids.append(req["eid"])
                    if req["eid"] not in id_to_best_crop:
                        id_to_best_crop[req["eid"]] = sq_crop

            if len(batch_imgs) >= BATCH_SIZE:
                inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                    emb = extract_tensor(outputs)
                    numpy_embs = normalize(emb.detach().cpu().numpy())
                    for eid, e_vec in zip(batch_eids, numpy_embs):
                        id_to_embeddings[eid].append(e_vec)
                batch_imgs, batch_eids = [], []

        if batch_imgs:
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                emb = extract_tensor(outputs)
                numpy_embs = normalize(emb.detach().cpu().numpy())
                for eid, e_vec in zip(batch_eids, numpy_embs):
                    id_to_embeddings[eid].append(e_vec)

        valid_ids, all_embeddings = [], []
        for eid, embs in id_to_embeddings.items():
            if embs:
                all_embeddings.append(np.mean(embs, axis=0))
                valid_ids.append(eid)

        if not all_embeddings: continue
        all_embs_np = np.array(all_embeddings)
        print(f" 🧠 Clustering {len(all_embs_np)} identities...")
        clustering = DBSCAN(eps=EPS, min_samples=1, metric="cosine").fit(all_embs_np)
        
        unique_labels = set(clustering.labels_)
        print(f" 📊 CLIP identified {len(unique_labels)} visual identities.")

        if VISUALIZE:
            for label in unique_labels:
                if label == -1: continue
                idx = np.where(clustering.labels_ == label)[0][0]
                eid = valid_ids[idx]
                if eid in id_to_best_crop:
                    cv2.imwrite(f"{DEBUG_DIR}/visual_clusters/cluster_{label}.jpg", id_to_best_crop[eid])

        new_entities, cluster_map = {}, {}
        for i, cluster_id in enumerate(clustering.labels_):
            orig_eid = valid_ids[i]
            master_id = cluster_map.get(cluster_id)

            if master_id is None:
                cluster_map[cluster_id] = orig_eid
                new_entities[orig_eid] = entities[orig_eid]
                new_entities[orig_eid]["merged_ids"] = [orig_eid]
            else:
                master_traj = {t["frame"]: t["bbox"] for t in new_entities[master_id]["trajectory"]}
                new_traj = {t["frame"]: t["bbox"] for t in entities[orig_eid]["trajectory"]}
                overlap_frames = set(master_traj.keys()) & set(new_traj.keys())
                
                should_merge = True
                if len(overlap_frames) > OVERLAP_LIMIT:
                    for f in list(overlap_frames)[:5]:
                        c1, c2 = get_centroid(master_traj[f]), get_centroid(new_traj[f])
                        if np.linalg.norm(np.array(c1) - np.array(c2)) > DIST_THRESHOLD:
                            should_merge = False
                            break
                
                if should_merge:
                    new_entities[master_id]["trajectory"].extend(
                        [t for t in entities[orig_eid]["trajectory"] if t["frame"] not in master_traj]
                    )
                    new_entities[master_id]["merged_ids"].append(orig_eid)
                else:
                    new_entities[orig_eid] = entities[orig_eid]
                    new_entities[orig_eid]["merged_ids"] = [orig_eid]

        if VISUALIZE:
            for mid in new_entities:
                if mid in id_to_best_crop:
                    cv2.imwrite(f"{DEBUG_DIR}/final_identities/id_{mid}.jpg", id_to_best_crop[mid])

        for eid in new_entities:
            new_entities[eid]["trajectory"] = sorted(new_entities[eid]["trajectory"], key=lambda x: x["frame"])
            
        out_path = os.path.join(OUTPUT_DIR, m_file.replace("_tracking.json", "_reid_cleaned.json"))
        with open(out_path, "w") as f:
            json.dump(data | {"entities": new_entities}, f, indent=2)
        
        print(f" ✨ Final: {len(entities)} -> {len(new_entities)} Identities.")
        if not use_cuda: cap.release()

if __name__ == "__main__":
    run_reid()