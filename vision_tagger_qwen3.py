# vision_tagger_qwen3.py
import base64
import requests
import json
import cv2
import os

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-vl" # Update if using a specific quantization like qwen3-vl

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def classify_entity(crop_path):
    img_b64 = encode_image(crop_path)
    prompt = """Analyze this cropped image of a person or object. 
    Return ONLY a JSON object with a single key 'category'. 
    Choose strictly from: [Police Officer, Witness, Suspect, Document, License Plate, Unknown]."""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        result = response.json()
        category = json.loads(result["response"]).get("category", "Unknown")
        return category
    except Exception as e:
        print(f"[ERROR] VLM Classification failed: {e}")
        return "Unknown"

def generate_ui_manifest(tracking_json, video_path, output_manifest):
    with open(tracking_json, "r") as f:
        data = json.load(f)
    
    cap = cv2.VideoCapture(video_path)
    manifest = {"video_file": video_path, "entities": {}}
    os.makedirs("temp_crops", exist_ok=True)

    for entity_id, info in data["entities"].items():
        # Get a frame from the middle of the track for the best crop
        mid_idx = len(info["trajectory"]) // 2
        target_frame = info["trajectory"][mid_idx]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame["frame"])
        ret, frame = cap.read()
        if ret:
            x1, y1, x2, y2 = target_frame["bbox"]
            crop = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20] # Add 20px padding
            crop_path = f"temp_crops/entity_{entity_id}.jpg"
            cv2.imwrite(crop_path, crop)
            
            category = classify_entity(crop_path)
            print(f"Entity {entity_id} classified as: {category}")
            
            manifest["entities"][entity_id] = {
                "category": category,
                "blur_recommended": category in ["Witness", "Suspect", "Document", "License Plate"],
                "thumbnail": crop_path
            }

    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)