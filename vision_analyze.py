"""
FILE: vision_analyze.py
ROLE: Contextual Scene Intelligence & Structured JSON Generation
-------------------------------------------------------------------------
DESCRIPTION:
Performs high-level reasoning by analyzing pre-boxed images from YOLO. 
It merges visual evidence with audio transcriptions to assign specific 
roles (e.g., Suspect, Victim, Cop) to detected targets.

INPUT: 
- Images in 'qwen_input/' (expected to have YOLO 'green boxes').
- Audio context/transcription passed via prompt.

OUTPUT:
- video_analysis.json: Detailed list of roles and bounding boxes 
  normalized for downstream redaction or reporting.

HARDWARE COMPATIBILITY:
- Uses Ollama (qwen3-vl). Optimized for Pascal (1080 Ti) via quantization.
-------------------------------------------------------------------------
"""

import os
import ollama
import json
import ast
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_NAME = 'qwen3-vl'
#KEYFRAMES_DIR = 'keyframes' # old input folder
KEYFRAMES_DIR = 'qwen_input' # fodler containing boxed images after YOLO analysis
OUTPUT_FILE = 'video_analysis.json'
LIMIT_FRAMES = 5  # Set to None for full processing

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=ollama_host)
print(f"🔗 Connecting to Ollama at: {ollama_host}")

# --- YOUR PREFERRED PROMPT ---
'''REDACTION_PROMPT = (
    "Perform a high-precision security scan of this frame. "
    "Identify and locate every instance of the following categories. "
    "For each instance, provide a unique ID and its exact coordinates in [ymin, xmin, ymax, xmax] format (scale 0-1000). "
    "\n\nCATEGORIES TO DETECT:"
    "\n1. PEOPLE: Label specifically as 'face_person_1', 'face_cop_1', 'face_witness_1', 'face_bystander_1', 'face_driver_1', 'face_suspect_1',"
    " 'face_fireman_1', 'face_emt_1' etc. based on context."
    "\n2. DOCUMENTS: Label as 'doc_license', 'doc_ssn_card', 'doc_credit_card', 'doc_id_badge'."
    "\n3. TEXT PII: Label as 'text_ssn', 'text_address', 'text_dob', 'text_phone'."
    "\n\nOUTPUT FORMAT (Strict JSON list):"
    "\n[{'id': 'unique_id', 'label': 'category_label', 'bbox_2d': [ymin, xmin, ymax, xmax]}]"
    "\nCOORDINATE RULE: Provide a bounding box [ymin, xmin, ymax, xmax] (scale 0-1000) that starts at the top of the hair/forehead"
     " and ends at the base of the chin. The width should span from the left ear to the right ear."
     "\nIf nothing is found, return []."
    "\nCRITICAL: Do not use Markdown code blocks. Do not nest results under a 'result' key. Always use 'bbox_2d' for the key name."
)
'''
# Updated prompt for YOLOed images
REDACTION_PROMPT = f"""
I have provided a frame with green boxes labeled TARGET_0, TARGET_1, etc.
AUDIO CONTEXT FROM THIS SCENE: "{transcription_text}"

TASK: Identify the role of each TARGET based on the visual evidence (clothing, tools, actions, location etc.) AND the provided audio context.

ROLES TO CHOOSE FROM:
- Public Service: Cop, Security Guard, EMT, Firefighter.
- Transit/Street: Bus Driver, Passenger, Driver, Walker, Passerby.
- Commercial: Shopper, Salesperson, Real Estate Agent, Manager.
- Legal: Suspect, Witness, Victim.

OUTPUT FORMAT (Strict JSON list):
[
  {{"id": "TARGET_0", "role": "identified_role", "reason": "short explanation"}}
]

CRITICAL: If a person's role is unclear, default to 'Passerby' or 'Bystander' rather than returning an empty list.
"""
def normalize_detections(raw_content):
    """Cleans up inconsistencies in AI output before saving."""
    try:
        # Remove Markdown if present
        clean_str = raw_content.replace('```json', '').replace('```', '').strip()
        
        # Try standard JSON first, fallback to literal_eval for single-quote strings
        try:
            data = json.loads(clean_str)
        except json.JSONDecodeError:
            data = ast.literal_eval(clean_str)

        # Flatten nested 'result' key if the AI hallucinated it
        if isinstance(data, dict) and 'result' in data:
            data = data['result']
        
        # Ensure result is always a list
        if isinstance(data, dict):
            data = [data]
        
        if not isinstance(data, list):
            return []

        # Force key consistency: 'bbox' -> 'bbox_2d'
        standardized = []
        for item in data:
            if not isinstance(item, dict): continue
            
            box = item.get('bbox_2d') or item.get('bbox')
            label = item.get('label') or item.get('category') or "target"
            
            if box and len(box) == 4:
                standardized.append({
                    "id": item.get('id', label),
                    "label": label,
                    "bbox_2d": box
                })
        return standardized
    except Exception as e:
        print(f"⚠️ Normalization Warning: {e}")
        return []

def analyze_scenes():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try: analysis_results = json.load(f)
            except: analysis_results = {}
    else:
        analysis_results = {}

    video_folders = [f for f in os.listdir(KEYFRAMES_DIR) if os.path.isdir(os.path.join(KEYFRAMES_DIR, f))]

    for video_name in video_folders:
        # Check if already processed (and not in test mode)
        if LIMIT_FRAMES is None and video_name in analysis_results:
            print(f"⏩ Skipping {video_name}")
            continue

        print(f"\n🧠 AI Scanning: {video_name}")
        analysis_results[video_name] = []
        folder_path = os.path.join(KEYFRAMES_DIR, video_name)
        all_images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])
        images_to_process = all_images[:LIMIT_FRAMES] if LIMIT_FRAMES else all_images

        for img_name in tqdm(images_to_process, desc=f"Scanning {video_name}"):
            img_path = os.path.join(folder_path, img_name)
            try:
                response = client.chat(
                    model=MODEL_NAME,
                    messages=[{'role': 'user', 'content': REDACTION_PROMPT, 'images': [img_path]}]
                )
                
                raw_content = response['message']['content'].strip()
                clean_detections = normalize_detections(raw_content)

                analysis_results[video_name].append({
                    "frame": img_name,
                    "detections": clean_detections
                })

            except Exception as e:
                print(f"\n❌ Error on {img_name}: {e}")

        # Save progress after each video
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(analysis_results, f, indent=4)

    print(f"\n✨ Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_scenes()