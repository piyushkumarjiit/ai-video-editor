import os
import ollama
import json
from tqdm import tqdm

# --- CONFIGURATION (EASY TUNING) ---
MODEL_NAME = 'qwen3-vl'
KEYFRAMES_DIR = 'keyframes'
OUTPUT_FILE = 'video_analysis.json'

# 🧪 TEST MODE: Set to a number (e.g., 5) to only process the first few frames.
# Set to None to process the entire video.
LIMIT_FRAMES = 5 

# --- CATEGORICAL REDACTION PROMPT ---
# Optimized for Qwen3-VL Visual Grounding [ymin, xmin, ymax, xmax] (0-1000 scale)
REDACTION_PROMPT = (
    "Perform a high-precision security scan of this frame. "
    "Identify and locate every instance of the following categories. "
    "For each instance, provide a unique ID and its exact coordinates in [ymin, xmin, ymax, xmax] format (scale 0-1000). "
    "\n\nCATEGORIES TO DETECT:"
    "\n1. PEOPLE: Label specifically as 'face_person_1', 'face_cop_1', 'face_witness_1' based on context."
    "\n2. DOCUMENTS: Label as 'doc_license', 'doc_ssn_card', 'doc_credit_card', 'doc_id_badge'."
    "\n3. TEXT PII: Label as 'text_ssn', 'text_address', 'text_dob', 'text_phone'."
    "\n\nOUTPUT FORMAT (Strict JSON list):"
    "\n[{'id': 'unique_id', 'label': 'category_label', 'bbox_2d': [ymin, xmin, ymax, xmax]}]"
    "\nIf nothing is found, return []."
)

def analyze_scenes():
    # Load existing data to support RESUME (won't re-process finished videos)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                analysis_results = json.load(f)
            except json.JSONDecodeError:
                analysis_results = {}
    else:
        analysis_results = {}

    # Get folders (one folder per video) from the keyframes directory
    if not os.path.exists(KEYFRAMES_DIR):
        print(f"❌ Error: {KEYFRAMES_DIR} directory not found.")
        return

    video_folders = [f for f in os.listdir(KEYFRAMES_DIR) if os.path.isdir(os.path.join(KEYFRAMES_DIR, f))]

    if not video_folders:
        print("📁 No video folders found in 'keyframes/'. Run detect_scenes.py first.")
        return

    for video_name in video_folders:
        # Skip if this video is already fully analyzed (and not in test mode)
        if LIMIT_FRAMES is None and video_name in analysis_results and len(analysis_results[video_name]) > 0:
            print(f"⏩ Skipping {video_name} (Already analyzed).")
            continue

        print(f"\n🧠 AI Scanning for Redaction Targets: {video_name}")
        analysis_results[video_name] = []
        
        folder_path = os.path.join(KEYFRAMES_DIR, video_name)
        all_images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])

        # Apply Spot-Check limit if active
        if LIMIT_FRAMES:
            images_to_process = all_images[:LIMIT_FRAMES]
            print(f"🧪 Test Mode: Processing only first {LIMIT_FRAMES} frames.")
        else:
            images_to_process = all_images

        for img_name in tqdm(images_to_process, desc=f"Scanning {video_name}"):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Talking to the Dockerized Ollama container on your R720
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{
                        'role': 'user',
                        'content': REDACTION_PROMPT,
                        'images': [img_path]
                    }]
                )
                
                raw_content = response['message']['content'].strip()
                
                # Clean up potential Markdown code blocks if the AI includes them
                json_str = raw_content.replace('```json', '').replace('```', '').strip()
                
                try:
                    detections = json.loads(json_str)
                except json.JSONDecodeError:
                    # Fallback if AI output isn't perfect JSON
                    detections = json_str 

                analysis_results[video_name].append({
                    "frame": img_name,
                    "detections": detections
                })

            except Exception as e:
                print(f"\n❌ Error on {img_name}: {e}")

        # Save after each video is finished
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(analysis_results, f, indent=4)

    print(f"\n✨ Analysis complete! View results in {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_scenes()