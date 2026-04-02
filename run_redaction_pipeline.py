import json
import os
import sys
from tracker_interpolator import generate_full_tracking
from apply_redaction import process_video_cuda

def run_redaction_flow(video_in, video_out, ui_manifest_path, tracked_data_path):
    # 1. Load User Selections (The "User-Driven" part)
    if not os.path.exists(ui_manifest_path):
        print(f"❌ Error: {ui_manifest_path} not found. Run get_ui_manifest.py first.")
        return

    with open(ui_manifest_path, 'r') as f:
        ui_selections = json.load(f)
    
    # Filter for IDs where "selected" is True
    selected_ids = {item['id'] for item in ui_selections if item.get('selected') == True}
    
    if not selected_ids:
        print("⚠️ No entities were selected for blurring in ui_manifest.json. Exiting.")
        return
    
    print(f"🎯 IDs selected for blurring: {selected_ids}")

    # 2. Load Raw AI Keyframe Data
    with open(tracked_data_path, 'r') as f:
        raw_ai_data = json.load(f)

    # 3. Filter AI data to ONLY include selected IDs
    filtered_data = {
        entity_id: instances 
        for entity_id, instances in raw_ai_data.items() 
        if entity_id in selected_ids
    }

    # 4. Run Interpolator (Bridge the gaps + Scale 0-1000 to Pixels)
    print("🔄 Generating smooth tracking paths and scaling coordinates...")
    full_manifest = generate_full_tracking(video_in, filtered_data)

    # 5. Execute Final GPU Redaction
    print(f"🚀 Initializing 1080 Ti for rendering {len(full_manifest)} frames...")
    process_video_cuda(video_in, video_out, full_manifest)
    
    print(f"✨ SUCCESS: Redacted video saved to {video_out}")

if __name__ == "__main__":
    # Standard paths for your project structure
    VIDEO_INPUT = "samples/sanitized/input_video.mp4"
    VIDEO_OUTPUT = "output/final_redacted.mp4"
    UI_MANIFEST = "ui_manifest.json"
    TRACKED_JSON = "tracked_manifest.json"
    
    if not os.path.exists("output"): os.makedirs("output")
    
    run_redaction_flow(VIDEO_INPUT, VIDEO_OUTPUT, UI_MANIFEST, TRACKED_JSON)