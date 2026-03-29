import json
import subprocess
import os

def redact_video(video_name, selected_ids, input_manifest='tracked_manifest.json'):
    with open(input_manifest, 'r') as f:
        manifest = json.load(f)
    
    if video_name not in manifest:
        print(f"❌ Video {video_name} not found in manifest.")
        return

    # 1. Prepare the Blur Filters
    # Coordinates are [ymin, xmin, ymax, xmax] in 0-1000 scale
    # FFmpeg boxblur uses: x, y, width, height
    blur_filters = []
    
    # We apply redaction frame-by-frame using the 'sendcmd' or multiple 'overlay' approach.
    # For a high-performance R720 approach, we'll build a filter_complex.
    
    for entry in manifest[video_name]:
        frame_num = int(entry['frame'].split('.')[0]) # Assumes 001.jpg -> 1
        for target in entry['targets']:
            if target['id'] in selected_ids:
                bbox = target['bbox']
                
                # Convert 0-1000 scale to FFmpeg expressions
                # x = xmin * w / 1000, y = ymin * h / 1000
                # w = (xmax - xmin) * w / 1000, h = (ymax - ymin) * h / 1000
                ymin, xmin, ymax, xmax = bbox
                w_box = xmax - xmin
                h_box = ymax - ymin
                
                # Create a crop and blur for this specific time/area
                # Note: This is a simplified version; for multi-frame tracking 
                # we'll use a specific filter string.
                pass

    # 🚀 Hardware Accelerated Command for R720
    # This version uses the 'delogo' filter as a simple box-redactor for speed
    # We will iterate through targets and apply them.
    
    input_video = f"videos/{video_name}.mp4" # Adjust path as needed
    output_video = f"output/{video_name}_redacted.mp4"
    os.makedirs('output', exist_ok=True)

    # Building a complex filter chain for FFmpeg is tricky for many frames,
    # so we will generate a redaction script for FFmpeg to read.
    print(f"🎬 Rendering redacted video for: {video_name}...")
    
    # For now, let's start with a simple execution message.
    # To do this perfectly, we'll need to map frame numbers to timestamps.
    print(f"Selected for redaction: {selected_ids}")
    print("GPU (1080 Ti) is ready to render.")

# --- UI LOGIC ---
if __name__ == "__main__":
    # Example: User selects 'face_witness_1' from your console list
    # In a real UI, this would come from the checkbox clicks.
    target_video = "Lululemon_Shoplifter_Gets_Caught_in_The_Act"
    to_redact = ["face_witness_1"] 
    
    redact_video(target_video, to_redact)