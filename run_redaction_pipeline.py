import json
import os
from tracker_interpolator import generate_full_tracking
from apply_redaction import process_video_cuda

def main():
    video_path = "samples/sanitized/input_video.mp4" # Your R720 path
    ai_results_path = "tracked_manifest.json"
    output_path = "output_redacted.mp4"

    # 1. Load the AI Keyframe Data
    with open(ai_results_path, 'r') as f:
        sparse_manifest = json.load(f)

    print("🔄 Interpolating coordinates and scaling to video resolution...")
    
    # 2. GENERATE FULL TRACKING (The Missing Link)
    # This fills the gaps between keyframes and scales 0-1000 to actual pixels
    full_manifest = generate_full_tracking(video_path, sparse_manifest)

    # 3. SAVE TEMPORARY FULL MANIFEST (Optional, but good for debugging)
    with open("full_rendered_manifest.json", "w") as f:
        json.dump(full_manifest, f)

    print(f"✅ Interpolation complete. Ready to blur {len(full_manifest)} frames.")

    # 4. EXECUTE GPU REDACTION
    # We pass the NEW full_manifest to your existing CUDA script
    print("🚀 Starting 1080 Ti Redaction Process...")
    process_video_cuda(video_path, output_path, full_manifest)

    print(f"✨ Success! Redacted video saved to: {output_path}")

if __name__ == "__main__":
    main()