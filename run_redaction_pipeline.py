import json
import os
import sys
import cv2
from tracker_interpolator import generate_full_tracking
from apply_redaction import process_video_cuda

def main(video_input_path, ui_manifest_path, output_path):
    # 1. Load the User Selections (from the UI stage)
    print(f"[1/3] Loading UI manifest: {ui_manifest_path}")
    with open(ui_manifest_path, 'r') as f:
        redaction_data = json.load(f)
    
    # 2. Generate Frame-by-Frame Tracking
    # This fills the gaps between AI-sampled keyframes
    print(f"[2/3] Interpolating tracks for smooth redaction...")
    tracked_manifest = generate_full_tracking(video_input_path, redaction_data)
    
    # Save a temporary manifest for debugging/verification
    temp_manifest = "tracked_manifest_internal.json"
    with open(temp_manifest, 'w') as f:
        json.dump(tracked_manifest, f, indent=4)
    print(f"-> Intermediate tracked manifest saved to {temp_manifest}")

    # 3. Execute CUDA-Accelerated Rendering
    print(f"[3/3] Starting CUDA render: {output_path}")
    if not os.path.exists(video_input_path):
        print(f"Error: Input video {video_input_path} not found.")
        return

    try:
        process_video_cuda(video_input_path, output_path, tracked_manifest)
        print(f"🎉 Success! Redacted video saved to: {output_path}")
    except Exception as e:
        print(f"❌ Redaction failed: {str(e)}")
        print("Tip: Ensure your OpenCV is compiled with CUDA and NVIDIA drivers are active.")

if __name__ == "__main__":
    # Example usage: python run_redaction_pipeline.py input.mp4 ui_manifest.json output_redacted.mp4
    if len(sys.argv) < 4:
        print("Usage: python run_redaction_pipeline.py <input_video> <ui_manifest> <output_video>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])