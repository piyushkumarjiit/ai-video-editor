"""
FILE: verify_redaction.py
ROLE: Final Redaction Safety Check (Visual Preview)
-------------------------------------------------------------------------
DESCRIPTION:
The final visual verification script before actual video blurring. It 
produces high-visibility previews (thick borders/large text) to allow 
the user to manually confirm that every sensitive target (Face, License) 
has a correctly sized box for redaction.

INPUT: 
- tracked_manifest.json (Latest version after normalization).
- keyframes/{video_name}/details.json.

OUTPUT:
- verify_{video_name}.jpg: High-visibility preview image for human sign-off.

PIPELINE STAGE: 
- Final QA. This is the last script run before the 'apply_blur' stage 
  to prevent privacy leaks.
-------------------------------------------------------------------------
"""

import cv2
import json
import os
import glob

def run_verification():
    # 1. Find the latest tracked manifest
    manifest_path = 'tracked_manifest.json'
    if not os.path.exists(manifest_path):
        print(f"❌ Error: {manifest_path} not found. Run normalization/tracking first.")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # 2. Iterate through videos in the manifest
    for video_name, frames in manifest.items():
        # Find the matching folder (e.g., "VideoName_1920x1080")
        search_pattern = f"keyframes/{video_name}_*"
        matching_folders = glob.glob(search_pattern)

        if not matching_folders:
            print(f"⚠️ Could not find keyframe folder for {video_name}")
            continue

        folder_path = matching_folders[0]
        
        # Load resolution from our new details.json
        with open(f"{folder_path}/details.json", 'r') as f:
            details = json.load(f)
            W, H = details['width'], details['height']

        # Pick the first frame that actually has a detection
        test_frame = next((f for f in frames if f['targets']), None)
        
        if test_frame:
            img_path = f"{folder_path}/{test_frame['frame']}"
            img = cv2.imread(img_path)
            
            for target in test_frame['targets']:
                # AI gives [ymin, xmin, ymax, xmax] (0-1000)
                ymin, xmin, ymax, xmax = target['bbox']
                
                # Convert to Pixels
                start = (int(xmin * W / 1000), int(ymin * H / 1000))
                end = (int(xmax * W / 1000), int(ymax * H / 1000))
                
                # Draw Box + Label
                cv2.rectangle(img, start, end, (0, 255, 0), 3) # Green for verified
                cv2.putText(img, target['id'], (start[0], start[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_name = f"verify_{video_name}.jpg"
            cv2.imwrite(output_name, img)
            print(f"✅ Verification image created: {output_name}")

if __name__ == "__main__":
    run_verification()