import cv2
import json
import os
import glob

def verify_grounding():
    manifest_path = 'tracked_manifest.json'
    if not os.path.exists(manifest_path):
        print("❌ Error: tracked_manifest.json not found.")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    for video_name, frames in manifest.items():
        # Find the folder with the resolution suffix
        search_pattern = f"keyframes/{video_name}_*"
        matching_folders = glob.glob(search_pattern)
        
        if not matching_folders:
            print(f"⚠️ Folder for {video_name} not found.")
            continue
            
        folder_path = matching_folders[0]
        
        # Load the resolution from the details.json we added to detect_scenes.py
        with open(f"{folder_path}/details.json", 'r') as f:
            details = json.load(f)
            W, H = details['width'], details['height']

        # Pick the first frame with detections
        target_frame = next((f for f in frames if f['targets']), None)
        
        if target_frame:
            img_path = os.path.join(folder_path, target_frame['frame'])
            img = cv2.imread(img_path)
            
            for target in target_frame['targets']:
                ymin, xmin, ymax, xmax = target['bbox']
                
                # Convert normalized (0-1000) to actual pixels
                start_point = (int(xmin * W / 1000), int(ymin * H / 1000))
                end_point = (int(xmax * W / 1000), int(ymax * H / 1000))
                
                # Draw Box (Green) and Label
                cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(img, target['id'], (start_point[0], start_point[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_file = f"verify_{video_name}.jpg"
            cv2.imwrite(output_file, img)
            print(f"✅ Created verification image: {output_file}")

if __name__ == "__main__":
    verify_grounding()