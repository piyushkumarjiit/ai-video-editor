import cv2
import json

def apply_tracked_blur(video_path, manifest_path, output_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    trackers = {} # Format: {entity_id: cv2.Tracker}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        
        # 1. Update existing trackers
        for ent_id, tracker in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                roi = frame[y:y+h, x:x+w]
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51, 51), 0)

        # 2. Initialize new trackers if frame is a VLM Keyframe
        if frame_idx in manifest.get('key_frames', {}):
            for entity in manifest['key_frames'][frame_idx]['entities']:
                ent_id = entity['id']
                if ent_id in manifest.get('selected_for_redaction', []):
                    # Convert [0-1000] grounded coords to pixel coords
                    gx, gy, gw, gh = entity['bbox']
                    px, py = int(gx * width / 1000), int(gy * height / 1000)
                    pw, ph = int(gw * width / 1000), int(gh * height / 1000)
                    
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (px, py, pw, ph))
                    trackers[ent_id] = tracker
                    
        out.write(frame)
        
    cap.release()
    out.release()