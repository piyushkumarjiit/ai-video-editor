"""
FILE: render_production.py
ROLE: Master Assembly & Hardware-Accelerated Rendering
-------------------------------------------------------------------------
DESCRIPTION:
The final 'Glue' script. It consumes:
1. tracking.json (The physical 'where' - trajectories)
2. redaction_manifest.json (The user 'who' - selection from UI)
3. *_mute_manifest.json (The audio 'when' - PII timestamps)

It outputs a final, redacted MP4 using NVENC for speed.
-------------------------------------------------------------------------
"""

import subprocess
import json
import cv2
import os
import numpy as np
from tqdm import tqdm

def render_final_output(video_in, video_out, tracking_json, selection_json, audio_manifest=None):
    # 1. Load All Intelligence Data
    with open(tracking_json, "r") as f:
        tracking_data = json.load(f)
    with open(selection_json, "r") as f:
        selection_data = json.load(f)
    
    # Identify which IDs the user actually wants to redact
    ids_to_blur = [str(eid) for eid in selection_data.get("selected_for_redaction", [])]
    
    # Re-index trajectories for O(1) frame-by-frame lookup
    frame_map = {}
    for eid in ids_to_blur:
        if eid in tracking_data["entities"]:
            for point in tracking_data["entities"][eid]["trajectory"]:
                f_idx = point["frame"]
                if f_idx not in frame_map: frame_map[f_idx] = []
                frame_map[f_idx].append(point["bbox"])

    # 2. Setup Video Capture and FFmpeg Pipe
    cap = cv2.VideoCapture(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Base FFmpeg command for NVENC
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',  # Input from Python stdin
        '-i', video_in, # Second input for audio mapping
    ]

    # 3. Build Audio Filtergraph (PII Muting)
    # This creates a "volume=0" filter for each PII segment
    audio_filter = "anullsrc=channel_layout=stereo:sample_rate=44100[silent];"
    if audio_manifest and os.path.exists(audio_manifest):
        with open(audio_manifest, "r") as f:
            mutes = json.load(f)
        
        # Build volume ducking string: volume=0:enable='between(t,start,end)'
        v_filters = []
        for segment in mutes.get("mute_segments", []):
            v_filters.append(f"volume=0:enable='between(t,{segment['start']},{segment['end']})'")
        
        af_chain = ",".join(v_filters) if v_filters else "volume=1"
        cmd += ['-filter_complex', f'[1:a]{af_chain}[outa]', '-map', '0:v', '-map', '[outa]']
    else:
        cmd += ['-map', '0:v', '-map', '1:a']

    # Final encoder settings for 1080 Ti
    cmd += ['-c:v', 'h264_nvenc', '-preset', 'p6', '-tune', 'hq', '-b:v', '8M', video_out]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # 4. Main Rendering Loop
    print(f"🎬 Rendering Redacted Video: {video_out}")
    pbar = tqdm(total=total_frames)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Apply Visual Blurs based on Tracking Data
        if frame_idx in frame_map:
            for (x1, y1, x2, y2) in frame_map[frame_idx]:
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Legal standard: Gaussian blur with high sigma
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blurred_roi

        # Send frame to FFmpeg pipe
        process.stdin.write(frame.tobytes())
        frame_idx += 1
        pbar.update(1)

    # Cleanup
    pbar.close()
    cap.release()
    process.stdin.close()
    process.wait()
    print(f"✅ Success! MVP Output generated at: {video_out}")

if __name__ == "__main__":
    # Example paths - replace with your actual file names
    render_final_output(
        video_in="samples/sanitized/test_video.mp4",
        video_out="output/REDACTED_FINAL.mp4",
        tracking_json="tracking.json",
        selection_json="redaction_manifest.json",
        audio_manifest="transcripts/test_video_mute_manifest.json"
    )