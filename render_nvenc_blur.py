"""
FILE: render_nvenc_blur.py
ROLE: GPU-Accelerated Video Rendering (NVENC)
-------------------------------------------------------------------------
DESCRIPTION:
An optimized rendering engine that uses the 1080 Ti's hardware H.264 
encoder (NVENC). It reads a UI manifest and applies blurs while 
streaming the output directly through FFmpeg for maximum throughput.

HARDWARE COMPATIBILITY:
- Optimized for NVIDIA 'Pascal' architecture (1080 Ti). 
- Uses 'p6' preset for high-quality archival output.
-------------------------------------------------------------------------
"""

import subprocess
import json
import cv2
import numpy as np

def render_final_video(video_in, video_out, tracking_json, manifest_json):
    with open(tracking_json, "r") as f:
        tracking = json.load(f)
    with open(manifest_json, "r") as f:
        manifest = json.load(f)

    # Determine which IDs the user selected to blur
    ids_to_blur = [
        int(eid) for eid, data in manifest["entities"].items() 
        if data.get("user_confirmed_blur", data.get("blur_recommended"))
    ]

    # Reformat tracks by frame for fast O(1) lookup during rendering
    frame_actions = {}
    for eid in ids_to_blur:
        for point in tracking["entities"][str(eid)]["trajectory"]:
            f_num = point["frame"]
            if f_num not in frame_actions:
                frame_actions[f_num] = []
            frame_actions[f_num].append(point["bbox"])

    cap = cv2.VideoCapture(video_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # FFmpeg pipe: NVDEC (Decode) -> Python -> NVENC (Encode)
    cmd_out = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-', # Input from stdin
        '-c:v', 'h264_nvenc', # GPU Encoder
        '-preset', 'p6',
        '-tune', 'hq',
        '-b:v', '10M',
        '-pix_fmt', 'yuv420p',
        video_out
    ]

    process_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(video_in)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frame_actions:
            for (x1, y1, x2, y2) in frame_actions[frame_idx]:
                # Optimized sub-region Gaussian Blur
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
                    
        process_out.stdin.write(frame.tobytes())
        frame_idx += 1

    cap.release()
    process_out.stdin.close()
    process_out.wait()
    print(f"[SUCCESS] Hardware render complete: {video_out}")