import os
import subprocess
import json

# --- CONFIGURATION ---
INPUT_DIR = 'samples/sanitized'
OUTPUT_BASE_DIR = 'keyframes'
FRAME_RATE = 1  # Extract 1 frame per second of video
# ---------------------

def get_video_resolution(video_path):
    """Uses ffprobe to get the width and height of the video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        return width, height
    except Exception as e:
        print(f"⚠️ Warning: Could not get resolution for {video_path}. Error: {e}")
        return 1280, 720  # Fallback to 720p if probe fails

def extract_scenes():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory '{INPUT_DIR}' not found.")
        return

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp4', '.mkv', '.avi'))]

    if not video_files:
        print(f"📁 No sanitized videos found in {INPUT_DIR}.")
        return

    for video_file in video_files:
        video_path = os.path.join(INPUT_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # --- PART 1: RESOLUTION-AWARE LOGIC ---
        width, height = get_video_resolution(video_path)
        
        # Create folder name with resolution suffix
        output_folder = os.path.join(OUTPUT_BASE_DIR, f"{video_name}_{width}x{height}")
        os.makedirs(output_folder, exist_ok=True)

        # Save metadata for downstream scripts (Verification & Redaction)
        metadata = {
            "video_name": video_name,
            "width": width,
            "height": height,
            "fps_extracted": FRAME_RATE
        }
        with open(os.path.join(output_folder, "details.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"🎬 Processing: {video_name} ({width}x{height})")
        
        # FFmpeg command to extract frames
        # %03d.jpg creates 001.jpg, 002.jpg, etc.
        output_pattern = os.path.join(output_folder, "%03d.jpg")
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={FRAME_RATE}",
            "-q:v", "2", # High quality JPEGs
            output_pattern,
            "-y" # Overwrite existing
        ]

        try:
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
            print(f"✅ Extracted keyframes to: {output_folder}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed on {video_name}: {e}")

if __name__ == "__main__":
    extract_scenes()