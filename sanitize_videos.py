import os
import subprocess

SOURCE_DIR = 'samples'
DEST_DIR = os.path.join(SOURCE_DIR, 'sanitized')

def check_gpu():
    """Returns True if an NVIDIA GPU with NVENC support is detected."""
    try:
        # Check if nvidia-smi exists and returns successfully
        subprocess.run(['nvidia-smi'], check=True, capture_output=True)
        # Check if ffmpeg actually supports nvenc
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def sanitize_pipeline():
    os.makedirs(DEST_DIR, exist_ok=True)
    has_gpu = check_gpu()
    
    if has_gpu:
        print("🚀 NVIDIA GPU detected! Using NVENC for high-speed sanitization.")
    else:
        print("🐌 No GPU detected or NVENC missing. Falling back to CPU (libx264).")

    video_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]

    for video in video_files:
        input_path = os.path.join(SOURCE_DIR, video)
        output_path = os.path.join(DEST_DIR, video)
        
        if os.path.exists(output_path):
            continue

        print(f"🛠️ Processing: {video}...")

        # Base command
        cmd = ['ffmpeg', '-y', '-i', input_path]

        if has_gpu:
            # GPU Logic (Fast)
            cmd += [
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-cq', '22'
            ]
        else:
            # CPU Logic (Standard)
            cmd += [
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '22'
            ]

        # Common flags for both
        cmd += ['-c:a', 'copy', '-movflags', '+faststart', output_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Finished {video}")
        else:
            print(f"❌ Error on {video}: {result.stderr}")

if __name__ == "__main__":
    sanitize_pipeline()