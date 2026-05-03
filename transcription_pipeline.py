# NOT USED as existing transcribe_diarize runs just fine.

import os
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_DIR = Path("samples/sanitized")
DENOISED_DIR = Path("samples/denoised")
OUTPUT_DIR = Path("transcripts")

# CLEANUP SETTINGS
DELETE_RAW_FFMPEG = True     # Deletes the initial audio extract
DELETE_DENOISED_WAV = True    # Deletes the enhanced audio after processing

# PYTHON PATHS
PYTHON_DENOISE = "/home/pk/.virtualenvs/ai-video-denoise/bin/python"
PYTHON_ASR = "/home/pk/.virtualenvs/ai-video-asr/bin/python"
PYTHON_DIARIZE = "/home/pk/.virtualenvs/ai-video-diarize/bin/python"

def process_video(video_path):
    base_name = video_path.stem
    raw_wav = VIDEO_DIR / f"{base_name}_raw.wav"
    
    # 1. FFMPEG Extraction
    print(f"🔊 Extracting audio...")
    subprocess.run(['ffmpeg', '-y', '-i', str(video_path), '-vn', '-ar', '48000', '-ac', '1', str(raw_wav)], check=True, capture_output=True)

    # 2. Denoising
    print(f"🧹 Denoising...")
    # Create an environment copy and hide the GPU
    env = os.environ.copy()
    # Forcing the code to use CPU as 1080TI has issues with DNN
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([PYTHON_DENOISE, "-m", "df.enhance", str(raw_wav), "--output-dir", str(DENOISED_DIR)], env=env, check=True)
    
    # DeepFilterNet naming convention
    denoised_wav = DENOISED_DIR / f"{base_name}_raw_DeepFilterNet3.wav"

    try:
        # 3. ASR
        print(f"📝 Transcribing...")
        subprocess.run([PYTHON_ASR, "asr_worker.py", str(denoised_wav), str(OUTPUT_DIR / f"{base_name}_segments.json")], check=True)

        # 4. Diarization
        print(f"🧬 Diarizing...")
        subprocess.run([PYTHON_DIARIZE, "diarize_worker.py", str(denoised_wav), str(OUTPUT_DIR / f"{base_name}_diarize.json")], check=True)

    finally:
        # --- CLEANUP SECTION ---
        print("🧹 Running disk cleanup...")
        
        if DELETE_RAW_FFMPEG and raw_wav.exists():
            os.remove(raw_wav)
            print(f"🗑️ Deleted raw extract: {raw_wav.name}")

        if DELETE_DENOISED_WAV and denoised_wav.exists():
            os.remove(denoised_wav)
            print(f"🗑️ Deleted denoised file: {denoised_wav.name}")

if __name__ == "__main__":
    for video in VIDEO_DIR.glob("*.mp4"):
        process_video(video)