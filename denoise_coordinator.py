import subprocess
import os

# Update this path to your specific denoising venv
DENOISE_PYTHON = "/home/piyush/.virtualenvs/YOUR_DENOISE_VENV/bin/python"

def process_audio_isolated(video_path):
    raw_wav = "temp_raw.wav"
    clean_wav = "temp_clean.wav"
    
    # Step 1: Extract high-fidelity audio for DeepFilterNet (48kHz)
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path, 
        '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', 
        raw_wav
    ], check=True, capture_output=True)

    # Step 2: Call the OTHER environment to denoise
    print(f"🧹 Launching isolated denoising for {os.path.basename(video_path)}...")
    subprocess.run([
        DENOISE_PYTHON, "denoise_worker.py", raw_wav, clean_wav
    ], check=True)

    os.remove(raw_wav)
    return clean_wav