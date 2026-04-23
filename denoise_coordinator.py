"""
FILE: denoise_coordinator.py
ROLE: Audio Extraction & Isolation Manager (Phase 0)
-------------------------------------------------------------------------
DESCRIPTION:
Acts as the entry point for audio preprocessing. It extracts 48kHz mono 
audio from source videos and hands it off to an isolated 'worker' 
environment for denoising.

INPUT: Source video path (.mp4, .mkv).
OUTPUT: temp_clean.wav (Denoised audio ready for transcription).
-------------------------------------------------------------------------
"""

import subprocess
import os

# Update this path to your specific denoising venv
DENOISE_PYTHON = "/home/piyush/.virtualenvs/ai-video-v2/bin/python"

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