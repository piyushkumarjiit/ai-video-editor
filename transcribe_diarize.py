"""
FILE: transcribe_diarize.py
ROLE: Stable Multimodal Transcription for Legal/Police SaaS
-------------------------------------------------------------------------
DESCRIPTION:
A high-speed transcription fallback that uses Faster-Whisper. It provides 
a quick text dump of a video's audio without the overhead of 
denoising or speaker diarization.

INPUT: 
- Video files in 'samples/sanitized/'.

OUTPUT:
- tracking_results/{video_name}_metadata.json: Full word-level JSON.
- tracking_results/{video_name}_clean.txt: Clean speaker-labeled text.

HARDWARE COMPATIBILITY:
- Uses float32 for GTX 1080 Ti stability.
-------------------------------------------------------------------------
"""

import os
import sys
import gc
import torch
import whisperx
import subprocess
import json
import time
import logging
import argparse
from dotenv import load_dotenv
import huggingface_hub
from huggingface_hub import hf_hub_download as real_download
from safe_globals import register_omegaconf_only
register_omegaconf_only()

# torch.serialization.add_safe_globals([
#     omegaconf.listconfig.ListConfig,
#     omegaconf.dictconfig.DictConfig,
#     omegaconf.base.ContainerMetadata,
# ])

# Patch for older HuggingFace hub versions
def patched_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return real_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- CONFIGURATION ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda"
COMPUTE_TYPE = "float32" 
VIDEO_DIR = "samples/sanitized"
DENOISED_DIR = "samples/denoised"
OUTPUT_DIR = "tracking_results"
DENOISE_PYTHON = sys.executable
SUPPORTED_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.mpeg', '.mpg')

# Global toggle - controlled via CLI
DELETE_DENOISED_FILES = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DENOISED_DIR, exist_ok=True)

def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def is_file_stable(filepath, wait_time=5):
    """Checks if a file size is stable (avoids processing partial uploads)."""
    try:
        first_size = os.path.getsize(filepath)
        time.sleep(wait_time)
        second_size = os.path.getsize(filepath)
        return first_size == second_size and first_size > 0
    except Exception:
        return False

def save_outputs_atomic(result, video_name):
    """Saves metadata and text files using atomic replacement to prevent corruption."""
    base_name = os.path.splitext(video_name)[0]
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}_metadata.json")
    tmp_json_path = json_path + ".tmp"
    
    # Atomic JSON Save
    with open(tmp_json_path, "w") as f:
        json.dump(result, f, indent=4)
    os.replace(tmp_json_path, json_path)

    # Clean Text Export
    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}_clean.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            f.write(f"[{speaker}]: {text}\n")
            
    logging.info(f"✅ Outputs saved atomically for {base_name}")

def get_video_codec(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name', '-of', 'csv=p=0',
        video_path
    ]
    try:
        return subprocess.check_output(cmd).decode('utf-8').strip()
    except Exception:
        return None

def run_isolated_denoise(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_wav = os.path.join(VIDEO_DIR, f"{base_name}_raw_tmp.wav")
    codec = get_video_codec(video_path)
    
    decoder_map = {
        'h264': 'h264_cuvid', 'hevc': 'hevc_cuvid', 'mjpeg': 'mjpeg_cuvid',
        'vp8': 'vp8_cuvid', 'vp9': 'vp9_cuvid'
    }
    hw_decoder = decoder_map.get(codec)
    ffmpeg_cmd = ['ffmpeg', '-y', '-hwaccel', 'cuda']

    if hw_decoder:
        ffmpeg_cmd.extend(['-c:v', hw_decoder])
        logging.info(f"🚀 HW Decoder: {hw_decoder}")
    
    ffmpeg_cmd.extend([
        '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '48000', '-ac', '1', raw_wav
    ])
    
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" 
    
    try:
        logging.info(f"🧹 Phase 0: Denoising {base_name}...")
        subprocess.run([
            DENOISE_PYTHON, "-m", "df.enhance", 
            raw_wav, "--output-dir", DENOISED_DIR
        ], env=env, check=True) 

        enhanced_file = None
        for file in os.listdir(DENOISED_DIR):
            if base_name in file and "DeepFilterNet" in file:
                enhanced_file = os.path.join(DENOISED_DIR, file)
                break
        
        if enhanced_file:
            final_16k_wav = os.path.join(DENOISED_DIR, f"{base_name}_final_16k.wav")
            subprocess.run([
                'ffmpeg', '-y', '-i', enhanced_file, 
                '-ar', '16000', '-ac', '1', final_16k_wav
            ], check=True, capture_output=True)
            
            if os.path.exists(raw_wav): os.remove(raw_wav)
            if os.path.exists(enhanced_file): os.remove(enhanced_file)
            return final_16k_wav
            
        return raw_wav 
    except Exception as e:
        logging.error(f"⚠️ Denoising failed: {e}")
        return raw_wav

def process_video(video_path, BATCH_SIZE):
    video_file = os.path.basename(video_path)
    audio_file = run_isolated_denoise(video_path)
    
    try:
        # Phase 1: Transcribe
        logging.info(f"📝 Phase 1: Transcribing {video_file}...")
        model = whisperx.load_model("medium", DEVICE, compute_type=COMPUTE_TYPE)
        audio_data = whisperx.load_audio(audio_file)
        result = model.transcribe(audio_data, batch_size=BATCH_SIZE)
        
        # Phase 2: Align
        logging.info("🔗 Phase 2: Aligning word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, DEVICE)
        
        del model, model_a
        flush_gpu()

        # Phase 3: Diarize
        logging.info("🧬 Phase 3: Diarizing speakers...")
        #diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio_data)
        
        del diarize_model
        flush_gpu()

        # Phase 4: Assign Speakers
        logging.info("🏷️ Phase 4: Finalizing labels...")
        #result = whisperx.assign_word_speakers(diarize_segments, result)
        result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
        return result

    finally:
        if os.path.exists(audio_file) and DELETE_DENOISED_FILES:
            os.remove(audio_file)
            logging.info(f"🗑️ Deleted temporary audio: {os.path.basename(audio_file)}")

def main():
    global DELETE_DENOISED_FILES
    
    parser = argparse.ArgumentParser(description="Multimodal Transcription & Diarization Pipeline")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for WhisperX")
    parser.add_argument("--delete_tmp", action="store_true", help="Cleanup intermediate wavs")
    args = parser.parse_args()

    DELETE_DENOISED_FILES = args.delete_tmp

    # Use os.scandir for an iterative, memory-efficient directory crawl
    processed_count = 0
    logging.info(f"🔍 Monitoring {VIDEO_DIR} for new files...")

    with os.scandir(VIDEO_DIR) as entries:
        for entry in entries:
            # 1. Immediate Filter: Skip directories and unsupported extensions
            if not entry.is_file() or not entry.name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            video_path = entry.path
            base_name = os.path.splitext(entry.name)[0]
            output_check = os.path.join(OUTPUT_DIR, f"{base_name}_metadata.json")
            
            # 2. Check if already processed
            if os.path.exists(output_check):
                continue

            # 3. Perform stability check on THIS file only
            logging.info(f"🧐 Found: {entry.name}. Checking stability...")
            if not is_file_stable(video_path):
                logging.warning(f"⚠️ Skipping {entry.name}: File is still being written.")
                continue

            # 4. Immediate execution
            try:
                logging.info(f"🚀 Starting Pipeline: {entry.name}")
                
                # Sequence: CPU Denoise -> GPU Transcribe -> GPU Diarize
                final_result = process_video(video_path, args.batch_size)
                
                if final_result:
                    save_outputs_atomic(final_result, entry.name)
                    logging.info(f"✨ Successfully processed {entry.name}")
                    processed_count += 1
                
                flush_gpu()
                
            except Exception as e:
                logging.error(f"❌ Critical failure on {entry.name}: {e}", exc_info=True)
                # Ensure GPU memory is cleared even after a failure
                flush_gpu()
                continue 

    if processed_count == 0:
        logging.info("☕ No new stable files were processed. Standby.")
    else:
        logging.info(f"✅ Finished processing {processed_count} files.")

if __name__ == "__main__":
    main()