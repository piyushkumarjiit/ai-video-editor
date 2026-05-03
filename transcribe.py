"""
FILE: transcribe.py
ROLE: Phase 1+2 — Transcription & Word-Level Alignment
-------------------------------------------------------------------------
DESCRIPTION:
Loads denoised 16kHz WAV files and runs WhisperX transcription followed
by word-level timestamp alignment. Outputs an intermediate JSON used
by diarize.py in the next phase.

INPUT:
- samples/denoised/{video_name}_final_16k.wav

OUTPUT:
- tracking_results/{video_name}_transcript.json

IDEMPOTENT:
- Skips files where transcript JSON already exists.

HARDWARE COMPATIBILITY:
- Uses float32 for GTX 1080 Ti stability.
-------------------------------------------------------------------------
"""

import os
import sys
import time
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import argparse
import torch
import whisperx
from safe_globals import register_omegaconf_only, patch_hf_hub
from pipeline_utils import setup_logging, video_output_dir, OUTPUT_DIR, wait_for_gpu, \
 get_gpu_requirements,MAX_RETRIES

# ── HuggingFace Hub Patch and weights only  ────────────────────────────────────
register_omegaconf_only()
patch_hf_hub()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
COMPUTE_TYPE = "float32"   # float32 for GTX 1080 Ti stability
DEFAULT_LANGUAGE="en"
BATCH_SIZE = 8


#LOG_DIR = "logs"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging Configuration ─────────────────────────────────────────────────────

setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ── Helpers ───────────────────────────────────────────────────────────────────
def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def transcript_path_for(base_name: str) -> str:
    return os.path.join(video_output_dir(base_name), "_transcript.json")


# def wait_for_gpu(required_mb, max_retries=MAX_RETRIES):
#     """
#     Waits for a specific amount of VRAM to be free. 
#     MAX_RETRIES retries at 15s intervals = 5 minutes of waiting.
#     """
#     if not torch.cuda.is_available():
#         return
    
#     required_mb = int(required_mb)
#     retries = 0
    
#     while retries < max_retries:
#         free_bytes, _ = torch.cuda.mem_get_info()
#         os_free_mb = free_bytes / (1024**2)
        
#         reserved_mb = torch.cuda.memory_reserved() / (1024**2)
#         allocated_mb = torch.cuda.memory_allocated() / (1024**2)
#         pytorch_actual_free = os_free_mb + (reserved_mb - allocated_mb)

#         if pytorch_actual_free >= required_mb:
#             logging.info(f"✅ GPU VRAM Sufficient: {pytorch_actual_free:.0f}MB effectively free.")
#             return # Success
            
#         retries += 1
#         logging.info(f"⏳ GPU Busy ({retries}/{max_retries}): {pytorch_actual_free:.0f}MB free. Waiting...")
#         time.sleep(15)

#     raise RuntimeError(f"❌ GPU timeout: Could not reclaim {required_mb}MB after {max_retries} attempts.")


def transcribe(audio_path, batch_size, model) -> dict:
    """
    Runs WhisperX transcription + word-level alignment on a WAV file.
    """
    result = None
    try:
        logging.info(f"📝 Phase 1: Transcribing {os.path.basename(audio_path)}...")
        audio_data = whisperx.load_audio(audio_path)
        result = model.transcribe(audio_data, batch_size=batch_size)
        lang = result.get("language", DEFAULT_LANGUAGE)
    except Exception as e:
        # Use audio_path here to avoid NameError if base_name isn't global
        logging.error(f"❌ Transcription failed for {os.path.basename(audio_path)}: {e}", exc_info=True)
        return None

    try:
        logging.info(f"🔗 Phase 2: Aligning word-level timestamps ({lang})...")
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, DEVICE)
        # Explicitly delete alignment model to free VRAM
        del model_a
    except Exception as e:
        logging.warning(f"⚠️ Alignment failed: {e}. Returning unaligned segments.")
    finally:
        flush_gpu()

    return result


def save_transcript(result: dict, base_name: str):
    """Atomically writes transcript JSON to avoid partial-write corruption."""
    out_path = transcript_path_for(base_name)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    os.replace(tmp_path, out_path)
    logging.info(f"✅ Transcript saved → {out_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 1+2: Transcription & Alignment")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="WhisperX batch size (default: 4)")
    parser.add_argument("--video", type=str, default=None, help="Process a single video file (e.g. MyVideo.mp4) instead of directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    
    if args.video:
        # 1. Extract the base name (e.g., 'Karen_Fights_...')
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        
        # 2. Point to the structured tracking_results folder
        video_dir = video_output_dir(base_name)
        
        # 3. Fix the path to look for the file denoise.py just created[cite: 2, 4]
        wav = os.path.join(video_dir, "_final_16k.wav")
        
        targets = [(base_name, wav)] if os.path.exists(wav) else []
        if not targets:
            logging.error(f"❌ Denoised WAV not found for: {args.video} — run denoise.py first.")
    else:
        targets = []
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_dir():
                wav = os.path.join(entry.path, "_final_16k.wav")
                if os.path.exists(wav):
                    targets.append((entry.name, wav))

    # Check if enough VRAM is available to load the model
    try:
        script_name = os.path.basename(__file__)
        wait_for_gpu(get_gpu_requirements(script_name), MAX_RETRIES)
        # LOADING MODEL ONCE (Moved out of loop for VRAM stability)
        logging.info(f"🚀 Loading WhisperX 'medium' model on {DEVICE}...")
        model = whisperx.load_model("medium", DEVICE, compute_type=COMPUTE_TYPE)
    except RuntimeError as e:
        logging.error(f"💀 Pipeline Aborted: {e}")
        sys.exit(1) # Exit with error code for the orchestrator

    processed = skipped = failed = 0
    targets = sorted(targets)
    for base_name, wav_path in targets:
        out = transcript_path_for(base_name)

        if os.path.exists(out) and not args.force:
            logging.info(f"⏭️  Skipping {base_name} — transcript already exists.")
            skipped += 1
            continue

        try:
            #result = transcribe(wav_path, args.batch_size)
            result = transcribe(wav_path, args.batch_size, model)
            if result and result.get("segments"):
                save_transcript(result, base_name)
                processed += 1
            else:
                logging.warning(f"⚠️  No speech segments found for {base_name}.")
                failed += 1
        except Exception as e:
            logging.error(f"❌ Failed on {base_name}: {e}", exc_info=True)
            failed += 1
        finally:
            flush_gpu() # Cleanup after every file
    # Final Cleanup before exiting so Diarize phase can start
    del model
    flush_gpu()
    logging.info("🏁 GPU memory released for next phase.")
    logging.info(f"✅ Transcription complete — processed: {processed}, skipped: {skipped}, failed: {failed}")

if __name__ == "__main__":
    main()