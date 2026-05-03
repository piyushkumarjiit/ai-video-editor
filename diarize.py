"""
FILE: diarize.py
ROLE: Phase 3+4 — Speaker Diarization & Final Output
-------------------------------------------------------------------------
DESCRIPTION:
Loads aligned transcripts and runs Pyannote speaker diarization.
Assigns speaker labels to each word/segment and writes the final
metadata JSON and clean speaker-labeled text file.

INPUT:
- samples/denoised/{video_name}_final_16k.wav
- tracking_results/{video_name}_transcript.json

OUTPUT:
- tracking_results/{video_name}_metadata.json  (full word-level JSON)
- tracking_results/{video_name}_clean.txt       (speaker-labeled text)

IDEMPOTENT:
- Skips files where metadata JSON already exists.
-------------------------------------------------------------------------
"""

import os
import sys
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import argparse
import torch
import whisperx
import pyannote.audio.core.pipeline
from dotenv import load_dotenv
from safe_globals import register_omegaconf_only, patch_hf_hub
from pipeline_utils import setup_logging, video_output_dir, OUTPUT_DIR, wait_for_gpu, \
 get_gpu_requirements,MAX_RETRIES

# ── HuggingFace Hub Patch and weights only  ────────────────────────────────────
register_omegaconf_only()
patch_hf_hub()

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN     = os.getenv("HF_TOKEN")
DEVICE       = "cuda"
DEVICE       = "cuda"
COMPUTE_TYPE = "float32"   # float32 for GTX 1080 Ti stability
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging Configuration ─────────────────────────────────────────────────────

setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ── Helpers ───────────────────────────────────────────────────────────────────
def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def metadata_path_for(base_name: str) -> str:
    return os.path.join(video_output_dir(base_name), "_metadata.json")

def transcript_path_for(base_name: str) -> str:
    return os.path.join(video_output_dir(base_name), "_transcript.json")

def save_outputs(result: dict, base_name: str):
    """Atomically writes metadata JSON and clean speaker-labeled text."""
    # ── Metadata JSON ─────────────────────────────────────────────────────
    meta_path = metadata_path_for(base_name)
    tmp_path  = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    os.replace(tmp_path, meta_path)
    logging.info(f"✅ Metadata saved → {meta_path}")

    # ── Clean text export ─────────────────────────────────────────────────
    txt_path = os.path.join(video_output_dir(base_name), "_clean.txt")
    segments = result.get("segments", [])
    if not segments:
        logging.warning(f"⚠️ No segments found for {base_name}. Clean text will be empty.")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            if text:
                f.write(f"[{speaker}]: {text}\n")
    logging.info(f"✅ Clean text saved → {txt_path}")


# def video_output_dir(base_name: str) -> str:
#     """Returns the per-video output directory, creating it if needed."""
#     path = os.path.join(OUTPUT_DIR, base_name)
#     os.makedirs(path, exist_ok=True)
#     return path


def diarize(base_name: str, diarize_model, min_speakers=None, max_speakers=None ) -> dict:
    """
    Runs Pyannote diarization and assigns speaker labels to transcript segments.
    Returns the final labeled result dict.
    """
    out_dir         = video_output_dir(base_name)
    wav_path        = os.path.join(out_dir, "_final_16k.wav")
    transcript_path = os.path.join(out_dir, "_transcript.json")
    result = None    

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path} — run denoise.py first.")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path} — run transcribe.py first.")

    try:
        logging.info(f"📂 Loading transcript JSON for {base_name}...")
        with open(transcript_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        logging.info(f"🎙️ Loading WAV file {base_name}...")
        audio_data = whisperx.load_audio(wav_path)
    except Exception as e:
        # Use audio_path here to avoid NameError if base_name isn't global
        logging.error(f"❌ Transcript JSON or WAV loading failed for {base_name}: {e}", exc_info=True)
        return None

    # Phase 3: Diarize
    # Note: whisperx.DiarizationPipeline was moved to whisperx.diarize in newer versions of whisperx. Use the submodule directly.
    try:
        logging.info(f"🧬 Phase 3: Diarizing {base_name}...")
        #diarize_model    = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as e:
        logging.warning(f"⚠️ Diarization failed: {e}. Returning segments without speakers.")
    finally:
        flush_gpu()
    
    # Phase 4: Assign speakers
    if diarize_segments is not None:
        try:
            logging.info("🏷️  Phase 4: Assigning speaker labels...")
            result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            logging.warning(f"⚠️ Speaker assigninment failed: {e}. Returning unlabelled segments.")
        finally:
            flush_gpu()
    else:
        logging.warning("⏩ Skipping Speaker Assignment: No diarization data available.")
    return result

# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 3+4: Diarization & Speaker Assignment")
    parser.add_argument("--video", type=str, default=None, help="Process a single video file (e.g. MyVideo.mp4) instead of directory.")
    parser.add_argument("--delete_wav", action="store_true", help="Delete denoised WAV after successful diarization.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()
    

    if args.video:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        video_dir = os.path.join(OUTPUT_DIR, base_name)
        targets = [base_name] if os.path.exists(video_dir) else []
    else:
        targets = [
            entry.name for entry in os.scandir(OUTPUT_DIR)
            if entry.is_dir() and
            os.path.exists(os.path.join(entry.path, "_transcript.json"))
        ]
        targets = sorted(targets)

    if not targets:
        logging.info("☕ No transcripts found to diarize. Run transcribe.py first.")
        return

    # Check if enough VRAM si available to load the model
    try:
        script_name = os.path.basename(__file__)
        wait_for_gpu(get_gpu_requirements(script_name), MAX_RETRIES)
        # LOADING MODEL ONCE (Moved out of loop for VRAM stability)
        logging.info(f"🚀 Loading WhisperX 'medium' model on {DEVICE}...")
        #model = whisperx.load_model("medium", DEVICE, compute_type=COMPUTE_TYPE)
        diarize_model    = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    except RuntimeError as e:
        logging.error(f"💀 Pipeline Aborted: {e}")
        sys.exit(1) # Exit with error code for the orchestrator
        
    processed = skipped = failed = 0

    for base_name in targets:
        if os.path.exists(metadata_path_for(base_name))  and not args.force:
            logging.info(f"⏭️  Skipping {base_name} — already diarized.")
            skipped += 1
            continue

        try:
            result = diarize(base_name, diarize_model)
            if result:
                save_outputs(result, base_name)
                if args.delete_wav:
                    out_dir = video_output_dir(base_name)
                    wav = os.path.join(out_dir, "_final_16k.wav")
                    if os.path.exists(wav):
                        os.remove(wav)
                        logging.info(f"🗑️ Deleted WAV: {wav}")
                processed += 1
            else:
                failed += 1
        except Exception as e:
            logging.error(f"❌ Failed on {base_name}: {e}", exc_info=True)
            flush_gpu()
            failed += 1

      # Final Cleanup before exiting so Diarize phase can start
    del diarize_model
    flush_gpu()
    logging.info("🏁 GPU memory released for next phase.")
    logging.info(f"✅ Diarization complete — processed: {processed}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()