"""
FILE: denoise.py
ROLE: Phase 0 — Audio Extraction & Denoising
-------------------------------------------------------------------------
DESCRIPTION:
Extracts audio from video files and runs DeepFilterNet3 denoising.
Outputs a clean 16kHz mono WAV file ready for WhisperX transcription.

INPUT:
- Video files in 'samples/sanitized/'

OUTPUT:
# - tracking_results/{video_name}/_final_16k.wav

IDEMPOTENT:
- Skips files where output WAV already exists.

HARDWARE:
- DeepFilterNet3 runs on CUDA by default for ~10-15x speedup over CPU.
- Falls back to CPU automatically if GPU memory is insufficient.
- A pre-flight guard prevents running alongside transcribe/diarize phases.
-------------------------------------------------------------------------
"""


import os
import sys
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import argparse
import torch
from pathlib import Path
from safe_globals import register_omegaconf_only, patch_hf_hub 
from pipeline_utils import setup_logging, video_output_dir, OUTPUT_DIR

# ── HuggingFace Hub Patch and weights only  ────────────────────────────────────
register_omegaconf_only()
patch_hf_hub()

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR            = "samples/sanitized"
DENOISE_PYTHON       = sys.executable
SUPPORTED_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.mpeg', '.mpg')

#LOG_DIR = "logs"

# Safety threshold — if more than this is reserved, another phase is likely running.
# DeepFilterNet3 needs ~1.5GB; 2GB threshold is conservative but safe on an 11GB card.
GPU_CONFLICT_THRESHOLD_GB = 2.0
LOG_DIR  = "logs"

# ── Logging Configuration ─────────────────────────────────────────────────────

setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ── GPU Helpers ───────────────────────────────────────────────────────────────
def get_gpu_reserved_gb() -> float:
    """Returns currently reserved GPU memory in GB. Returns 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024 ** 3)


def check_gpu_free():
    """
    Raises RuntimeError if GPU memory is already heavily in use,
    indicating another pipeline phase (transcribe/diarize) may be running.
    Safe to call even on CPU-only machines.
    """
    reserved_gb = get_gpu_reserved_gb()
    if reserved_gb > GPU_CONFLICT_THRESHOLD_GB:
        raise RuntimeError(
            f"GPU already has {reserved_gb:.1f}GB reserved — "
            f"is transcribe.py or diarize.py running concurrently? "
            f"Denoise must run alone to avoid OOM on an 11GB card."
        )


def select_denoise_device() -> str:
    """
    DeepFilterNet3 has cuDNN GRU compatibility issues on Pascal-architecture
    GPUs (GTX 1080 Ti, compute capability 6.1) with PyTorch 2.6+.
    CPU is the stable choice here — GPU time is better saved for WhisperX
    and Pyannote which give much larger speedups.
    """
    if not torch.cuda.is_available():
        logging.info("ℹ️  No CUDA available — using CPU for denoising.")
        return "cpu"

    cc_major = torch.cuda.get_device_properties(0).major
    if cc_major < 7:
        # Pascal (6.x) and older have cuDNN GRU issues with DeepFilterNet3
        # on PyTorch 2.6+. Fall back to CPU silently.
        gpu_name = torch.cuda.get_device_properties(0).name
        logging.warning(
            f"⚠️  {gpu_name} (compute {cc_major}.x) has cuDNN GRU compatibility "
            f"issues with DeepFilterNet3 on PyTorch 2.6+. Using CPU for denoising."
        )
        return "cpu"

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logging.info(f"🎮 GPU compatible — {total_gb:.1f}GB total. Using CUDA for denoising.")
    return "cuda"


# ── Core Helpers ──────────────────────────────────────────────────────────────
def get_video_codec(video_path: str) -> str | None:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0",
        video_path,
    ]
    try:
        return subprocess.check_output(cmd).decode("utf-8").strip()
    except Exception:
        return None

def get_log_file_handle():
    """
    Returns the file handle from the already-open logging FileHandler.
    Avoids permission conflicts from opening the log file a second time.
    """
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.stream
    return None

def output_path_for(video_name: str) -> str:
    base = os.path.splitext(video_name)[0]
    return os.path.join(video_output_dir(base), "_final_16k.wav")


def denoise(video_path: str, device: str) -> str:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    #raw_wav   = os.path.join(VIDEO_DIR, f"{base_name}_raw_tmp.wav")
    target_dir = video_output_dir(base_name)
    raw_wav = os.path.join(target_dir, f"{base_name}_raw_tmp.wav")
    final_wav = output_path_for(os.path.basename(video_path))

    # ── Step 1: Extract raw audio via ffmpeg ──────────────────────────────
    codec       = get_video_codec(video_path)
    decoder_map = {
        "h264": "h264_cuvid", "hevc": "hevc_cuvid", "mjpeg": "mjpeg_cuvid",
        "vp8":  "vp8_cuvid",  "vp9":  "vp9_cuvid",
    }
    hw_decoder = decoder_map.get(codec)
    ffmpeg_cmd = ["ffmpeg", "-y", "-hwaccel", "cuda"]

    if hw_decoder:
        ffmpeg_cmd.extend(["-c:v", hw_decoder])
        logging.info(f"🚀 Video decode: {hw_decoder} (GPU-accelerated ffmpeg, independent of denoise device)")

    ffmpeg_cmd.extend([
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        raw_wav,
    ])
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    logging.info(f"🎙️  Raw audio extracted → {raw_wav}")

    # ── Step 2: DeepFilterNet3 denoising ─────────────────────────────────
    # df.enhance does not support a --device flag — device is controlled
    # via CUDA_VISIBLE_DEVICES. Setting it to "" forces CPU; leaving it
    # unset lets DeepFilterNet pick the GPU automatically.
    logging.info(f"🧹 Denoising {base_name} on {device.upper()}...")
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU → forces CPU
    log_handle = get_log_file_handle()
    target_dir = video_output_dir(base_name) # Get the tracking_results/{base_name} path
    subprocess.run(
        [DENOISE_PYTHON, "-m", "df.enhance", raw_wav, "--output-dir", target_dir],
        env=env,
        check=True,
        stdout=log_handle,
        stderr=log_handle,
    )

    # Improved file detection: specifically look for the file just created
    # DeepFilterNet appends the model name, e.g., "input_DeepFilterNet3.wav"
    expected_name = f"{base_name}_raw_tmp_DeepFilterNet3.wav"
    enhanced_file = os.path.join(target_dir, expected_name)

    if not os.path.exists(enhanced_file):
        # Fallback to general search if model version changes
        enhanced_file = next(
            (os.path.join(target_dir, f) for f in os.listdir(target_dir)
             if f.startswith(f"{base_name}_raw_tmp") and f.endswith(".wav")),
            None
        )

    if not enhanced_file:
        logging.warning("⚠️  No enhanced file found — falling back to raw audio.")
        return raw_wav

    # ── Step 3: Resample to 16kHz mono (WhisperX requirement) ────────────
    subprocess.run(
        ["ffmpeg", "-y", "-i", enhanced_file, "-ar", "16000", "-ac", "1", final_wav],
        check=True, capture_output=True,
    )
    logging.info(f"✅ Denoised audio ready → {final_wav}")

    # ── Cleanup intermediates ─────────────────────────────────────────────
    for f in (raw_wav, enhanced_file):
        if f and os.path.exists(f):
            os.remove(f)

    return final_wav


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 0: Audio Extraction & Denoising")
    parser.add_argument("--video", type=str, default=None, help="Process a single video file instead of the whole directory.",)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU denoising even if GPU is available.",)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    # ── GPU pre-flight check ──────────────────────────────────────────────
    if not args.force_cpu:
        try:
            check_gpu_free()
        except RuntimeError as e:
            logging.warning(str(e))
            logging.warning("⚠️  Proceeding with CPU fallback.")

    device = "cpu" if args.force_cpu else select_denoise_device()
    logging.info(f"🎛️  Denoise device: {device.upper()}")

    # ── Discover targets ──────────────────────────────────────────────────
    if args.video:
        targets = [args.video] if os.path.isfile(args.video) else []
        if not targets:
            logging.error(f"❌ File not found: {args.video}")
    else:
        targets = [
            os.path.join(VIDEO_DIR, f)
            for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

    processed = skipped = failed = 0
    targets = sorted(targets)

    for video_path in targets:
        video_name = os.path.basename(video_path)
        out = output_path_for(video_name)

        if os.path.exists(out) and not args.force:
            logging.info(f"⏭️  Skipping {video_name} — already denoised.")
            skipped += 1
            continue

        try:
            denoise(video_path, device)
            processed += 1
        except Exception as e:
            logging.error(f"❌ Failed on {video_name}: {e}", exc_info=True)
            failed += 1

    logging.info(
        f"✅ Denoising complete — processed: {processed}, skipped: {skipped}, failed: {failed}"
    )


if __name__ == "__main__":
    main()