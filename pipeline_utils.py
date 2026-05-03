"""
FILE: pipeline_utils.py
ROLE: Centralized Utility Engine for AI Video Pipeline
-------------------------------------------------------------------------
DESCRIPTION:
Provides a unified framework for logging, hardware orchestration, and 
directory management. Ensures consistent environment behavior across 
denoising, transcription, and diarization phases.

CORE CAPABILITIES:
- HARDWARE: GPU VRAM monitoring and conflict resolution.
- LOGGING: Global RotatingFileHandler configuration with noise reduction.
- IO: Atomic directory creation and path resolution for idempotent processing.
- PATCHING: (Via safe_globals integration) Handles HF Hub and OmegaConf 
  compatibility for PyTorch 2.0+ environments.

HARDWARE COMPATIBILITY:
- Optimized for NVIDIA Pascal architecture (GTX 1080 Ti).
- Handles cuDNN GRU compatibility fallbacks for DeepFilterNet3.
-------------------------------------------------------------------------
"""

import os
import sys
import time
import logging
import torch
from logging.handlers import RotatingFileHandler

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = "logs"
OUTPUT_DIR = "tracking_results"
MAX_RETRIES = 20  # Set to 20 for the 5-minute wait (20 * 15s)
TRANSCRIBE_MINIMUM_VRAM = 3500
DIARIZE_MINIMUM_VRAM = 3500

# ── Methods ───────────────────────────────────────────────────────────────────

def setup_logging(script_name: str, log_dir: str = LOG_DIR) -> str:
    """Configures rotating file + console logging. Returns log file path."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{script_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler(sys.stdout),
        ],
        force=True # Ensures reconfiguration if the orchestrator already started logging
    )
    
    for noisy in ("whisperx", "speechbrain", "pyannote", "faster_whisper", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path

def video_output_dir(base_name: str) -> str:
    """Returns the per-video output directory, creating it if needed."""
    path = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_gpu_requirements(script_name):
    estimates = {
        "transcribe.py": TRANSCRIBE_MINIMUM_VRAM,
        "diarize.py": DIARIZE_MINIMUM_VRAM,
    }
    return estimates.get(script_name, 2000)


def wait_for_gpu(required_mb, max_retries=MAX_RETRIES):
    """
    Waits for a specific amount of VRAM to be free. 
    MAX_RETRIES retries at 15s intervals.
    """
    if not torch.cuda.is_available():
        return
    
    required_mb = int(required_mb)
    retries = 0
    
    while retries < max_retries:
        free_bytes, _ = torch.cuda.mem_get_info()
        os_free_mb = free_bytes / (1024**2)
        
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        pytorch_actual_free = os_free_mb + (reserved_mb - allocated_mb)

        if pytorch_actual_free >= required_mb:
            logging.info(f"✅ GPU VRAM Sufficient: {pytorch_actual_free:.0f}MB effectively free.")
            return
            
        retries += 1
        logging.info(f"⏳ GPU Busy ({retries}/{max_retries}): {pytorch_actual_free:.0f}MB free. Waiting...")
        time.sleep(15)

    raise RuntimeError(f"❌ GPU timeout: Could not reclaim {required_mb}MB after {max_retries} attempts.")