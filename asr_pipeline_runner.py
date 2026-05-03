import os
import sys
import time
import logging
import argparse
import subprocess
import threading
import queue
import torch
from pipeline_utils import setup_logging, video_output_dir, OUTPUT_DIR, wait_for_gpu, \
    get_gpu_requirements, MAX_RETRIES

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR            = "samples/sanitized"
PYTHON               = sys.executable
SUPPORTED_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
FILE_STABILITY_WAIT  = 5 

# Queues
cpu_queue = queue.Queue()
gpu_queue = queue.Queue()

# Global Stats for Thread-Safe Reporting
stats = {"processed": 0, "failed": 0, "skipped": 0}
stats_lock = threading.Lock()

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_file_stable(filepath: str) -> bool:
    try:
        size_before = os.path.getsize(filepath)
        time.sleep(FILE_STABILITY_WAIT)
        return os.path.getsize(filepath) == size_before and size_before > 0
    except Exception:
        return False

def run_phase(script, args_list):
    cmd = [PYTHON, script] + args_list
    result = subprocess.run(cmd)
    return result.returncode == 0

# ── Workers ──────────────────────────────────────────────────────────────────
def cpu_worker(args):
    """Consumer for Phase 0 (Denoising)."""
    threading.current_thread().name = "CPU-Worker"
    base_flags = ["--force"] if args.force else []

    while True:
        task = cpu_queue.get()
        if task is None: 
            cpu_queue.task_done()
            break
            
        base_name, video_path = task
        
        if not is_file_stable(video_path):
            logging.warning(f"⚠️ File {base_name} is unstable. Skipping.")
            with stats_lock:
                stats["skipped"] += 1
            cpu_queue.task_done()
            continue

        denoised_wav = os.path.join(OUTPUT_DIR, base_name, "_final_16k.wav")
        
        if not os.path.exists(denoised_wav) or args.force:
            logging.info(f"🔊 Denoising: {base_name}")
            if run_phase("denoise.py", ["--video", video_path] + base_flags):
                gpu_queue.put((base_name, video_path))
            else:
                logging.error(f"❌ Denoise failed for {base_name}. Task aborted.")
                with stats_lock:
                    stats["failed"] += 1
        else:
            logging.info(f"⏭️ Denoise skipped (exists) for {base_name}")
            gpu_queue.put((base_name, video_path))
        
        cpu_queue.task_done()

def gpu_worker(args):
    """Consumer for Phase 1-2 (Transcribe/Diarize)."""
    threading.current_thread().name = "GPU-Worker"
    force_flag = ["--force"] if args.force else []

    while True:
        task = gpu_queue.get()
        if task is None: 
            gpu_queue.task_done()
            break
            
        base_name, video_path = task

        # Phase 1: Transcribe
        wait_for_gpu(get_gpu_requirements("transcribe.py"), MAX_RETRIES)
        t_args = ["--video", video_path, "--batch_size", str(args.batch_size)] + force_flag
        
        if run_phase("transcribe.py", t_args):
            # Phase 2: Diarize
            wait_for_gpu(get_gpu_requirements("diarize.py"), MAX_RETRIES)
            d_args = ["--video", video_path] + force_flag
            if args.delete_wav: d_args.append("--delete_wav")
            
            if run_phase("diarize.py", d_args):
                with stats_lock:
                    stats["processed"] += 1
                logging.info(f"✨ Full Pipeline Complete: {base_name}")
            else:
                with stats_lock:
                    stats["failed"] += 1
        else:
            with stats_lock:
                stats["failed"] += 1
        
        gpu_queue.task_done()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--delete_wav", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if files exist.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start Consumers
    threading.Thread(target=cpu_worker, args=(args,), daemon=False).start()
    threading.Thread(target=gpu_worker, args=(args,), daemon=False).start()

    logging.info(f"🔍 Scanning folders (Force Mode: {args.force})...")
    for file in os.listdir(VIDEO_DIR):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            base_name = os.path.splitext(file)[0]
            metadata_path = os.path.join(OUTPUT_DIR, base_name, "_metadata.json")
            
            if not os.path.exists(metadata_path) or args.force:
                cpu_queue.put((base_name, os.path.join(VIDEO_DIR, file)))

    # Waterfall Shutdown
    cpu_queue.put(None)  
    cpu_queue.join() 
    
    gpu_queue.put(None) 
    gpu_queue.join() 
    
    logging.info("─" * 40)
    logging.info(f"🏁 Pipeline complete.")
    logging.info(f"✅ Processed: {stats['processed']}")
    logging.info(f"⚠️ Skipped:   {stats['skipped']}")
    logging.info(f"❌ Failed:    {stats['failed']}")
    logging.info("─" * 40)

if __name__ == "__main__":
    main()