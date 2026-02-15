import csv
import subprocess
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

INPUT = "IMG_3839.MOV"
CSV = "scenes.csv/IMG_3839-Scenes.csv"
OUTDIR = "clips"
MIN_DURATION = 3.0  # seconds

# GPU acceleration settings
USE_GPU = True  # Set to False to disable GPU
GPU_DECODER = "hevc_cuvid"  # NVIDIA: hevc_cuvid, Intel: hevc_qsv, AMD: hevc_vaapi

os.makedirs(OUTDIR, exist_ok=True)


def clean_float(x):
    return float(x.strip().replace("\ufeff", ""))


def check_gpu_available():
    """Check if GPU decoding is available"""
    if not USE_GPU:
        return False
    
    try:
        # Test NVIDIA CUDA decoder
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=5
        )
        return GPU_DECODER in result.stdout
    except:
        return False


def get_scenes_from_csv():
    """Parse CSV and return list of scenes to extract"""
    scenes = []
    
    with open(CSV, newline="") as f:
        reader = csv.reader(f)
        
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            
            scene = int(row[0])
            start = clean_float(row[3])
            end = clean_float(row[6])
            duration = end - start
            
            if duration < MIN_DURATION:
                continue
            
            scenes.append({
                'scene': scene,
                'start': start,
                'end': end,
                'duration': duration
            })
    
    return scenes


def extract_scene(scene_info):
    """Extract a single scene (for parallel processing)"""
    scene = scene_info['scene']
    start = scene_info['start']
    duration = scene_info['duration']
    out = f"{OUTDIR}/scene_{scene:03d}.mkv"
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    
    # GPU-accelerated decoding (if available and needed for re-encoding)
    # For stream copy, GPU doesn't help, but keep this for future re-encoding
    if scene_info.get('use_gpu'):
        cmd += ["-hwaccel", "cuda", "-c:v", GPU_DECODER]
    
    # Fast seeking: -ss before -i for keyframe seeking
    cmd += [
        "-ss", str(start),
        "-i", INPUT,
        "-t", str(duration),
        "-map", "0:v",
        "-map", "0:a",
        "-c", "copy",  # Stream copy - fastest, no re-encoding
        "-avoid_negative_ts", "make_zero",
        out
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=300
        )
        return (scene, True, None)
    except subprocess.CalledProcessError as e:
        return (scene, False, str(e))
    except Exception as e:
        return (scene, False, str(e))


def extract_scenes_parallel(scenes, max_workers=None):
    """Extract multiple scenes in parallel"""
    if max_workers is None:
        # Use number of CPU cores, but cap at 4 for I/O reasons
        max_workers = min(os.cpu_count() or 4, 4)
    
    print(f"Processing {len(scenes)} scenes with {max_workers} parallel workers\n")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_scene = {
            executor.submit(extract_scene, scene): scene 
            for scene in scenes
        }
        
        # Process results as they complete
        for future in as_completed(future_to_scene):
            scene_info = future_to_scene[future]
            scene_num, success, error = future.result()
            
            if success:
                print(f"✓ Scene {scene_num:03d}: {scene_info['start']:.2f}s → {scene_info['end']:.2f}s ({scene_info['duration']:.1f}s)")
            else:
                print(f"✗ Scene {scene_num:03d} FAILED: {error}")
            
            results.append((scene_num, success))
    
    return results


def extract_scenes_single_pass(scenes):
    """
    Alternative approach: Use ffmpeg segment muxer for single-pass extraction.
    This can be faster for many small scenes, but more complex to set up.
    Currently not used but kept for reference.
    """
    # Build segment times
    segment_times = []
    for i, scene in enumerate(scenes):
        if i > 0:
            segment_times.append(str(scene['start']))
    
    cmd = [
        "ffmpeg", "-i", INPUT,
        "-f", "segment",
        "-segment_times", ",".join(segment_times),
        "-map", "0:v", "-map", "0:a",
        "-c", "copy",
        "-reset_timestamps", "1",
        f"{OUTDIR}/scene_%03d.mkv"
    ]
    
    subprocess.run(cmd, check=True)


def main():
    print(f"🎬 Fast Scene Splitter with Parallel Processing\n")
    print(f"Input: {INPUT}")
    print(f"CSV: {CSV}")
    print(f"Output: {OUTDIR}/\n")
    
    # Check GPU availability
    gpu_available = check_gpu_available()
    if USE_GPU:
        print(f"GPU Decoder: {'✓ Available' if gpu_available else '✗ Not available (using CPU)'}\n")
    
    # Parse scenes
    scenes = get_scenes_from_csv()
    
    if not scenes:
        print("No scenes found!")
        return
    
    total_duration = sum(s['duration'] for s in scenes)
    print(f"Found {len(scenes)} scenes (total: {total_duration/60:.1f} minutes)\n")
    
    # Extract scenes in parallel
    results = extract_scenes_parallel(scenes, max_workers=4)
    
    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"\n{'='*50}")
    print(f"✅ Complete: {successful}/{len(results)} scenes extracted")
    if failed:
        print(f"❌ Failed: {failed} scenes")
    print(f"📁 Output directory: {OUTDIR}/")


if __name__ == "__main__":
    main()
