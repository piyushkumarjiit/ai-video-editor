import subprocess
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

INPUT_DIR = "clips"
OUTPUT_DIR = "clips_speed"
ANALYSIS_JSON = "scene_analysis.json"

# Thresholds for classifying scenes
HIGH_CHANGE_THRESHOLD = 15.0   # High visual activity → keep 1x
LOW_CHANGE_THRESHOLD = 5.0     # Low visual activity → speed up

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_sample_frames(video_path, num_samples=10):
    """Extract sample frames at intervals throughout the video"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get video duration first
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        
        # Extract frames at key intervals
        frames = []
        interval = duration / (num_samples + 1)
        
        for i in range(1, num_samples + 1):
            timestamp = i * interval
            output = f"{tmpdir}/frame_{i:04d}.jpg"
            
            cmd = [
                "ffmpeg", "-ss", str(timestamp), "-i", video_path,
                "-frames:v", "1",
                "-vf", "scale=320:-1",
                "-q:v", "2",
                output
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=30)
            
            if os.path.exists(output):
                img = Image.open(output)
                frames.append(np.array(img))
        
        return frames, duration


def calculate_visual_change_fast(frames):
    """Fast visual change calculation using simple metrics"""
    if len(frames) < 2:
        return 0.0
    
    changes = []
    for i in range(len(frames) - 1):
        # Calculate simple difference between frames
        diff = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
        mean_diff = np.mean(diff)
        changes.append(mean_diff)
    
    return np.mean(changes)


def classify_scene(change_score):
    """Classify scene and determine speed multiplier"""
    if change_score >= HIGH_CHANGE_THRESHOLD:
        return "interesting", 1.0
    elif change_score >= LOW_CHANGE_THRESHOLD:
        return "moderate", 2.0
    else:
        return "boring", 4.0


def analyze_scene(clip_path):
    """Analyze a single scene clip"""
    print(f"Analyzing {clip_path.name}...", end=" ", flush=True)
    
    try:
        # Extract sample frames (faster than full fps extraction)
        frames, duration = extract_sample_frames(str(clip_path), num_samples=10)
        
        if len(frames) < 2:
            print("SKIP (insufficient frames)")
            return None
        
        # Calculate visual change score
        change_score = calculate_visual_change_fast(frames)
        category, speed = classify_scene(change_score)
        
        print(f"✓ score={change_score:.1f} → {category} ({speed}x)")
        
        return {
            "file": clip_path.name,
            "duration": round(duration, 2),
            "change_score": round(change_score, 2),
            "category": category,
            "recommended_speed": speed
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def apply_speed_filter(input_path, output_path, speed):
    """Re-encode video with speed filter"""
    if speed == 1.0:
        # Just copy if no speed change needed
        subprocess.run(["cp", input_path, output_path], check=True)
        return
    
    # Use setpts filter for video speed and atempo for audio
    video_filter = f"setpts={1/speed}*PTS"
    
    # Audio speed: use multiple atempo filters if speed > 2x
    audio_filters = []
    remaining_speed = speed
    while remaining_speed > 2:
        audio_filters.append("atempo=2.0")
        remaining_speed /= 2
    if remaining_speed > 1:
        audio_filters.append(f"atempo={remaining_speed}")
    audio_filter = ",".join(audio_filters)
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:v", video_filter,
        "-filter:a", audio_filter,
        "-c:v", "libx265", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    
    subprocess.run(cmd, check=True)


def main():
    # Parse command line arguments
    apply_speed = "--apply" in sys.argv or "-a" in sys.argv
    
    clips_dir = Path(INPUT_DIR)
    clip_files = sorted(clips_dir.glob("scene_*.mkv"))
    
    if not clip_files:
        print(f"No scene clips found in {INPUT_DIR}/")
        return
    
    print(f"🎬 Analyzing {len(clip_files)} scene clips\n")
    
    results = []
    
    for clip_path in clip_files:
        result = analyze_scene(clip_path)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No scenes could be analyzed")
        return
    
    # Save analysis report
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 Analysis complete! Report saved to {ANALYSIS_JSON}\n")
    
    # Summary statistics
    interesting = sum(1 for r in results if r["category"] == "interesting")
    moderate = sum(1 for r in results if r["category"] == "moderate")
    boring = sum(1 for r in results if r["category"] == "boring")
    
    total_duration = sum(r["duration"] for r in results)
    sped_up_duration = sum(r["duration"] / r["recommended_speed"] for r in results)
    time_saved = total_duration - sped_up_duration
    
    print(f"Summary:")
    print(f"  Interesting (1x): {interesting} scenes")
    print(f"  Moderate (2x):    {moderate} scenes")
    print(f"  Boring (4x):      {boring} scenes")
    print(f"\n  Original duration: {total_duration/60:.1f} minutes")
    print(f"  After speed-up:    {sped_up_duration/60:.1f} minutes")
    print(f"  Time saved:        {time_saved/60:.1f} minutes ({time_saved/total_duration*100:.1f}%)")
    print(f"{'='*60}")
    
    # Apply speed changes if requested
    if apply_speed:
        print(f"\n🎬 Applying speed changes to {OUTPUT_DIR}/\n")
        for result in results:
            input_path = clips_dir / result["file"]
            output_path = Path(OUTPUT_DIR) / result["file"]
            speed = result["recommended_speed"]
            
            if speed == 1.0:
                print(f"✓ {result['file']} - keeping at 1x (copying)")
                subprocess.run(["cp", str(input_path), str(output_path)], check=True)
            else:
                print(f"⚡ {result['file']} - speeding up to {speed}x (re-encoding)")
                apply_speed_filter(str(input_path), str(output_path), speed)
        
        print(f"\n✅ Speed-adjusted clips saved to {OUTPUT_DIR}/")
    else:
        print(f"\n💡 To apply speed changes, run: python analyze_scenes.py --apply")


if __name__ == "__main__":
    main()
