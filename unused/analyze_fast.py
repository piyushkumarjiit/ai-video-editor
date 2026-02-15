#!/usr/bin/env python3
"""
Fast scene analyzer using ffprobe statistics instead of frame extraction.
Fully automated batch processor - no user input required.
"""
import subprocess
import json
import sys
from pathlib import Path

INPUT_DIR = "clips"
OUTPUT_DIR = "clips_speed"
ANALYSIS_JSON = "scene_analysis.json"

# Classification thresholds based on bitrate variance
HIGH_VARIANCE_THRESHOLD = 0.15  # High activity
LOW_VARIANCE_THRESHOLD = 0.05   # Low activity


def get_video_stats(video_path):
    """Get video statistics using ffprobe"""
    # Get basic info
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,bit_rate:stream=codec_type,nb_frames",
        "-of", "json",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    info = json.loads(result.stdout)
    
    duration = float(info["format"].get("duration", 0))
    bitrate = int(info["format"].get("bit_rate", 0))
    
    return duration, bitrate


def analyze_scene_fast(clip_path):
    """Quick analysis using thumbnail extraction at key points"""
    print(f"Analyzing {clip_path.name}...", end=" ", flush=True)
    
    try:
        duration, bitrate = get_video_stats(clip_path)
        file_size = clip_path.stat().st_size
        
        # Extract 3 thumbnails: beginning, middle, end
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            thumbnails = []
            times = [duration * 0.2, duration * 0.5, duration * 0.8]  # 20%, 50%, 80%
            
            for i, t in enumerate(times):
                out = f"{tmpdir}/thumb_{i}.jpg"
                cmd = [
                    "ffmpeg", "-ss", str(t), "-i", str(clip_path),
                    "-frames:v", "1", "-vf", "scale=160:-1",
                    "-q:v", "5", out
                ]
                subprocess.run(cmd, capture_output=True, timeout=10)
                
                if Path(out).exists():
                    # Get file size as complexity proxy
                    thumb_size = Path(out).stat().st_size
                    thumbnails.append(thumb_size)
            
            if len(thumbnails) >= 2:
                # Calculate variance in thumbnail sizes
                # High variance = changing scene (interesting)
                mean_size = sum(thumbnails) / len(thumbnails)
                variance = sum((s - mean_size) ** 2 for s in thumbnails) / len(thumbnails)
                score = (variance / mean_size) * 100 if mean_size > 0 else 0
            else:
                score = 0
        
        # Classify
        if score >= 15:
            category, speed = "interesting", 1.0
        elif score >= 5:
            category, speed = "moderate", 2.0
        else:
            category, speed = "boring", 4.0
        
        print(f"✓ score={score:.1f} → {category} ({speed}x)")
        
        return {
            "file": clip_path.name,
            "duration": round(duration, 2),
            "change_score": round(score, 2),
            "category": category,
            "recommended_speed": speed,
            "file_size_mb": round(file_size / 1024 / 1024, 1)
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def apply_speed_filter(input_path, output_path, speed):
    """Re-encode video with speed filter"""
    if speed == 1.0:
        subprocess.run(["cp", input_path, output_path], check=True)
        return
    
    video_filter = f"setpts={1/speed}*PTS"
    
    # Build audio filter chain for speeds > 2x
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
        "-c:v", "libx265", "-preset", "faster", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    apply_speed = "--apply" in sys.argv or "-a" in sys.argv
    
    clips_dir = Path(INPUT_DIR)
    clip_files = sorted(clips_dir.glob("scene_*.mkv"))
    
    if not clip_files:
        print(f"No scene clips found in {INPUT_DIR}/")
        return
    
    print(f"🎬 Fast Batch Analysis - {len(clip_files)} clips\n")
    
    results = []
    for clip_path in clip_files:
        result = analyze_scene_fast(clip_path)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No scenes analyzed")
        return
    
    # Save report
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    # Statistics
    interesting = sum(1 for r in results if r["category"] == "interesting")
    moderate = sum(1 for r in results if r["category"] == "moderate")
    boring = sum(1 for r in results if r["category"] == "boring")
    
    total_duration = sum(r["duration"] for r in results)
    sped_up_duration = sum(r["duration"] / r["recommended_speed"] for r in results)
    time_saved = total_duration - sped_up_duration
    
    print(f"\n{'='*60}")
    print(f"📊 Analysis Report: {ANALYSIS_JSON}")
    print(f"{'='*60}")
    print(f"  Interesting (1x): {interesting} scenes")
    print(f"  Moderate (2x):    {moderate} scenes")  
    print(f"  Boring (4x):      {boring} scenes")
    print(f"\n  Original:  {total_duration/60:.1f} min")
    print(f"  Speed-up:  {sped_up_duration/60:.1f} min")
    print(f"  Saved:     {time_saved/60:.1f} min ({time_saved/total_duration*100:.0f}%)")
    print(f"{'='*60}")
    
    if apply_speed:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        print(f"\n⚡ Applying speed changes...\n")
        
        for result in results:
            input_path = clips_dir / result["file"]
            output_path = Path(OUTPUT_DIR) / result["file"]
            speed = result["recommended_speed"]
            
            if speed == 1.0:
                print(f"  {result['file']} → 1x (copy)")
                subprocess.run(["cp", str(input_path), str(output_path)])
            else:
                print(f"  {result['file']} → {speed}x (encoding...)", end="", flush=True)
                apply_speed_filter(str(input_path), str(output_path), speed)
                print(" ✓")
        
        print(f"\n✅ Output: {OUTPUT_DIR}/")
    else:
        print(f"\n💡 To apply: python analyze_fast.py --apply")


if __name__ == "__main__":
    main()
