#!/usr/bin/env python3
"""
Smart scene analyzer that understands scale modeling workflow.
Detects actual progress vs repetitive boring footage.
"""
import subprocess
import json
import sys
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

INPUT_DIR = "clips"
OUTPUT_DIR = "clips_speed"
ANALYSIS_JSON = "scene_analysis.json"

# ============================================================
# TUNABLE PARAMETERS - Adjust these to control aggressiveness
# ============================================================

# Scene filtering thresholds
MIN_SCENE_DURATION = 10          # Skip scenes shorter than this (seconds)
MAX_STATIC_DURATION = 150        # Skip very long static scenes (seconds)
SIMILARITY_SKIP_THRESHOLD = 92   # Skip near-duplicates above this % (0-100)

# Progress score multipliers for classification
HIGH_PROGRESS_MULTIPLIER = 0.8   # Keep 1x: progress > avg + (std * this)
GOOD_PROGRESS_MULTIPLIER = 0.3   # Speed 1.5x: progress > avg + (std * this)
AVG_PROGRESS_MULTIPLIER = 0.3    # Speed 2.5x: progress within avg ± (std * this)

# Speed settings for each category
SPEED_INTERESTING = 1.0          # High progress scenes
SPEED_GOOD = 1.5                 # Good progress scenes  
SPEED_MODERATE = 2.5             # Average activity scenes
SPEED_BORING = 3.5               # Low activity scenes
SPEED_VERY_BORING = 4.0          # Long & static scenes

# ============================================================


def extract_key_frames(video_path):
    """Extract frames at key positions: start, middle, end"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        
        frames = []
        # Extract at 10%, 50%, 90% (skip very start/end which may be transitions)
        for pct, name in [(0.1, "start"), (0.5, "mid"), (0.9, "end")]:
            t = duration * pct
            out = f"{tmpdir}/{name}.jpg"
            
            subprocess.run([
                "ffmpeg", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-vf", "scale=640:-1", "-q:v", "2", out
            ], capture_output=True, timeout=15)
            
            if Path(out).exists():
                img = Image.open(out).convert('RGB')
                frames.append((name, np.array(img)))
        
        return frames, duration


def calculate_color_histogram(img):
    """Calculate color histogram for image"""
    hist_r = np.histogram(img[:,:,0], bins=32, range=(0, 256))[0]
    hist_g = np.histogram(img[:,:,1], bins=32, range=(0, 256))[0]
    hist_b = np.histogram(img[:,:,2], bins=32, range=(0, 256))[0]
    
    # Normalize
    total = hist_r.sum()
    if total > 0:
        hist_r = hist_r / total
        hist_g = hist_g / total
        hist_b = hist_b / total
    
    return np.concatenate([hist_r, hist_g, hist_b])


def calculate_edges(img):
    """Calculate edge intensity (complexity measure)"""
    gray = np.mean(img, axis=2).astype(np.uint8)
    # Simple Sobel-like edge detection
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    return np.mean(edges_x) + np.mean(edges_y)


def histogram_distance(hist1, hist2):
    """Calculate chi-square distance between histograms"""
    epsilon = 1e-10
    chi_sq = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
    return chi_sq


def analyze_scene_progress(frames):
    """
    Analyze if scene shows actual progress/changes.
    Returns: progress_score (0-100)
    """
    if len(frames) < 2:
        return 0.0
    
    # Extract images
    imgs = {name: img for name, img in frames}
    
    # Compare START to END (most important for progress detection)
    if "start" in imgs and "end" in imgs:
        start_img = imgs["start"]
        end_img = imgs["end"]
        
        # 1. Color histogram difference (detects painting/finishing changes)
        hist_start = calculate_color_histogram(start_img)
        hist_end = calculate_color_histogram(end_img)
        color_diff = histogram_distance(hist_start, hist_end)
        
        # 2. Pixel difference (detects new parts, assembly)
        pixel_diff = np.mean(np.abs(start_img.astype(float) - end_img.astype(float)))
        
        # 3. Edge difference (detects shape/structure changes)
        edges_start = calculate_edges(start_img)
        edges_end = calculate_edges(end_img)
        edge_diff = abs(edges_end - edges_start) / (max(edges_start, edges_end) + 1)
        
        # Combined score (weighted)
        progress_score = (color_diff * 10) + (pixel_diff * 0.5) + (edge_diff * 20)
        
        return min(progress_score, 100)  # Cap at 100
    
    return 0.0


def compare_scenes(scene1_frames, scene2_frames):
    """
    Compare two scenes to detect if they're similar/duplicate.
    Returns similarity score (0-100, higher = more similar)
    """
    if not scene1_frames or not scene2_frames:
        return 0.0
    
    # Compare middle frames (most representative)
    imgs1 = {name: img for name, img in scene1_frames}
    imgs2 = {name: img for name, img in scene2_frames}
    
    if "mid" in imgs1 and "mid" in imgs2:
        hist1 = calculate_color_histogram(imgs1["mid"])
        hist2 = calculate_color_histogram(imgs2["mid"])
        
        # Low distance = high similarity
        dist = histogram_distance(hist1, hist2)
        similarity = max(0, 100 - dist * 10)
        
        return similarity
    
    return 0.0


def analyze_all_scenes():
    """Analyze all scene clips with global comparison"""
    clips_dir = Path(INPUT_DIR)
    clip_files = sorted(clips_dir.glob("scene_*.mkv"))
    
    if not clip_files:
        print(f"No scene clips found in {INPUT_DIR}/")
        return []
    
    print(f"🎬 Smart Analysis - {len(clip_files)} clips")
    print(f"Phase 1: Extracting features from all scenes...\n")
    
    # PHASE 1: Extract features from ALL scenes
    scene_data = []
    for clip_path in clip_files:
        print(f"  {clip_path.name}...", end=" ", flush=True)
        
        try:
            frames, duration = extract_key_frames(str(clip_path))
            
            if not frames or len(frames) < 2:
                print("SKIP (no frames)")
                continue
            
            # Calculate features
            progress_score = analyze_scene_progress(frames)
            
            # Calculate histograms for clustering
            imgs = {name: img for name, img in frames}
            mid_hist = calculate_color_histogram(imgs["mid"]) if "mid" in imgs else None
            
            scene_data.append({
                "file": clip_path.name,
                "path": clip_path,
                "duration": duration,
                "frames": frames,
                "progress_score": progress_score,
                "mid_histogram": mid_hist
            })
            
            print(f"✓ progress={progress_score:.1f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not scene_data:
        print("\n❌ No scenes could be analyzed")
        return []
    
    print(f"\nPhase 2: Global comparison and classification...\n")
    
    # PHASE 2: Global comparison - find similar/duplicate scenes
    similarity_matrix = []
    for i, scene1 in enumerate(scene_data):
        similarities = []
        for j, scene2 in enumerate(scene_data):
            if i == j:
                similarities.append(0)
                continue
            
            if scene1["mid_histogram"] is not None and scene2["mid_histogram"] is not None:
                dist = histogram_distance(scene1["mid_histogram"], scene2["mid_histogram"])
                similarity = max(0, 100 - dist * 10)
                similarities.append(similarity)
            else:
                similarities.append(0)
        
        similarity_matrix.append(similarities)
    
    # PHASE 3: Calculate statistics across all scenes
    all_progress_scores = [s["progress_score"] for s in scene_data]
    avg_progress = np.mean(all_progress_scores)
    std_progress = np.std(all_progress_scores)
    
    print(f"Average progress score: {avg_progress:.1f} (std: {std_progress:.1f})")
    print(f"Progress range: {min(all_progress_scores):.1f} - {max(all_progress_scores):.1f}\n")
    
    # PHASE 4: Classify each scene based on global context
    results = []
    kept_indices = []
    
    for i, scene in enumerate(scene_data):
        # Check if TOO similar to any already kept scene
        max_similarity = 0
        for kept_idx in kept_indices:
            max_similarity = max(max_similarity, similarity_matrix[i][kept_idx])
        
        duration = scene["duration"]
        progress = scene["progress_score"]
        
        # Classification logic based on global context and tunable parameters
        reason = ""
        
        # Skip very short scenes
        if duration < MIN_SCENE_DURATION:
            category, speed = "skip", 0.0
            reason = "too short"
        
        # Skip extremely long static scenes
        elif duration > MAX_STATIC_DURATION and progress < avg_progress - std_progress:
            category, speed = "skip", 0.0
            reason = f"too long & static (score={progress:.0f})"
        
        # Skip near-duplicates with low progress
        elif max_similarity > SIMILARITY_SKIP_THRESHOLD and progress < avg_progress - std_progress * 0.5:
            category, speed = "skip", 0.0
            reason = f"near-duplicate & no progress (sim={max_similarity:.0f}%)"
        
        # Very high progress = keep at normal speed
        elif progress > avg_progress + std_progress * HIGH_PROGRESS_MULTIPLIER:
            category, speed = "interesting", SPEED_INTERESTING
            reason = f"high progress (score={progress:.0f}, avg={avg_progress:.0f})"
            kept_indices.append(i)
        
        # Good progress = slight speedup
        elif progress > avg_progress + std_progress * GOOD_PROGRESS_MULTIPLIER:
            category, speed = "moderate", SPEED_GOOD
            reason = f"good progress (score={progress:.0f})"
            kept_indices.append(i)
        
        # Average progress = moderate speedup
        elif progress > avg_progress - std_progress * AVG_PROGRESS_MULTIPLIER:
            category, speed = "moderate", SPEED_MODERATE
            reason = f"average activity (score={progress:.0f})"
            kept_indices.append(i)
        
        # Long scene with low progress = aggressive speedup
        elif duration > 100 and progress < avg_progress:
            category, speed = "boring", SPEED_VERY_BORING
            reason = f"long & static (score={progress:.0f})"
            kept_indices.append(i)
        
        # Low progress = fast speedup
        else:
            category, speed = "boring", SPEED_BORING
            reason = f"low activity (score={progress:.0f})"
            kept_indices.append(i)
        
        print(f"  {scene['file']}: {category} ({speed}x) - {reason}")
        
        results.append({
            "file": scene["file"],
            "duration": round(duration, 2),
            "progress_score": round(progress, 2),
            "max_similarity_to_kept": round(max_similarity, 2),
            "category": category,
            "recommended_speed": speed,
            "reason": reason
        })
    
    return results


def apply_speed_filter(input_path, output_path, speed):
    """Re-encode video with speed filter using GPU acceleration"""
    if speed == 1.0:
        subprocess.run(["cp", input_path, output_path], check=True)
        return
    
    video_filter = f"setpts={1/speed}*PTS"
    
    audio_filters = []
    remaining_speed = speed
    while remaining_speed > 2:
        audio_filters.append("atempo=2.0")
        remaining_speed /= 2
    if remaining_speed > 1:
        audio_filters.append(f"atempo={remaining_speed}")
    audio_filter = ",".join(audio_filters)
    
    # Try GPU encoding first (NVIDIA NVENC)
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",  # GPU decode
        "-hwaccel_output_format", "cuda",
        "-i", input_path,
        "-filter:v", video_filter,
        "-filter:a", audio_filter,
        "-c:v", "hevc_nvenc",  # GPU encode
        "-preset", "p4",  # Fast preset
        "-cq", "28",  # Quality
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    
    # Try GPU encoding, fallback to CPU if fails
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        print(" (GPU failed, using CPU)...", end="", flush=True)
        # Fallback to fast CPU encoding
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            "-c:v", "libx265", 
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        subprocess.run(cmd, check=True)


def main():
    apply_speed = "--apply" in sys.argv or "-a" in sys.argv
    
    results = analyze_all_scenes()
    
    if not results:
        print("\n❌ No scenes analyzed")
        return
    
    # Save report
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    # Statistics
    kept = [r for r in results if r["category"] != "skip"]
    skipped = [r for r in results if r["category"] == "skip"]
    interesting = [r for r in kept if r["category"] == "interesting"]
    moderate = [r for r in kept if r["category"] == "moderate"]
    boring = [r for r in kept if r["category"] == "boring"]
    
    total_duration = sum(r["duration"] for r in kept)
    sped_up_duration = sum(r["duration"] / r["recommended_speed"] for r in kept)
    time_saved = total_duration - sped_up_duration
    
    print(f"\n{'='*60}")
    print(f"📊 Smart Analysis Report: {ANALYSIS_JSON}")
    print(f"{'='*60}")
    print(f"  Interesting (1x): {len(interesting)} scenes")
    print(f"  Moderate (2x):    {len(moderate)} scenes")
    print(f"  Boring (4x):      {len(boring)} scenes")
    print(f"  Skipped:          {len(skipped)} scenes (duplicates/too short)")
    print(f"\n  Original:  {total_duration/60:.1f} min")
    print(f"  Speed-up:  {sped_up_duration/60:.1f} min")
    print(f"  Saved:     {time_saved/60:.1f} min ({time_saved/total_duration*100:.0f}%)")
    print(f"{'='*60}")
    
    if apply_speed:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        print(f"\n⚡ Applying speed changes...\n")
        
        for result in kept:
            input_path = Path(INPUT_DIR) / result["file"]
            output_path = Path(OUTPUT_DIR) / result["file"]
            speed = result["recommended_speed"]
            
            if speed == 1.0:
                print(f"  {result['file']} → 1x (copy)")
                subprocess.run(["cp", str(input_path), str(output_path)])
            else:
                print(f"  {result['file']} → {speed}x", end="", flush=True)
                apply_speed_filter(str(input_path), str(output_path), speed)
                print(" ✓")
        
        print(f"\n✅ Output: {OUTPUT_DIR}/")
        print(f"💡 Skipped scenes not included in output")
    else:
        print(f"\n💡 To apply: python analyze_smart.py --apply")


if __name__ == "__main__":
    main()
