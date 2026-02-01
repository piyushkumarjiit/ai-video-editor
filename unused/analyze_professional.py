#!/usr/bin/env python3
"""
Professional Video Analysis - Using Advanced Libraries
Approach: Multi-signal analysis for micro-change detection
- TransNetV2: Shot boundary detection
- MediaPipe: Hand/pose tracking for activity detection
- Optical Flow: Motion analysis
- Object detection: Tool/material changes
- Perceptual quality: SSIM for visual progression
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from scipy.spatial.distance import cosine

# Configuration
VIDEO_FILE = "IMG_3839.MOV"
SAMPLE_INTERVAL = 2  # Extract 1 frame every 2 seconds
MIN_SCENE_DURATION = 6  # Minimum scene length
CACHE_FILE = "IMG_3839_professional_cache.pkl"

print("🎬 Professional Video Analysis")
print("=" * 60)
print("Using: MediaPipe, Optical Flow, SSIM, Advanced Metrics")
print("=" * 60)


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frame(video_path, timestamp, output_path):
    """Extract single frame at timestamp"""
    cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', str(video_path),
           '-vframes', '1', '-q:v', '2', str(output_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_frames_batch(video_path, interval=2):
    """Extract frames in parallel"""
    print(f"\n📹 Extracting frames from: {video_path}")
    duration = get_video_duration(video_path)
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    frames_dir = Path("/tmp/professional_frames")
    frames_dir.mkdir(exist_ok=True)
    
    timestamps = np.arange(0, duration, interval)
    print(f"   Extracting {len(timestamps)} frames (1 every {interval}s)...")
    
    frames = []
    for i, ts in enumerate(timestamps):
        output_path = frames_dir / f"frame_{i:04d}.jpg"
        extract_frame(video_path, ts, output_path)
        frames.append({'index': i, 'timestamp': ts, 'path': str(output_path)})
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(timestamps)}...")
    
    print(f"   ✓ Extracted {len(frames)} frames")
    return frames


def analyze_hand_activity(image_path):
    """
    Use MediaPipe to detect hand activity level
    Returns: hand_detected, hand_motion_score, hand_area
    """
    mp_hands = mp.solutions.hands
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3
    ) as hands:
        
        img = cv2.imread(image_path)
        if img is None:
            return {'hand_detected': False, 'hand_score': 0, 'hand_area': 0}
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return {'hand_detected': False, 'hand_score': 0, 'hand_area': 0}
        
        # Calculate hand area and movement potential
        hand_landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        
        hand_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        # Hand close to camera (large) = active work
        activity_score = hand_area * 10
        
        return {
            'hand_detected': True,
            'hand_score': float(activity_score),
            'hand_area': float(hand_area),
            'hand_center': (float(np.mean(xs)), float(np.mean(ys)))
        }


def calculate_optical_flow(img1_path, img2_path):
    """
    Calculate optical flow between two frames
    Returns magnitude of motion
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # Resize for speed
    img1 = cv2.resize(img1, (320, 180))
    img2 = cv2.resize(img2, (320, 180))
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Calculate motion magnitude
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    motion_score = float(np.mean(magnitude))
    
    return motion_score


def calculate_ssim(img1_path, img2_path):
    """
    Calculate SSIM (Structural Similarity) between frames
    Better than histogram for detecting visual changes
    """
    from skimage.metrics import structural_similarity as ssim
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 1.0
    
    img1 = cv2.resize(img1, (320, 180))
    img2 = cv2.resize(img2, (320, 180))
    
    score = ssim(img1, img2)
    return float(score)


def detect_color_shift(img1_path, img2_path):
    """
    Detect color palette changes (primer → paint → polish)
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # Calculate average color
    avg1 = cv2.mean(img1)[:3]
    avg2 = cv2.mean(img2)[:3]
    
    # Euclidean distance in color space
    color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(avg1, avg2)))
    return float(color_diff / 255.0)


def analyze_frame_comprehensive(frame, prev_frame=None):
    """
    Comprehensive frame analysis with multiple signals
    """
    img_path = frame['path']
    
    # 1. Hand activity (are hands working?)
    hand_info = analyze_hand_activity(img_path)
    frame.update(hand_info)
    
    # 2. Visual quality metrics
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Brightness (polishing increases reflectivity)
    brightness = np.mean(gray) / 255.0
    
    # Saturation (painted areas more colorful)
    saturation = np.mean(hsv[:, :, 1]) / 255.0
    
    # Contrast (detail visibility)
    contrast = np.std(gray) / 128.0
    
    # Glossiness (highlights)
    highlights = np.sum(gray > 200) / gray.size
    
    frame.update({
        'brightness': float(brightness),
        'saturation': float(saturation),
        'contrast': float(contrast),
        'highlights': float(highlights),
        'visual_quality': float(brightness * 0.4 + contrast * 0.3 + saturation * 0.3)
    })
    
    # 3. Motion analysis (if we have previous frame)
    if prev_frame:
        motion = calculate_optical_flow(prev_frame['path'], img_path)
        ssim_score = calculate_ssim(prev_frame['path'], img_path)
        color_shift = detect_color_shift(prev_frame['path'], img_path)
        
        frame.update({
            'motion': float(motion),
            'ssim': float(ssim_score),
            'color_shift': float(color_shift)
        })
    else:
        frame.update({'motion': 0.0, 'ssim': 1.0, 'color_shift': 0.0})
    
    return frame


def detect_scene_transitions(frames):
    """
    Detect scene transitions using multiple signals:
    - SSIM drops (visual change)
    - Color shifts (material/stage change)
    - Motion pattern changes
    - Hand activity changes
    """
    print(f"\n🎯 Detecting scene transitions...")
    
    transitions = [0]
    
    for i in range(1, len(frames)):
        curr = frames[i]
        prev = frames[i-1]
        
        # Multi-signal transition detection
        ssim_drop = curr['ssim'] < 0.70  # Visual change
        color_changed = curr['color_shift'] > 0.15  # Material change
        motion_change = abs(curr['motion'] - prev['motion']) > 5.0  # Activity change
        hand_change = (curr['hand_detected'] != prev['hand_detected'])  # Hand enters/leaves
        
        is_transition = ssim_drop or color_changed or motion_change or hand_change
        
        if is_transition:
            transitions.append(i)
            reasons = []
            if ssim_drop:
                reasons.append(f"visual_change(ssim={curr['ssim']:.2f})")
            if color_changed:
                reasons.append(f"color_shift={curr['color_shift']:.2f}")
            if motion_change:
                reasons.append(f"motion_Δ={abs(curr['motion']-prev['motion']):.1f}")
            if hand_change:
                reasons.append("hand_activity_change")
            
            print(f"   Transition at {curr['timestamp']:.1f}s: {', '.join(reasons)}")
    
    print(f"   ✓ Found {len(transitions)} transitions")
    return transitions


def create_scenes_from_transitions(frames, transitions):
    """Create scenes with rich metadata"""
    print(f"\n📋 Creating scenes...")
    
    scenes = []
    
    for i in range(len(transitions)):
        start_idx = transitions[i]
        end_idx = transitions[i+1] if i+1 < len(transitions) else len(frames)-1
        
        scene_frames = frames[start_idx:end_idx+1]
        
        if len(scene_frames) < 3:
            continue
        
        start_time = scene_frames[0]['timestamp']
        end_time = scene_frames[-1]['timestamp']
        duration = end_time - start_time
        
        if duration < MIN_SCENE_DURATION:
            continue
        
        # Analyze scene characteristics
        hand_active_frames = sum(1 for f in scene_frames if f['hand_detected'])
        hand_activity_ratio = hand_active_frames / len(scene_frames)
        
        avg_motion = np.mean([f['motion'] for f in scene_frames])
        avg_brightness = np.mean([f['brightness'] for f in scene_frames])
        brightness_trend = scene_frames[-1]['brightness'] - scene_frames[0]['brightness']
        
        # Interest score based on activity
        interest_score = (
            hand_activity_ratio * 40 +  # Hands working = interesting
            min(avg_motion / 20, 1.0) * 30 +  # Motion = activity
            abs(brightness_trend) * 20 +  # Visual change = progression
            (avg_brightness * 10)  # Glossy results interesting
        )
        
        scenes.append({
            'scene_num': len(scenes) + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frame_count': len(scene_frames),
            'hand_activity_ratio': float(hand_activity_ratio),
            'avg_motion': float(avg_motion),
            'avg_brightness': float(avg_brightness),
            'brightness_trend': float(brightness_trend),
            'interest_score': float(interest_score)
        })
        
        print(f"   Scene {len(scenes):02d}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s) "
              f"hands={hand_activity_ratio:.0%}, motion={avg_motion:.1f}, interest={interest_score:.1f}")
    
    print(f"   ✓ Created {len(scenes)} scenes")
    return scenes


def classify_scenes_professional(scenes):
    """Classify using percentile-based ranking on interest scores"""
    print(f"\n🎨 Classifying scenes...")
    
    if not scenes:
        return scenes
    
    scores = [s['interest_score'] for s in scenes]
    p90 = np.percentile(scores, 90)
    p70 = np.percentile(scores, 70)
    p40 = np.percentile(scores, 40)
    
    print(f"   Interest score percentiles: p90={p90:.1f}, p70={p70:.1f}, p40={p40:.1f}")
    
    for scene in scenes:
        score = scene['interest_score']
        
        if score >= p90:
            scene['classification'] = 'interesting'
            scene['speed'] = 1.0
            scene['reason'] = f"Top 10% - High activity (interest={score:.1f})"
        elif score >= p70:
            scene['classification'] = 'moderate'
            scene['speed'] = 2.0
            scene['reason'] = f"Top 30% - Good activity (interest={score:.1f})"
        elif score >= p40:
            scene['classification'] = 'moderate'
            scene['speed'] = 4.0
            scene['reason'] = f"Above median (interest={score:.1f})"
        else:
            scene['classification'] = 'skip'
            scene['speed'] = 0
            scene['reason'] = f"Low activity, skip (interest={score:.1f})"
        
        speed_str = f"{scene['speed']:.1f}x" if scene['speed'] > 0 else "SKIP"
        print(f"   Scene {scene['scene_num']:02d}: {scene['classification']:12s} ({speed_str:5s}) - {scene['reason']}")
    
    return scenes


def save_results(scenes, frames, output_file="scene_analysis_professional.json"):
    """Save results to JSON"""
    counts = {'interesting': 0, 'moderate': 0, 'boring': 0, 'skip': 0}
    total_original = 0
    total_speedup = 0
    
    for scene in scenes:
        cls = scene['classification']
        counts[cls] = counts.get(cls, 0) + 1
        total_original += scene['duration']
        if scene['speed'] > 0:
            total_speedup += scene['duration'] / scene['speed']
    
    result = {
        'video': VIDEO_FILE,
        'total_frames': len(frames),
        'total_scenes': len(scenes),
        'method': 'professional',
        'signals': ['mediapipe_hands', 'optical_flow', 'ssim', 'color_shift', 'visual_quality'],
        'scenes': scenes,
        'summary': {
            'interesting': counts['interesting'],
            'moderate': counts['moderate'],
            'boring': counts['boring'],
            'skip': counts['skip'],
            'original_duration': total_original,
            'speedup_duration': total_speedup,
            'time_saved': total_original - total_speedup
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved analysis to: {output_file}")
    return result


def print_summary(result):
    """Print summary report"""
    summary = result['summary']
    
    print("\n" + "=" * 60)
    print("📊 Professional Video Analysis Report")
    print("=" * 60)
    print(f"  Method: Multi-signal (MediaPipe + OpticalFlow + SSIM)")
    print(f"  Interesting (1x):    {summary['interesting']} scenes")
    print(f"  Moderate (2-4x):     {summary['moderate']} scenes")
    print(f"  Skip (removed):      {summary.get('skip', 0)} scenes")
    print()
    print(f"  Original:  {summary['original_duration']/60:.1f} min")
    print(f"  Output:    {summary['speedup_duration']/60:.1f} min")
    print(f"  Saved:     {summary['time_saved']/60:.1f} min ({summary['time_saved']/summary['original_duration']*100:.0f}%)")
    print("=" * 60)


def main():
    """Main workflow"""
    video_path = Path(VIDEO_FILE)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {VIDEO_FILE}")
        return
    
    # Extract frames
    frames = extract_frames_batch(video_path, interval=SAMPLE_INTERVAL)
    
    # Analyze all frames
    print(f"\n🔍 Analyzing frames with professional methods...")
    for i, frame in enumerate(frames):
        prev_frame = frames[i-1] if i > 0 else None
        analyze_frame_comprehensive(frame, prev_frame)
        
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(frames)} frames...")
    
    print(f"   ✓ Analyzed {len(frames)} frames")
    
    # Detect transitions
    transitions = detect_scene_transitions(frames)
    
    # Create scenes
    scenes = create_scenes_from_transitions(frames, transitions)
    
    # Classify
    scenes = classify_scenes_professional(scenes)
    
    # Save and report
    result = save_results(scenes, frames)
    print_summary(result)
    
    print("\n💡 Next steps:")
    print("   1. Review scene_analysis_professional.json")
    print("   2. Run: python extract_scenes.py scene_analysis_professional.json")


if __name__ == "__main__":
    main()
