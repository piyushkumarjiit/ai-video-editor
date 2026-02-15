#!/usr/bin/env python3
"""
AI-Powered Video Analysis - Analyze Original Video with Semantic Understanding
Uses CLIP to understand what's happening in scale modeling:
- Starting work (picking up tools, preparing)
- Applying (painting, polishing, gluing, assembly)
- Finishing (shiny results, reflections, completed work)
Filters out: shaky footage, repetitive actions, irrelevant sections
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configuration
VIDEO_FILE = "IMG_3839.MOV"
SAMPLE_INTERVAL = 2  # Extract 1 frame every 2 seconds (faster for prototype)
MIN_SCENE_DURATION = 4  # Minimum scene length in seconds
SIMILARITY_THRESHOLD = 0.40  # ULTRA AGGRESSIVE - force splits on subtle changes
ACTIVITY_CHANGE_THRESHOLD = 0.15  # Detect when work/activity changes
SHAKE_THRESHOLD = 15  # LOWERED - was filtering 68% of frames! Focus on content, not blur
CACHE_FILE = "IMG_3839_frame_analysis_cache.pkl"  # Cache file for analyzed frames
TRIM_SHAKY_BOUNDARIES = 1  # Trim N seconds from start/end if shaky

# Scale modeling workflow concepts - detailed understanding
WORKFLOW_CONCEPTS = {
    'unboxing': [
        "opening model kit box",
        "layout of plastic model parts",
        "examining instruction booklet",
        "plastic model sprues and parts"
    ],
    'preparation': [
        "washing model parts in water",
        "sanding plastic parts",
        "trimming mold lines with knife",
        "cleaning model surfaces"
    ],
    'priming': [
        "spraying primer on model",
        "grey primer coating surface",
        "airbrush priming parts"
    ],
    'painting': [
        "spray painting model car",
        "airbrushing metallic paint",
        "painting model with brush",
        "applying thin coat of paint",
        "hand holding airbrush"
    ],
    'detailing': [
        "painting small details",
        "applying decals to model",
        "fine brush work on miniature",
        "painting interior details"
    ],
    'polishing': [
        "polishing shiny car surface",
        "buffing glossy paint",
        "using polishing compound",
        "circular polishing motion"
    ],
    'varnish': [
        "spraying clear coat",
        "applying gloss varnish",
        "wet glossy surface reflection"
    ],
    'assembly': [
        "assembling model parts together",
        "attaching wheels to model car",
        "gluing parts carefully",
        "fitting body panels"
    ],
    'final': [
        "completed glossy model car",
        "shiny reflection on painted surface",
        "close up of finished details",
        "professional model photography"
    ]
}

# Flatten all concepts for scoring
INTERESTING_CONCEPTS = [concept for concepts in WORKFLOW_CONCEPTS.values() for concept in concepts]

BORING_CONCEPTS = [
    "blurry motion",
    "out of focus image",
    "empty background only",
    "camera shake",
    "no activity visible",
    "dark unclear scene"
]

# Key moments = high priority content
KEY_MOMENT_CONCEPTS = [
    "before and after comparison",
    "glossy reflection on painted surface",
    "spray painting in action",
    "polishing creating shine",
    "completed model showcase",
    "close up of intricate details"
]

# Global CLIP model cache
_clip_model = None
_clip_processor = None


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_single_frame(args):
    """Extract a single frame (for parallel processing)"""
    i, timestamp, video_path, frames_dir = args
    output_path = frames_dir / f"frame_{i:04d}.jpg"
    
    cmd = [
        'ffmpeg', '-y', '-ss', str(timestamp),
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',  # High quality JPEG
        str(output_path)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {
        'index': i,
        'timestamp': timestamp,
        'path': str(output_path)
    }


def extract_keyframes(video_path, interval=2):
    """Extract frames at regular intervals from the original video (parallelized)"""
    print(f"\n📹 Analyzing original video: {video_path}")
    
    duration = get_video_duration(video_path)
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Create temp directory for frames
    frames_dir = Path("/tmp/original_frames")
    frames_dir.mkdir(exist_ok=True)
    
    # Extract frames every N seconds
    timestamps = np.arange(0, duration, interval)
    print(f"   Extracting {len(timestamps)} frames (1 every {interval}s) with {multiprocessing.cpu_count()} workers...")
    
    # Parallel extraction
    args_list = [(i, timestamp, str(video_path), frames_dir) for i, timestamp in enumerate(timestamps)]
    
    frames = []
    with ThreadPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as executor:
        for i, frame in enumerate(executor.map(extract_single_frame, args_list)):
            frames.append(frame)
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(timestamps)} frames...")
    
    print(f"   ✓ Extracted {len(frames)} frames")
    return frames


def detect_shake(image_path, threshold=50):
    """Detect if frame is shaky/blurry using Laplacian variance"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        # Calculate Laplacian variance (measures blur)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        # Low variance = blurry/shaky
        return variance < threshold
    except:
        return False

def analyze_frame_quality(image_path):
    """
    Analyze frame for interesting visual qualities:
    - Glossiness (bright highlights = polished surface)
    - Contrast (detail visibility)
    - Motion blur (activity level)
    - Saturation (colorful = painted)
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {}
        
        # Convert to different color spaces
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Glossiness detection - bright highlights indicate shine
        brightness = np.mean(img_gray)
        highlight_pixels = np.sum(img_gray > 200) / img_gray.size
        glossiness = highlight_pixels * (brightness / 128.0)
        
        # 2. Contrast - high contrast = good detail visibility
        contrast = np.std(img_gray) / 128.0
        
        # 3. Saturation - colorful painted surface
        saturation = np.mean(img_hsv[:, :, 1]) / 255.0
        
        # 4. Edge density - detail richness
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. Sharpness (inverse of motion blur)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 500.0, 1.0)
        
        return {
            'glossiness': float(glossiness),
            'contrast': float(contrast),
            'saturation': float(saturation),
            'edge_density': float(edge_density),
            'sharpness': float(sharpness),
            'brightness': float(brightness / 255.0)
        }
    except:
        return {}


def calculate_motion_between_frames(path1, path2):
    """
    Calculate motion/activity between two frames.
    High motion = active work (polishing, painting)
    Low motion = paused/inspecting
    """
    try:
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize for speed
        img1 = cv2.resize(img1, (320, 180))
        img2 = cv2.resize(img2, (320, 180))
        
        # Calculate frame difference
        diff = cv2.absdiff(img1, img2)
        motion_score = np.mean(diff) / 255.0
        
        return float(motion_score)
    except:
        return 0.0

def get_clip_embeddings(image_path):
    """Extract CLIP embeddings for semantic understanding"""
    global _clip_model, _clip_processor
    
    try:
        # Load CLIP model once
        if _clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n🤖 Loading CLIP model on {device}...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
        
        device = next(_clip_model.parameters()).device
        
        image = Image.open(image_path)
        inputs = _clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            vision_outputs = _clip_model.vision_model(**inputs)
            image_features = vision_outputs.pooler_output
            image_features = _clip_model.visual_projection(image_features)
            image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f"CLIP error: {e}")
        return None


def get_text_embeddings(text):
    """Get CLIP embeddings for text concepts"""
    global _clip_model, _clip_processor
    
    try:
        if _clip_model is None:
            return None
        
        device = next(_clip_model.parameters()).device
        
        inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_outputs = _clip_model.text_model(**inputs)
            text_features = text_outputs.pooler_output
            text_features = _clip_model.text_projection(text_features)
            text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    except Exception as e:
        return None


def calculate_concept_score(image_embedding, concepts):
    """Calculate how well an image matches interesting concepts"""
    if image_embedding is None:
        return 0.0
    
    scores = []
    for concept in concepts:
        text_emb = get_text_embeddings(concept)
        if text_emb is not None:
            similarity = float(np.dot(image_embedding, text_emb))
            scores.append(similarity)
    
    return max(scores) if scores else 0.0


def detect_workflow_stage(image_embedding):
    """Identify which workflow stage this frame belongs to"""
    if image_embedding is None:
        return 'unknown', 0.0
    
    best_stage = 'unknown'
    best_score = 0.0
    
    for stage_name, concepts in WORKFLOW_CONCEPTS.items():
        score = calculate_concept_score(image_embedding, concepts)
        if score > best_score:
            best_score = score
            best_stage = stage_name
    
    return best_stage, best_score


def is_key_moment(image_embedding):
    """Check if this frame is a key moment (high priority content)"""
    if image_embedding is None:
        return False, 0.0
    
    score = calculate_concept_score(image_embedding, KEY_MOMENT_CONCEPTS)
    return score > 0.28, score  # Balance: not too many (was 53%), not zero (0.32 gave 0)


def calculate_semantic_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def save_frame_cache(frames, cache_file):
    """Save analyzed frames to cache file"""
    print(f"\n💾 Saving frame analysis cache to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(frames, f)
    print(f"   ✓ Cached {len(frames)} analyzed frames")


def load_frame_cache(cache_file):
    """Load analyzed frames from cache file"""
    if not Path(cache_file).exists():
        return None
    
    print(f"\n📂 Loading frame analysis cache from: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            frames = pickle.load(f)
        print(f"   ✓ Loaded {len(frames)} analyzed frames from cache")
        return frames
    except Exception as e:
        print(f"   ⚠️  Cache load failed: {e}")
        return None


def analyze_frames(frames, use_cache=True):
    """Analyze all frames with CLIP embeddings and quality checks"""
    
    # Try to load from cache first
    if use_cache:
        cached_frames = load_frame_cache(CACHE_FILE)
        if cached_frames is not None:
            print(f"   🚀 Using cached analysis (skip with --no-cache to re-analyze)")
            return cached_frames
    
    print(f"\n🔍 Analyzing semantic content with AI + visual quality...")
    
    filtered_frames = []
    shaky_count = 0
    key_moment_count = 0
    
    for i, frame in enumerate(frames):
        # Check if frame is shaky/blurry
        is_shaky = detect_shake(frame['path'], threshold=SHAKE_THRESHOLD)
        
        if is_shaky:
            shaky_count += 1
            frame['quality'] = 'shaky'
            continue  # Skip shaky frames
        
        # Analyze visual quality (glossiness, contrast, etc.)
        quality_metrics = analyze_frame_quality(frame['path'])
        frame.update(quality_metrics)
        
        # Get embeddings
        frame['embedding'] = get_clip_embeddings(frame['path'])
        
        if frame['embedding'] is not None:
            # Calculate concept scores
            frame['interesting_score'] = calculate_concept_score(frame['embedding'], INTERESTING_CONCEPTS)
            frame['boring_score'] = calculate_concept_score(frame['embedding'], BORING_CONCEPTS)
            
            # Detect workflow stage
            stage, stage_score = detect_workflow_stage(frame['embedding'])
            frame['workflow_stage'] = stage
            frame['stage_score'] = stage_score
            
            # Check if key moment
            is_key, key_score = is_key_moment(frame['embedding'])
            frame['is_key_moment'] = is_key
            frame['key_moment_score'] = key_score
            if is_key:
                key_moment_count += 1
            
            frame['quality'] = 'good'
            filtered_frames.append(frame)
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(frames)} frames analyzed...")
    
    # Calculate motion between consecutive frames
    print(f"   Calculating motion/activity between frames...")
    for i in range(len(filtered_frames) - 1):
        motion = calculate_motion_between_frames(
            filtered_frames[i]['path'],
            filtered_frames[i+1]['path']
        )
        filtered_frames[i]['motion'] = motion
    if len(filtered_frames) > 0:
        filtered_frames[-1]['motion'] = 0.0  # Last frame has no next frame
    
    print(f"   ✓ Analyzed {len(frames)} frames")
    print(f"   ✓ Kept {len(filtered_frames)} good frames, filtered {shaky_count} shaky frames")
    print(f"   ✓ Detected {key_moment_count} key moments")
    
    # Show workflow stage distribution
    stages = {}
    for f in filtered_frames:
        stage = f.get('workflow_stage', 'unknown')
        stages[stage] = stages.get(stage, 0) + 1
    print(f"   ✓ Workflow stages: {', '.join(f'{k}={v}' for k, v in sorted(stages.items()))}")
    
    # Save to cache
    save_frame_cache(filtered_frames, CACHE_FILE)
    
    return filtered_frames


def calculate_color_histogram(frame_bgr):
    """Calculate normalized color histogram for BGR frame"""
    hist_b = cv2.calcHist([frame_bgr], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([frame_bgr], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([frame_bgr], [2], None, [32], [0, 256])
    
    # Normalize
    hist_b = hist_b.flatten() / hist_b.sum()
    hist_g = hist_g.flatten() / hist_g.sum()
    hist_r = hist_r.flatten() / hist_r.sum()
    
    return np.concatenate([hist_r, hist_g, hist_b])


def histogram_similarity(hist1, hist2):
    """Calculate histogram similarity using correlation (0-1, higher=more similar)"""
    return cv2.compareHist(
        hist1.astype(np.float32), 
        hist2.astype(np.float32), 
        cv2.HISTCMP_CORREL
    )


def detect_semantic_transitions(frames, similarity_threshold=0.85):
    """
    Detect where VISUAL CONTENT changes (color/brightness/histogram).
    CLIP embeddings are useless for detecting primer→paint→polish changes.
    Use histogram analysis instead!
    """
    print(f"\n🎯 Detecting scene transitions (histogram + activity analysis)...")
    
    transitions = [0]  # Always start at beginning
    
    # Calculate histograms for all frames
    print(f"   Computing color histograms for {len(frames)} frames...")
    for frame_data in frames:
        if 'histogram' not in frame_data:
            frame_bgr = cv2.imread(frame_data['path'])
            frame_data['histogram'] = calculate_color_histogram(frame_bgr)
    
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        # Check 1: HISTOGRAM similarity (detects color/brightness changes)
        hist_sim = histogram_similarity(prev_frame['histogram'], curr_frame['histogram'])
        
        # Check 2: CLIP semantic similarity (detects object/scene changes)
        prev_emb = prev_frame['embedding']
        curr_emb = curr_frame['embedding']
        semantic_sim = calculate_semantic_similarity(prev_emb, curr_emb)
        
        # Check 3: Activity/workflow stage change
        prev_stage = prev_frame.get('workflow_stage', 'unknown')
        curr_stage = curr_frame.get('workflow_stage', 'unknown')
        activity_changed = prev_stage != curr_stage
        
        # Check 4: Interest level change
        prev_interest = prev_frame.get('interesting_score', 0)
        curr_interest = curr_frame.get('interesting_score', 0)
        interest_change = abs(curr_interest - prev_interest)
        
        # Transition if:
        # - Histogram similarity drops (COLOR CHANGED - primer→paint!)
        # - OR semantic content changed significantly
        # - OR activity changed with big interest shift
        is_transition = (
            hist_sim < 0.65 or  # LOWERED to detect subtle polishing changes
            semantic_sim < similarity_threshold or
            (activity_changed and interest_change > ACTIVITY_CHANGE_THRESHOLD) or
            interest_change > 0.25
        )
        
        if is_transition:
            transitions.append(i)
            time_curr = curr_frame['timestamp']
            reasons = []
            if hist_sim < 0.65:
                reasons.append(f"color_change={hist_sim:.2f}")
            if semantic_sim < similarity_threshold:
                reasons.append(f"semantic={semantic_sim:.2f}")
            if activity_changed:
                reasons.append(f"{prev_stage}→{curr_stage}")
            if interest_change > 0.20:
                reasons.append(f"interest±{interest_change:.2f}")
            print(f"   Transition at {time_curr:.1f}s ({', '.join(reasons)})")
    
    print(f"   ✓ Found {len(transitions)} transitions")
    return transitions



def create_scenes(frames, transitions, min_duration=5):
    """Create scenes from transition points with semantic scoring and boundary trimming"""
    print(f"\n📋 Creating scenes (min duration: {min_duration}s, trim shaky: {TRIM_SHAKY_BOUNDARIES}s)...")
    
    scenes = []
    
    for i in range(len(transitions)):
        start_idx = transitions[i]
        end_idx = transitions[i + 1] if i + 1 < len(transitions) else len(frames) - 1
        
        start_time = frames[start_idx]['timestamp']
        end_time = frames[end_idx]['timestamp']
        
        # Trim shaky boundaries - check first/last few seconds
        trim_start = 0
        trim_end = 0
        
        # Check if first TRIM_SHAKY_BOUNDARIES seconds are shaky
        for j in range(start_idx, min(start_idx + 3, end_idx)):
            if j < len(frames):
                # If any frame in first few seconds looks bad, trim it
                if frames[j].get('interesting_score', 0) < 0.25:
                    trim_start = TRIM_SHAKY_BOUNDARIES
                    break
        
        # Check if last TRIM_SHAKY_BOUNDARIES seconds are shaky
        for j in range(max(end_idx - 3, start_idx), end_idx):
            if j < len(frames):
                if frames[j].get('interesting_score', 0) < 0.25:
                    trim_end = TRIM_SHAKY_BOUNDARIES
                    break
        
        # Apply trimming
        start_time += trim_start
        end_time -= trim_end
        duration = end_time - start_time
        
        # Skip scenes that are too short after trimming
        if duration < min_duration:
            continue
        
        # Get scene frames (after trimming)
        scene_frames = [f for f in frames[start_idx:end_idx+1] 
                       if start_time <= f['timestamp'] <= end_time]
        
        if not scene_frames:
            continue
        
        # TEMPORAL ANALYSIS - understand progression across keyframes
        # Divide scene into start/middle/end to detect change
        n_frames = len(scene_frames)
        start_segment = scene_frames[:max(1, n_frames//3)]
        middle_segment = scene_frames[n_frames//3:2*n_frames//3] if n_frames > 3 else scene_frames
        end_segment = scene_frames[2*n_frames//3:] if n_frames > 3 else scene_frames[-1:]
        
        # Compare segments to detect progression
        def segment_avg_embedding(segment):
            embeddings = [f['embedding'] for f in segment if f.get('embedding') is not None]
            if not embeddings:
                return None
            return np.mean(embeddings, axis=0)
        
        start_emb = segment_avg_embedding(start_segment)
        middle_emb = segment_avg_embedding(middle_segment)
        end_emb = segment_avg_embedding(end_segment)
        
        # Measure temporal change (start → middle → end)
        progression_change = 0.0
        if start_emb is not None and end_emb is not None:
            # How different is end from start? (visual progression)
            progression_change = 1.0 - calculate_semantic_similarity(start_emb, end_emb)
        
        # Detect reveal moment (middle → end shows big change)
        reveal_detected = False
        if middle_emb is not None and end_emb is not None:
            mid_to_end_change = 1.0 - calculate_semantic_similarity(middle_emb, end_emb)
            if mid_to_end_change > 0.15:  # Significant change at end = reveal
                reveal_detected = True
        
        # Calculate semantic scores from frames
        interesting_scores = [f.get('interesting_score', 0) for f in scene_frames]
        boring_scores = [f.get('boring_score', 0) for f in scene_frames]
        key_moments = [f for f in scene_frames if f.get('is_key_moment', False)]
        
        avg_interesting = np.mean(interesting_scores) if interesting_scores else 0
        avg_boring = np.mean(boring_scores) if boring_scores else 0
        max_interesting = max(interesting_scores) if interesting_scores else 0
        
        # Workflow stage distribution
        stages = {}
        for f in scene_frames:
            stage = f.get('workflow_stage', 'unknown')
            stages[stage] = stages.get(stage, 0) + 1
        dominant_stage = max(stages, key=stages.get) if stages else 'unknown'
        
        # Calculate visual change (diversity)
        embeddings = [f['embedding'] for f in scene_frames]
        diversity = 0.0
        if len(embeddings) > 1:
            distances = []
            for j in range(len(embeddings) - 1):
                sim = calculate_semantic_similarity(embeddings[j], embeddings[j+1])
                distances.append(1.0 - sim)
            diversity = np.mean(distances)
        
        # Calculate visual quality progression (glossiness increase = polishing working!)
        glossiness_values = [f.get('glossiness', 0) for f in scene_frames]
        contrast_values = [f.get('contrast', 0) for f in scene_frames]
        motion_values = [f.get('motion', 0) for f in scene_frames]
        
        avg_glossiness = np.mean(glossiness_values) if glossiness_values else 0
        glossiness_increase = 0.0
        if len(glossiness_values) > 3:
            # Compare first third vs last third
            start_gloss = np.mean(glossiness_values[:len(glossiness_values)//3])
            end_gloss = np.mean(glossiness_values[2*len(glossiness_values)//3:])
            glossiness_increase = end_gloss - start_gloss
        
        avg_motion = np.mean(motion_values) if motion_values else 0
        avg_contrast = np.mean(contrast_values) if contrast_values else 0
        
        scenes.append({
            'scene_num': len(scenes) + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'frame_count': len(scene_frames),
            'avg_interesting_score': avg_interesting,
            'max_interesting_score': max_interesting,
            'avg_boring_score': avg_boring,
            'diversity': diversity,
            'semantic_quality': avg_interesting - avg_boring,
            'key_moment_count': len(key_moments),
            'avg_glossiness': avg_glossiness,
            'glossiness_increase': glossiness_increase,
            'avg_motion': avg_motion,
            'avg_contrast': avg_contrast,
            'workflow_stage': dominant_stage,
            'trimmed_start': trim_start,
            'trimmed_end': trim_end,
            # NEW: Temporal analysis
            'progression_change': progression_change,
            'reveal_detected': reveal_detected
        })
        
        stage_emoji = {'painting': '🎨', 'polishing': '✨', 'assembly': '🔧', 'priming': '🖌️', 
                       'final': '🏆', 'detailing': '🔍', 'varnish': '💎'}.get(dominant_stage, '📹')
        
        reveal_mark = " [REVEAL]" if reveal_detected else ""
        
        print(f"   Scene {len(scenes):02d}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s) {stage_emoji}{dominant_stage} - "
              f"progression: {progression_change:.3f}, keys: {len(key_moments)}{reveal_mark}")
    
    print(f"   ✓ Created {len(scenes)} scenes")
    return scenes


def classify_scenes(scenes):
    """
    Classify scenes based on MICRO-CHANGES within polishing:
    - Glossiness increase (dull → shiny)
    - Motion/activity level (working vs paused)
    - Visual progression (before/after within same task)
    """
    print(f"\n🎨 Classifying scenes by micro-changes and activity...")
    
    if not scenes:
        return scenes
    
    # Analyze distributions
    qualities = [s['semantic_quality'] for s in scenes]
    glossiness_increases = [s.get('glossiness_increase', 0) for s in scenes]
    motions = [s.get('avg_motion', 0) for s in scenes]
    progressions = [s['progression_change'] for s in scenes]
    
    avg_quality = np.mean(qualities)
    std_quality = np.std(qualities) if len(qualities) > 1 else 0.1
    avg_gloss_increase = np.mean(glossiness_increases)
    avg_motion = np.mean(motions)
    avg_progression = np.mean(progressions)
    
    max_scores = [s['max_interesting_score'] for s in scenes]
    avg_max_score = np.mean(max_scores)
    
    key_moments = [s['key_moment_count'] for s in scenes]
    avg_key_moments = np.mean(key_moments) if key_moments else 0
    
    print(f"   Average semantic quality: {avg_quality:.3f} (std: {std_quality:.3f})")
    print(f"   Average glossiness increase: {avg_gloss_increase:.3f}")
    print(f"   Average motion/activity: {avg_motion:.3f}")
    print(f"   Average progression: {avg_progression:.3f}")
    print(f"   Quality range: {min(qualities):.3f} - {max(qualities):.3f}")
    print(f"   Average key moments per scene: {avg_key_moments:.1f}")
    
    # Calculate combined scores for all scenes first
    all_scores = []
    for scene in scenes:
        progression = scene['progression_change']
        reveal = scene['reveal_detected']
        motion = scene.get('avg_motion', 0)
        diversity = scene['diversity']
        key_count = scene['key_moment_count']
        quality = scene['semantic_quality']
        max_interest = scene['max_interesting_score']
        
        # Simple scoring: what does AI think is interesting?
        score = (
            quality * 10 +                  # Main: semantic interestingness
            max_interest * 8 +              # Peak interesting moments
            progression * 5 +               # Visual change
            motion * 3 +                    # Activity/motion
            diversity * 3 +                 # Variety
            min(key_count / 20, 1.0) * 2 +  # Key moments (capped)
            (1.0 if reveal else 0)          # Reveal bonus
        )
        all_scores.append(score)
        scene['_combined_score'] = score
    
    # Use PERCENTILES to classify relatively
    p90 = np.percentile(all_scores, 90)  # Top 10%
    p70 = np.percentile(all_scores, 70)  # Top 30%
    p40 = np.percentile(all_scores, 40)  # Above median
    
    print(f"   Score percentiles: p90={p90:.2f}, p70={p70:.2f}, p40={p40:.2f}")
    
    for scene in scenes:
        score = scene['_combined_score']
        stage = scene['workflow_stage']
        progression = scene['progression_change']
        motion = scene.get('avg_motion', 0)
        duration = scene['duration']
        quality = scene['semantic_quality']
        
        # SIMPLE PERCENTILE-BASED CLASSIFICATION
        # Goal: 17 min → 3-5 min (70-80% reduction)
        # Just rank by AI score and cut accordingly
        
        # Top 10% = most interesting (keep at 1x)
        if score >= p90:
            scene['classification'] = 'interesting'
            scene['speed'] = 1.0
            reason = f"{stage}: Best moments (top 10%, quality={quality:.2f})"
        
        # Top 30% = good (2x speedup)
        elif score >= p70:
            scene['classification'] = 'moderate'
            scene['speed'] = 2.0
            reason = f"{stage}: Good (top 30%, quality={quality:.2f})"
        
        # Above median = keep but speed up (4x)
        elif score >= p40:
            scene['classification'] = 'moderate'
            scene['speed'] = 4.0
            reason = f"{stage}: Average (quality={quality:.2f})"
        
        # Below median = skip entirely  
        else:
            scene['classification'] = 'skip'
            scene['speed'] = 0
            reason = f"{stage}: Low interest, skipped (quality={quality:.2f})"
        
        scene['reason'] = reason
        
        speed_str = f"{scene['speed']:.1f}x" if scene['speed'] > 1 else "1.0x" if scene['speed'] > 0 else "SKIP"
        print(f"   Scene {scene['scene_num']:02d}: {scene['classification']:12s} ({speed_str:5s}) - {reason}")
    
    return scenes


def save_results(scenes, frames, output_file="scene_analysis_smart.json"):
    """Save analysis results to JSON"""
    
    # Count classifications (including skip)
    counts = {'interesting': 0, 'moderate': 0, 'boring': 0, 'skip': 0}
    total_original = 0
    total_speedup = 0
    
    for scene in scenes:
        cls = scene['classification']
        counts[cls] = counts.get(cls, 0) + 1
        total_original += scene['duration']
        if scene['speed'] > 0:
            total_speedup += scene['duration'] / scene['speed']
        # else: skip = 0 duration added
    
    result = {
        'video': VIDEO_FILE,
        'total_frames': len(frames),
        'total_scenes': len(scenes),
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
    print("📊 AI Scene Analysis Report")
    print("=" * 60)
    print(f"  Interesting (1x):    {summary['interesting']} scenes")
    print(f"  Moderate (2.5-4x):   {summary['moderate']} scenes")
    print(f"  Boring (5x):         {summary['boring']} scenes")
    print(f"  Skip (removed):      {summary.get('skip', 0)} scenes")
    print()
    print(f"  Original:  {summary['original_duration']/60:.1f} min")
    print(f"  Speed-up:  {summary['speedup_duration']/60:.1f} min")
    print(f"  Saved:     {summary['time_saved']/60:.1f} min ({summary['time_saved']/summary['original_duration']*100:.0f}%)")
    print("=" * 60)
    print()


def main():
    """Main analysis workflow"""
    video_path = Path(VIDEO_FILE)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {VIDEO_FILE}")
        return
    
    # Step 1: Extract keyframes from original video
    frames = extract_keyframes(video_path, interval=SAMPLE_INTERVAL)
    
    # Step 2: Analyze frames with CLIP
    frames = analyze_frames(frames)
    
    # Step 3: Detect semantic transitions
    transitions = detect_semantic_transitions(frames, similarity_threshold=SIMILARITY_THRESHOLD)
    
    # Step 4: Create scenes from transitions
    scenes = create_scenes(frames, transitions, min_duration=MIN_SCENE_DURATION)
    
    # Step 5: Classify scenes by AI semantic analysis
    scenes = classify_scenes(scenes)
    
    # Step 6: Save and report
    result = save_results(scenes, frames)
    print_summary(result)
    
    print("\n💡 Next steps:")
    print("   1. Review scene_analysis_smart.json")
    print("   2. Run: python extract_scenes.py (to extract the classified scenes)")


if __name__ == "__main__":
    main()
