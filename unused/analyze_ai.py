#!/usr/bin/env python3
"""
AI-powered scene analyzer using vision models and LLM.
Uses CLIP embeddings for semantic understanding + optional LLM for decisions.
"""
import subprocess
import json
import sys
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

INPUT_DIR = "clips"
OUTPUT_DIR = "clips_speed"
ANALYSIS_JSON = "scene_analysis_ai.json"

# AI Model configuration
USE_AI_ANALYSIS = True  # Set to False to use simple threshold mode
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Or set your API key
USE_LOCAL_MODEL = True  # Use local CLIP instead of API calls

# Simple thresholds (fallback if AI disabled)
MIN_SCENE_DURATION = 10
SIMILARITY_SKIP_THRESHOLD = 92


def extract_key_frames(video_path):
    """Extract frames at key positions"""
    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True, timeout=10
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    
    # Create temp dir for this scene (don't auto-delete)
    temp_dir = Path(f"/tmp/scene_frames_{Path(video_path).stem}")
    temp_dir.mkdir(exist_ok=True)
    
    frames = []
    for pct, name in [(0.1, "start"), (0.5, "mid"), (0.9, "end")]:
        t = duration * pct
        out = temp_dir / f"{name}.jpg"
        
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
            "-frames:v", "1", "-vf", "scale=640:-1", "-q:v", "2", str(out)
        ], capture_output=True, timeout=15)
        
        if out.exists():
            img = Image.open(out).convert('RGB')
            frames.append((name, np.array(img), str(out)))
    
    return frames, duration


# Global CLIP model cache
_clip_model = None
_clip_processor = None

def get_clip_embeddings(image_path):
    """Get CLIP embeddings for semantic understanding (uses GPU)"""
    global _clip_model, _clip_processor
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        # Load CLIP model once and cache (moves to GPU if available)
        if _clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading CLIP model on {device}...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()  # Set to eval mode
        
        device = next(_clip_model.parameters()).device
        
        image = Image.open(image_path)
        inputs = _clip_processor(images=image, return_tensors="pt")
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get vision model output and extract the pooled features
            vision_outputs = _clip_model.vision_model(**inputs)
            image_features = vision_outputs.pooler_output  # Extract the pooled tensor
            # Apply the visual projection layer
            image_features = _clip_model.visual_projection(image_features)
            # Normalize embeddings using torch.norm on the tensor
            image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f"CLIP error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_vision_llm(frames_data):
    """Use vision LLM to understand what's happening in the scene"""
    try:
        import base64
        from openai import OpenAI
        
        if not OPENAI_API_KEY:
            return None
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare images
        messages = [{
            "role": "system",
            "content": "You are analyzing scale model building workflow videos. Identify what activity is shown and rate progress/interest."
        }]
        
        # Add images to prompt
        image_content = []
        for name, img, path in frames_data[:2]:  # Use start and end
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        image_content.append({
            "type": "text",
            "text": """Analyze these frames from a scale modeling video. 

Compare start to end and determine:
1. Activity type (painting, polishing, assembly, etc)
2. Visible progress (0-100): How much changed?
3. Interest level (interesting/moderate/boring): Is this showing new results or repetitive work?
4. Brief reason

Respond in JSON format:
{
  "activity": "...",
  "progress_score": 0-100,
  "interest": "interesting|moderate|boring",
  "reason": "..."
}"""
        })
        
        messages.append({"role": "user", "content": image_content})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"LLM error: {e}")
        return None


def calculate_semantic_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    if emb1 is None or emb2 is None:
        return 0.0
    dot = np.dot(emb1, emb2)
    return float(dot) * 100  # Convert to percentage


def analyze_scene_ai(clip_path):
    """AI-powered scene analysis"""
    print(f"Analyzing {clip_path.name}...", end=" ", flush=True)
    
    try:
        frames, duration = extract_key_frames(str(clip_path))
        
        if len(frames) < 2:
            print("SKIP (no frames)")
            return None
        
        # Get CLIP embeddings for semantic understanding
        embeddings = {}
        for name, img, path in frames:
            emb = get_clip_embeddings(path)
            if emb is not None:
                embeddings[name] = emb
        
        # Calculate semantic change (start to end)
        semantic_change = 0
        if "start" in embeddings and "end" in embeddings:
            # Inverse similarity = change
            similarity = calculate_semantic_similarity(embeddings["start"], embeddings["end"])
            semantic_change = 100 - similarity
        
        # Optional: Use LLM for deeper understanding
        llm_result = None
        if USE_AI_ANALYSIS and OPENAI_API_KEY:
            llm_result = analyze_with_vision_llm(frames)
        
        # Combine AI insights
        if llm_result:
            progress_score = llm_result["progress_score"]
            category = llm_result["interest"]
            reason = f"{llm_result['activity']}: {llm_result['reason']}"
        else:
            # Use CLIP-based semantic change
            progress_score = semantic_change
            if semantic_change > 30:
                category = "interesting"
            elif semantic_change > 15:
                category = "moderate"
            else:
                category = "boring"
            reason = f"semantic change: {semantic_change:.1f}%"
        
        print(f"✓ {category} (progress={progress_score:.0f}) - {reason}")
        
        return {
            "file": clip_path.name,
            "duration": round(duration, 2),
            "progress_score": round(progress_score, 2),
            "semantic_change": round(semantic_change, 2),
            "category": category,
            "reason": reason,
            "embeddings": {k: v.tolist() for k, v in embeddings.items()},
            "llm_analysis": llm_result
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_and_assign_speed(results):
    """Assign speeds based on AI classifications"""
    for result in results:
        category = result["category"]
        duration = result["duration"]
        progress = result["progress_score"]
        
        # AI-informed speed assignment
        if category == "interesting":
            speed = 1.0
        elif category == "moderate":
            if progress > 20:
                speed = 1.5
            else:
                speed = 2.5
        else:  # boring
            if duration > 120:
                speed = 4.0
            else:
                speed = 3.0
        
        result["recommended_speed"] = speed


def analyze_all_scenes():
    """Analyze all scenes with AI"""
    clips_dir = Path(INPUT_DIR)
    clip_files = sorted(clips_dir.glob("scene_*.mkv"))
    
    if not clip_files:
        print(f"No scene clips found in {INPUT_DIR}/")
        return []
    
    print(f"🤖 AI-Powered Analysis - {len(clip_files)} clips")
    if USE_AI_ANALYSIS:
        print(f"Using: CLIP embeddings + {'LLM' if OPENAI_API_KEY else 'semantic similarity'}\n")
    else:
        print("Using: Simple threshold mode\n")
    
    results = []
    
    for clip_path in clip_files:
        result = analyze_scene_ai(clip_path)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No scenes analyzed")
        return []
    
    # Classify and assign speeds
    classify_and_assign_speed(results)
    
    # Find duplicates using semantic embeddings
    print("\n🔍 Detecting semantic duplicates...")
    for i, r1 in enumerate(results):
        if "embeddings" not in r1 or "mid" not in r1["embeddings"]:
            continue
            
        max_sim = 0
        for j, r2 in enumerate(results):
            if i >= j or "embeddings" not in r2 or "mid" not in r2["embeddings"]:
                continue
            
            emb1 = np.array(r1["embeddings"]["mid"])
            emb2 = np.array(r2["embeddings"]["mid"])
            sim = calculate_semantic_similarity(emb1, emb2)
            
            if sim > max_sim:
                max_sim = sim
        
        r1["max_similarity_to_others"] = round(max_sim, 2)
        
        # Skip near-duplicates
        if max_sim > SIMILARITY_SKIP_THRESHOLD and r1["progress_score"] < 20:
            r1["category"] = "skip"
            r1["recommended_speed"] = 0.0
            r1["reason"] += f" (duplicate, sim={max_sim:.0f}%)"
    
    return results


def apply_speed_filter(input_path, output_path, speed):
    """Re-encode with GPU acceleration"""
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
    
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", input_path,
        "-filter:v", video_filter,
        "-filter:a", audio_filter,
        "-c:v", "hevc_nvenc",
        "-preset", "p4",
        "-cq", "28",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        subprocess.run(cmd, check=True)


def main():
    apply_speed = "--apply" in sys.argv
    
    results = analyze_all_scenes()
    
    if not results:
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
    
    print(f"\n{'='*60}")
    print(f"📊 AI Analysis Report: {ANALYSIS_JSON}")
    print(f"{'='*60}")
    print(f"  Interesting (1x): {len(interesting)} scenes")
    print(f"  Moderate (1.5-2.5x): {len(moderate)} scenes")
    print(f"  Boring (3-4x): {len(boring)} scenes")
    print(f"  Skipped: {len(skipped)} scenes")
    print(f"\n  Original:  {total_duration/60:.1f} min")
    print(f"  Speed-up:  {sped_up_duration/60:.1f} min")
    print(f"  Saved:     {(total_duration - sped_up_duration)/60:.1f} min")
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
    else:
        print(f"\n💡 To apply: python analyze_ai.py --apply")


if __name__ == "__main__":
    main()
