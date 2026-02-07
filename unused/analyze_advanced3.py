#!/usr/bin/env python3
"""
Advanced AI Video Analysis v3 - VISION-LANGUAGE MODELS + SEMANTIC UNDERSTANDING
Uses cutting-edge GPU-accelerated deep learning for intelligent scene analysis:

1. **CLIP (ViT-B/32)** - Vision-language semantic understanding (GPU)
2. **ResNet-50** - Deep feature extraction for similarity analysis (GPU)
3. **Duplicate Detection** - Cosine similarity to find repetitive/boring scenes
4. **Semantic Prompts** - AI understands "interesting work" vs "boring repetition"
5. **Temporal Analysis** - Detects workflow progression over time

Intelligently detects:
- Meaningful work progression (paste application, polishing action)
- Boring repetition (same motion over and over)
- Duplicate scenes (identical content)
- Low-quality frames (blurry, static, unfocused)
"""

import argparse
import subprocess
import json
import csv
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from sklearn.metrics.pairwise import cosine_similarity
import warnings
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import gc
warnings.filterwarnings('ignore')

# Configuration
VIDEO_FILE = "IMG_3839.MOV"
SAMPLE_INTERVAL = 2
OUTPUT_FILE = "scene_analysis_advanced3.json"
TARGET_OUTPUT_RATIO = 0.15  # Aim for ~10-15% output duration
MAX_SPEED_MULTIPLIER = 8.0

KEYFRAME_PROMPTS = [
    {"key": "applying", "text": "applying polishing compound to model car surface"},
    {"key": "shine", "text": "visible reflection improvement and shine"},
    {"key": "detail", "text": "close-up macro of surface detail and texture"},
    {"key": "hands", "text": "hands working with polishing tool or cloth"},
    {"key": "angle", "text": "changing angle or view of the model car"},
    {"key": "comparison", "text": "before and after surface comparison"},
    {"key": "repetitive", "text": "repetitive unchanging polishing circular motion"},
    {"key": "static", "text": "static scene with no visible activity or change"},
    {"key": "blurry", "text": "blurry out of focus unclear image"}
]

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

print("🚀 Advanced AI Video Analysis v3 - VISION-LANGUAGE + GPU")
print("=" * 70)
print("Models: CLIP (semantic), ResNet-50 (features), Duplicate Detection")
print("=" * 70)

# Check GPU
if torch.cuda.is_available():
    GPU_DEVICE = 'cuda'
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💎 GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
    print(f"   CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
else:
    GPU_DEVICE = 'cpu'
    print("\n⚠️  No GPU - using CPU (will be slow)")


def get_video_info(video_path):
    """Get video metadata"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,duration',
           '-of', 'json', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    
    fps_str = stream['r_frame_rate'].split('/')
    fps = int(fps_str[0]) / int(fps_str[1])
    
    return {
        'duration': float(stream['duration']),
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': fps
    }


def load_project_config(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def load_clip_model():
    """Load CLIP for semantic scene understanding"""
    print("\n🤖 Loading CLIP vision-language model...")
    try:
        import clip
        
        model, preprocess = clip.load("ViT-B/32", device=GPU_DEVICE)
        model.eval()
        
        mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
        print(f"   ✓ CLIP ViT-B/32 loaded on {GPU_DEVICE.upper()}")
        print(f"   GPU memory: {mem:.2f}GB")
        
        return {'model': model, 'preprocess': preprocess, 'clip': clip}
    except ImportError:
        print("   ⚠️  CLIP not installed")
        print("   Install: pip install git+https://github.com/openai/CLIP.git")
        return None


def load_resnet_model():
    """Load ResNet-50 for deep feature extraction"""
    print("\n🤖 Loading ResNet-50 feature extractor...")
    
    try:
        weights = ResNet50_Weights.IMAGENET1K_V2
    except Exception:
        weights = ResNet50_Weights.IMAGENET1K_V1
    
    try:
        model = models.resnet50(weights=weights)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        model = model.to(GPU_DEVICE)
        model.eval()
        
        transform = weights.transforms()
        
        mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
        print(f"   ✓ ResNet-50 loaded on {GPU_DEVICE.upper()} (ImageNet weights)")
        print(f"   GPU memory: {mem:.2f}GB")
        
        return {'model': model, 'transform': transform}
    except Exception as e:
        print(f"   ⚠️  ResNet-50 failed to load with weights: {e}")
        
        try:
            model = models.resnet50(weights=None)
            model = nn.Sequential(*list(model.children())[:-1])
            model = model.to(GPU_DEVICE)
            model.eval()
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
            print(f"   ✓ ResNet-50 loaded on {GPU_DEVICE.upper()} (random weights)")
            print(f"   GPU memory: {mem:.2f}GB")
            
            return {'model': model, 'transform': transform}
        except Exception as e2:
            print(f"   ⚠️  ResNet-50 failed to load: {e2}")
            return None


def load_object_detection_model():
    """Load Faster R-CNN for object detection."""
    print("\n🤖 Loading object detection model (Faster R-CNN)...")
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model = model.to(GPU_DEVICE)
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
        print(f"   ✓ Faster R-CNN loaded on {GPU_DEVICE.upper()}")
        print(f"   GPU memory: {mem:.2f}GB")
        return {'model': model, 'weights': weights}
    except Exception as e:
        print(f"   ⚠️  Object detection failed to load: {e}")
        return None


def load_qwen_vl_model(model_name, device=None, torch_dtype=None, quantization=None):
    """Load Qwen2.5-VL vision-language model for image understanding."""
    print("\n🤖 Loading Qwen2.5-VL vision-language model...")
    try:
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except Exception:
            Qwen2_5_VLForConditionalGeneration = None
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None
    except ImportError:
        print("   ⚠️  transformers not installed")
        print("   Install: pip install transformers sentencepiece")
        return None

    try:
        device = device or GPU_DEVICE
        quantization = (quantization or "").lower().strip()
        load_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
        use_quant = device == "cuda" and quantization in {"8bit", "4bit"} and BitsAndBytesConfig is not None
        if use_quant:
            if quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.update({"quantization_config": quant_config, "device_map": {"": 0}})

        if Qwen2_5_VLForConditionalGeneration is not None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        if not use_quant:
            model = model.to(device)
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1024**3 if device == 'cuda' else 0
        print(f"   ✓ Qwen2.5-VL loaded on {device.upper()} ({model_name})")
        if device == 'cuda':
            print(f"   GPU memory: {mem:.2f}GB")
        return {"model": model, "processor": processor, "model_name": model_name, "device": device, "kind": "qwen2.5-vl"}
    except Exception as e:
        print(f"   ⚠️  Qwen2.5-VL failed to load: {e}")
        return None


def load_caption_model(model_name="Salesforce/blip-image-captioning-large", device=None, torch_dtype=None, quantization=None):
    """Load image captioning model (BLIP/BLIP-2/Qwen2.5-VL)."""
    if "qwen2.5-vl" in (model_name or "").lower() or "qwen2-5-vl" in (model_name or "").lower():
        return load_qwen_vl_model(model_name, device=device, torch_dtype=torch_dtype, quantization=quantization)

    print("\n🤖 Loading image captioning model (BLIP)...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None
    except ImportError:
        print("   ⚠️  transformers not installed")
        print("   Install: pip install transformers sentencepiece")
        return None
    try:
        device = device or GPU_DEVICE
        is_blip2 = "blip2" in model_name.lower()
        quantization = (quantization or "").lower().strip()

        load_kwargs = {"low_cpu_mem_usage": True}
        use_quant = device == "cuda" and quantization in {"8bit", "4bit"} and BitsAndBytesConfig is not None
        if use_quant:
            if quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.update({"quantization_config": quant_config, "device_map": {"": 0}})

        if is_blip2:
            processor = Blip2Processor.from_pretrained(model_name)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
        else:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )

        if not use_quant:
            model = model.to(device)
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1024**3 if device == 'cuda' else 0
        print(f"   ✓ Captioning model loaded on {device.upper()} ({model_name})")
        if device == 'cuda':
            print(f"   GPU memory: {mem:.2f}GB")
        return {"model": model, "processor": processor, "model_name": model_name, "device": device}
    except Exception as e:
        print(f"   ⚠️  Captioning model failed to load: {e}")
        return None


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def release_model_dict(model_dict):
    if not model_dict:
        return
    try:
        model = model_dict.get("model")
        if model is not None:
            del model
    except Exception:
        pass
    try:
        for k in list(model_dict.keys()):
            model_dict[k] = None
    except Exception:
        pass


def caption_keyframes(frames, caption_model, max_length=40, num_beams=5, prompt=None, use_prompt_for_blip=False, min_length=None, max_image_size=None):
    """Generate captions for keyframes."""
    if caption_model is None:
        for frame in frames:
            frame["caption"] = None
        return frames

    model = caption_model["model"]
    processor = caption_model["processor"]
    model_name = caption_model.get("model_name", "").lower()
    device = caption_model.get("device", GPU_DEVICE)
    model_kind = caption_model.get("kind", "")

    print("\n📝 Captioning kept keyframes...")
    iterator = frames
    if tqdm is not None:
        iterator = tqdm(frames, desc="Captioning", unit="frame")
    with torch.no_grad():
        for idx, frame in enumerate(iterator):
            img = Image.open(frame["path"]).convert("RGB")
            if model_kind == "qwen2.5-vl" or "qwen2.5-vl" in model_name or "qwen2-5-vl" in model_name:
                if max_image_size:
                    max_dim = max(img.size)
                    if max_dim > max_image_size:
                        scale = max_image_size / float(max_dim)
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        img = img.resize(new_size, Image.BICUBIC)
                cap_prompt = prompt or "Describe the image in detail with concrete objects and actions."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": cap_prompt},
                        ],
                    }
                ]
                if hasattr(processor, "apply_chat_template"):
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[img], return_tensors="pt")
                else:
                    inputs = processor(images=img, text=cap_prompt, return_tensors="pt")
                    text = None
                inputs = {k: v.to(device) for k, v in inputs.items()}
                gen_kwargs = {"max_new_tokens": max_length, "do_sample": False}
                output_ids = model.generate(**inputs, **gen_kwargs)
                decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                if text and decoded.startswith(text):
                    caption = decoded[len(text):].strip()
                else:
                    caption = decoded.strip()
                if "assistant" in caption:
                    caption = caption.split("assistant")[-1].strip("\n :")
                if caption.lower().startswith("you are a helpful assistant"):
                    caption = caption.split("\n", 1)[-1].strip()
            else:
                use_prompt = bool(prompt) and ("blip2" in model_name or use_prompt_for_blip)
                if use_prompt:
                    inputs = processor(images=img, text=prompt, return_tensors="pt")
                else:
                    inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
                if min_length is not None:
                    gen_kwargs["min_length"] = min_length
                output_ids = model.generate(**inputs, **gen_kwargs)
                caption = processor.decode(output_ids[0], skip_special_tokens=True)
                if use_prompt and caption.lower().startswith(prompt.lower()):
                    caption = caption[len(prompt):].lstrip(" :,-")

            frame["caption"] = caption
            frame["caption_model"] = caption_model.get("model_name")
            frame["caption_prompt"] = prompt
            if tqdm is None and ((idx + 1) % 50 == 0 or (idx + 1) == len(frames)):
                print(f"      {idx + 1}/{len(frames)} frames captioned")

    print("   ✓ Captioning complete")
    return frames


def analyze_with_object_detection(frames, detection_model, score_threshold=0.5, max_objects=5):
    """Run object detection on frames and store labels/scores."""
    if detection_model is None:
        return frames

    model = detection_model['model']
    preprocess = detection_model['weights'].transforms()

    print("\n🔍 [OD] Object detection on keyframes...")
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            img = Image.open(frame['path']).convert('RGB')
            tensor = preprocess(img).to(GPU_DEVICE)
            output = model([tensor])[0]

            scores = output.get('scores', [])
            labels = output.get('labels', [])
            boxes = output.get('boxes', [])

            objects = []
            for score, label, box in zip(scores, labels, boxes):
                score_val = float(score)
                if score_val < score_threshold:
                    continue
                label_idx = int(label)
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label_idx)
                objects.append({
                    'label': label_name,
                    'score': score_val,
                    'box': [float(v) for v in box.tolist()]
                })
                if len(objects) >= max_objects:
                    break

            frame['objects'] = objects

            if (idx + 1) % 100 == 0:
                print(f"      {idx + 1}/{len(frames)} frames analyzed")

    print("   ✓ Object detection complete")
    return frames


def extract_frames_parallel(video_path, interval=2):
    """Extract frames in parallel"""
    print(f"\n📹 Extracting frames from: {video_path}")
    
    info = get_video_info(video_path)
    duration = info['duration']
    
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   Resolution: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
    
    video_stem = Path(video_path).stem
    frames_dir = Path.cwd() / "tmp_advanced3_frames" / video_stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    timestamps = np.arange(0, duration, interval)
    total = len(timestamps)
    
    print(f"   Extracting {total} frames in parallel (24 workers)...")
    
    def extract_frame(i, ts):
        output_path = frames_dir / f"frame_{i:04d}.jpg"
        cmd = ['ffmpeg', '-y', '-ss', str(ts), '-i', str(video_path),
               '-vframes', '1', '-q:v', '2', str(output_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {'index': i, 'timestamp': ts, 'path': str(output_path)}
    
    frames = [None] * total
    completed = 0
    lock = Lock()
    
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(extract_frame, i, ts): i for i, ts in enumerate(timestamps)}
        
        for future in as_completed(futures):
            frame = future.result()
            frames[frame['index']] = frame
            completed += 1
            if completed % 50 == 0 or completed == total:
                with lock:
                    print(f"   Progress: {completed}/{total} ({completed*100//total}%)")
    
    print(f"   ✓ Extracted {total} frames")
    return frames


def compute_dhash(image_path, hash_size=8):
    """Compute perceptual dHash for an image file."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = ''.join('1' if v else '0' for v in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def scene_hash(scene_frames):
    """Get a representative hash for a scene using its midpoint frame."""
    if not scene_frames:
        return None
    mid_idx = len(scene_frames) // 2
    return compute_dhash(scene_frames[mid_idx]['path'])


def analyze_with_clip(frames, clip_model):
    """GPU-accelerated CLIP semantic analysis"""
    print(f"\n🔍 [1/3] CLIP semantic analysis on GPU...")
    
    if clip_model is None:
        print("   ⚠️  CLIP not available, skipping")
        for f in frames:
            f['semantic_interest'] = 0.5
            f['semantic_boring'] = 0.5
        return frames
    
    # Define semantic prompts for SCALE MODEL workflow detection
    text_prompts = [p["text"] for p in KEYFRAME_PROMPTS]
    
    model = clip_model['model']
    preprocess = clip_model['preprocess']
    clip = clip_model['clip']
    
    mem_start = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
    
    with torch.no_grad():
        # Encode text prompts once
        text_tokens = clip.tokenize(text_prompts).to(GPU_DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Batch process images
        BATCH_SIZE = 32
        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(frames))
            batch_images = []
            
            for i in range(batch_start, batch_end):
                img = Image.open(frames[i]['path']).convert('RGB')
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
            
            # GPU inference
            batch_tensor = torch.stack(batch_images).to(GPU_DEVICE)
            image_features = model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate semantic similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = similarity.cpu().numpy()
            
            # Also store raw CLIP image features for visual similarity comparison
            for i, img_feat in enumerate(image_features):
                idx = batch_start + i
                frames[idx]['clip_image_features'] = img_feat.cpu().numpy()
            
            for i, score in enumerate(scores):
                idx = batch_start + i
                frames[idx]['clip_applying'] = float(score[0])      # Polishing action
                frames[idx]['clip_shine'] = float(score[1])         # Results visible
                frames[idx]['clip_detail'] = float(score[2])        # Macro detail
                frames[idx]['clip_hands'] = float(score[3])         # Active work
                frames[idx]['clip_angle'] = float(score[4])         # View change
                frames[idx]['clip_comparison'] = float(score[5])    # Progress
                frames[idx]['clip_repetitive'] = float(score[6])    # Boring
                frames[idx]['clip_static'] = float(score[7])        # Boring
                frames[idx]['clip_blurry'] = float(score[8])        # Bad quality
                
                # Combined interest: prioritize active work/detail; reduce by boredom
                interest_raw = (score[0] * 0.30 + score[1] * 0.20 + score[2] * 0.20 + 
                               score[3] * 0.20 + score[4] * 0.07 + score[5] * 0.03)
                boring = score[6] * 0.55 + score[7] * 0.35 + score[8] * 0.10
                interest = max(0.0, interest_raw - (boring * 0.50))
                frames[idx]['semantic_interest'] = float(interest)
                frames[idx]['semantic_boring'] = float(boring)
            
            if batch_end % 64 == 0:
                mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
                print(f"      {batch_end}/{len(frames)} frames (GPU: {mem:.2f}GB)")
    
    mem_peak = torch.cuda.max_memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
    print(f"   ✓ CLIP complete (peak GPU: {mem_peak:.2f}GB)")
    if GPU_DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    return frames


def extract_features(frames, resnet_model):
    """GPU-accelerated deep feature extraction"""
    print(f"\n🔍 [2/3] ResNet-50 feature extraction on GPU...")
    
    if resnet_model is None:
        print("   ⚠️  ResNet not available, using basic features")
        # Use simple color histograms as fallback
        features_list = []
        for frame in frames:
            img = cv2.imread(frame['path'])
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten() / hist.sum()
            frame['features'] = hist
            features_list.append(hist)
        print(f"   ✓ Basic feature extraction complete")
        return frames, np.array(features_list)
    
    model = resnet_model['model']
    transform = resnet_model['transform']
    
    features_list = []
    
    with torch.no_grad():
        BATCH_SIZE = 64
        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(frames))
            batch_images = []
            
            for i in range(batch_start, batch_end):
                img = Image.open(frames[i]['path']).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            
            # GPU batch inference
            batch_tensor = torch.stack(batch_images).to(GPU_DEVICE)
            features = model(batch_tensor)
            features = features.squeeze().cpu().numpy()
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            for i, feat in enumerate(features):
                frames[batch_start + i]['features'] = feat
                features_list.append(feat)
            
            if batch_end % 128 == 0:
                mem = torch.cuda.memory_allocated(0) / 1024**3 if GPU_DEVICE == 'cuda' else 0
                print(f"      {batch_end}/{len(frames)} frames (GPU: {mem:.2f}GB)")
    
    print(f"   ✓ Feature extraction complete")
    return frames, np.array(features_list)


def detect_duplicates_and_repetition(frames, features):
    """Detect duplicate and repetitive scenes using CLIP semantic + visual similarity"""
    print(f"\n🔍 [3/3] Detecting duplicates and motion analysis...")
    
    # Build CLIP semantic feature matrix
    clip_features = []
    for f in frames:
        feat = [
            f.get('clip_applying', 0),
            f.get('clip_shine', 0),
            f.get('clip_detail', 0),
            f.get('clip_hands', 0),
            f.get('clip_angle', 0),
            f.get('clip_comparison', 0),
            f.get('clip_repetitive', 0),
            f.get('clip_static', 0),
            f.get('clip_blurry', 0),
        ]
        clip_features.append(feat)
    
    clip_features = np.array(clip_features)
    clip_similarity = cosine_similarity(clip_features)
    
    # Visual similarity from ResNet features (pretrained if available)
    feature_similarity = cosine_similarity(features) if features is not None else clip_similarity
    
    # Combine semantic + visual similarity (semantic dominates, visual refines)
    combined_similarity = (clip_similarity * 0.6) + (feature_similarity * 0.4)
    
    # Calculate motion between consecutive frames
    print("   Analyzing motion and semantic similarity...")
    for i in range(len(frames)):
        # Motion detection: check combined difference with next frame
        if i < len(frames) - 1:
            motion = 1.0 - combined_similarity[i, i + 1]
            frames[i]['motion'] = float(motion)
        else:
            frames[i]['motion'] = 0.0
        
        # Check similarity to nearby frames (±5 frames = ±10 seconds)
        window_start = max(0, i - 5)
        window_end = min(len(frames), i + 6)
        
        similarities = combined_similarity[i, window_start:window_end]
        similarities = np.delete(similarities, i - window_start)  # Remove self
        
        max_sim = np.max(similarities) if len(similarities) > 0 else 0
        avg_sim = np.mean(similarities) if len(similarities) > 0 else 0
        
        frames[i]['max_similarity'] = float(max_sim)
        frames[i]['avg_similarity'] = float(avg_sim)
        
        # Duplicate: very high similarity + very low motion
        frames[i]['is_duplicate'] = (max_sim > 0.975 and frames[i]['motion'] < 0.03)
        # Repetitive: high similarity to neighbors + low motion
        frames[i]['is_repetitive'] = (avg_sim > 0.93 and frames[i]['motion'] < 0.06)
        
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(frames)} frames analyzed")
    
    duplicates = sum(1 for f in frames if f['is_duplicate'])
    repetitive = sum(1 for f in frames if f['is_repetitive'])
    avg_motion = np.mean([f['motion'] for f in frames[:-1]])
    
    print(f"   ✓ Found {duplicates} duplicate, {repetitive} repetitive frames")
    print(f"   ✓ Average motion: {avg_motion:.3f}")
    return frames


def create_intelligent_scenes(frames):
    """Create scenes based on semantic understanding and detect boring parts"""
    print(f"\n📋 Creating intelligent scenes with AI analysis...")
    
    WINDOW_SIZE = 8  # 16 seconds - tighter windows for better transitions
    MIN_SCENE_DURATION = 8  # 8 seconds minimum
    
    transitions = [0]
    
    # Sliding window to detect significant semantic changes
    for i in range(WINDOW_SIZE, len(frames), WINDOW_SIZE):
        if i + WINDOW_SIZE >= len(frames):
            break
        
        prev_window = frames[i-WINDOW_SIZE:i]
        curr_window = frames[i:i+WINDOW_SIZE]
        
        # Semantic content change
        prev_interest = np.mean([f['semantic_interest'] for f in prev_window])
        curr_interest = np.mean([f['semantic_interest'] for f in curr_window])
        
        prev_boring = np.mean([f['semantic_boring'] for f in prev_window])
        curr_boring = np.mean([f['semantic_boring'] for f in curr_window])
        
        # Motion change (important!)
        prev_motion = np.mean([f.get('motion', 0) for f in prev_window])
        curr_motion = np.mean([f.get('motion', 0) for f in curr_window])
        
        # CLIP workflow stages detection
        prev_shine = np.mean([f.get('clip_shine', 0) for f in prev_window])
        curr_shine = np.mean([f.get('clip_shine', 0) for f in curr_window])
        
        prev_angle = np.mean([f.get('clip_angle', 0) for f in prev_window])
        curr_angle = np.mean([f.get('clip_angle', 0) for f in curr_window])
        
        # Calculate changes
        interest_change = abs(curr_interest - prev_interest)
        boring_change = abs(curr_boring - prev_boring)
        motion_change = abs(curr_motion - prev_motion)
        shine_change = abs(curr_shine - prev_shine)
        angle_change = abs(curr_angle - prev_angle)
        
        # Detect transitions: workflow stage changes, view changes, or progress visible
        if (interest_change > 0.05 or boring_change > 0.05 or motion_change > 0.03 or 
            shine_change > 0.08 or angle_change > 0.10):
            transitions.append(i)
            print(f"   Transition at {frames[i]['timestamp']:.1f}s: "
                  f"interest±{interest_change:.0%}, shine±{shine_change:.0%}, "
                  f"angle±{angle_change:.0%}, motion±{motion_change:.3f}")
    
    transitions.append(len(frames) - 1)
    print(f"   Found {len(transitions)-1} scene transitions")
    
    # GLOBAL SHOWCASE SEARCH: Find best 3 diverse moments across ENTIRE video
    # This ensures we don't miss interesting parts stuck in middle of long repetitive scenes
    print(f"🔍 Scanning entire video for 3 most diverse showcases...")
    
    showcase_duration = 25  # seconds
    subsection_frames_count = int((showcase_duration / 2.0) * (len(frames) / (frames[-1]['timestamp'] + 0.001)))
    if subsection_frames_count < 5:
        subsection_frames_count = 10
    
    global_showcases = []  # Will store (score, start_idx, features, variance, timestamp)
    
    # Scan every 3 frames (6 seconds) across ENTIRE video
    for start_idx in range(0, len(frames) - subsection_frames_count, 3):
        end_idx = min(start_idx + subsection_frames_count, len(frames))
        subsection = frames[start_idx:end_idx]
        
        # Score this subsection
        sub_interest = np.mean([f.get('semantic_interest', 0) for f in subsection])
        sub_dup = sum(1 for f in subsection if f.get('is_duplicate', False)) / max(1, len(subsection))
        sub_motion = np.mean([f.get('motion', 0) for f in subsection])
        sub_shine = np.mean([f.get('clip_shine', 0) for f in subsection])
        sub_applying = np.mean([f.get('clip_applying', 0) for f in subsection])
        sub_hands = np.mean([f.get('clip_hands', 0) for f in subsection])
        sub_angle = np.mean([f.get('clip_angle', 0) for f in subsection])
        sub_detail = np.mean([f.get('clip_detail', 0) for f in subsection])
        
        # Visual variance
        sub_features = np.array([f['features'] for f in subsection])
        sub_visual_variance = np.mean(np.std(sub_features, axis=0))
        
        # CLIP features
        sub_clip_features = np.mean([f['clip_image_features'] for f in subsection], axis=0)
        
        # IMPROVED SCORING: Prioritize actual work (applying+hands+detail) over just visual changes
        # Polishing IS somewhat repetitive, so don't penalize duplication too heavily
        # Only reward variance if there's also active work happening
        work_score = sub_applying * 100 + sub_hands * 80 + sub_detail * 60
        quality_score = sub_shine * 40 + sub_interest * 30
        
        # Only add variance bonus if there's actual work (applying > 0.15)
        variance_bonus = sub_visual_variance * 20 if sub_applying > 0.15 else 0
        
        # Light penalty for extreme duplication (>80%), but allow some repetition
        dup_penalty = max(0, (sub_dup - 0.80) * 100) if sub_dup > 0.80 else 0
        
        score = work_score + quality_score + variance_bonus - dup_penalty
        
        global_showcases.append({
            'score': score,
            'start_idx': start_idx,
            'timestamp': subsection[0]['timestamp'],
            'clip_features': sub_clip_features,
            'visual_variance': sub_visual_variance,
            'applying': sub_applying,
            'shine': sub_shine,
            'angle': sub_angle,
            'hands': sub_hands,
            'detail': sub_detail,
            'dup': sub_dup,
        })
    
    # Sort by score and pick top 3 with diversity enforcement
    global_showcases.sort(key=lambda x: x['score'], reverse=True)
    
    # Debug: Show top 10 candidates
    print(f"   Top 10 scoring moments:")
    for i, cand in enumerate(global_showcases[:10]):
        print(f"      #{i+1}: {cand['timestamp']:.1f}s score={cand['score']:.1f} "
              f"(applying={cand['applying']:.2f}, hands={cand['hands']:.2f}, "
              f"detail={cand['detail']:.2f}, dup={cand['dup']:.0%})")
    
    selected_showcases = []  # Track showcase timestamps and features for diversity
    final_showcase_candidates = []
    
    for candidate in global_showcases:
        if len(final_showcase_candidates) >= 3:
            break
        
        # Check diversity against already selected showcases
        diversity_penalty = 0
        for prev in final_showcase_candidates:
            time_diff = abs(candidate['timestamp'] - prev['timestamp'])
            
            # Time proximity penalty
            if time_diff < 150:  # At least 2.5 minutes apart
                diversity_penalty += 100
            elif time_diff < 240:  # Prefer 4+ minutes apart
                diversity_penalty += 40
            
            # Visual similarity penalty
            visual_similarity = np.dot(candidate['clip_features'], prev['clip_features']) / (
                np.linalg.norm(candidate['clip_features']) * np.linalg.norm(prev['clip_features'])
            )
            
            if visual_similarity > 0.95:
                diversity_penalty += 150
            elif visual_similarity > 0.92:
                diversity_penalty += 80
            elif visual_similarity > 0.88:
                diversity_penalty += 40
        
        adjusted_score = candidate['score'] - diversity_penalty
        
        # Accept if score still reasonable after penalties
        if len(final_showcase_candidates) == 0 or adjusted_score > -100:
            final_showcase_candidates.append(candidate)
            print(f"   Selected showcase at {candidate['timestamp']:.1f}s: "
                  f"score={adjusted_score:.1f}, variance={candidate['visual_variance']:.3f}")
    
    print(f"   ✓ Found {len(final_showcase_candidates)} global showcases")
    
    # Convert final showcases to selected_showcases for scene creation
    selected_showcases = final_showcase_candidates
    
    # Create scenes with diversity-aware showcase selection
    scenes = []
    
    for i in range(len(transitions) - 1):
        start_idx = transitions[i]
        end_idx = transitions[i+1]
        scene_frames = frames[start_idx:end_idx]
        
        if len(scene_frames) < 5:
            continue
        
        start_time = scene_frames[0]['timestamp']
        end_time = scene_frames[-1]['timestamp']
        duration = end_time - start_time
        
        if duration < MIN_SCENE_DURATION:
            continue
        
        # Calculate scene quality metrics
        avg_interest = np.mean([f['semantic_interest'] for f in scene_frames])
        avg_boring = np.mean([f['semantic_boring'] for f in scene_frames])
        duplicate_ratio = sum(1 for f in scene_frames if f['is_duplicate']) / len(scene_frames)
        repetitive_ratio = sum(1 for f in scene_frames if f['is_repetitive']) / len(scene_frames)
        
        # Feature variance (how much changes in the scene)
        features_matrix = np.array([f['features'] for f in scene_frames])
        feature_variance = np.mean(np.std(features_matrix, axis=0))
        
        # Combined quality score
        quality_score = (
            avg_interest * 50 +
            (1 - avg_boring) * 30 +
            (1 - duplicate_ratio) * 10 +
            (1 - repetitive_ratio) * 10 +
            min(feature_variance * 20, 20)
        )
        
        # Check if any global showcase falls within this scene
        scene_showcase = None
        for showcase in selected_showcases:
            if start_time <= showcase['timestamp'] < end_time:
                scene_showcase = showcase
                break
        
        # Smart split: if scene is long, repetitive, AND has a showcase within it
        if duration > 60 and (duplicate_ratio > 0.80 or repetitive_ratio > 0.85) and scene_showcase:
            showcase_duration = 25  # seconds
            showcase_start_time = scene_showcase['timestamp']
            showcase_idx_in_scene = None
            
            # Find the frame index in scene_frames that matches showcase timestamp
            for idx, frame in enumerate(scene_frames):
                if abs(frame['timestamp'] - showcase_start_time) < 1.0:  # Within 1 second
                    showcase_idx_in_scene = idx
                    break
            
            if showcase_idx_in_scene is not None:
                subsection_frames = int(len(scene_frames) * (showcase_duration / duration))
                if subsection_frames < 5:
                    subsection_frames = min(10, len(scene_frames))
                
                showcase_end_idx = min(showcase_idx_in_scene + subsection_frames, len(scene_frames))
                showcase_end_time = scene_frames[showcase_end_idx - 1]['timestamp']
                
                # Part 1: Beginning to showcase (if exists) - speedup
                if showcase_start_time > start_time + 5:
                    before_frames = scene_frames[:showcase_idx_in_scene]
                    before_duration = showcase_start_time - start_time
                    scenes.append({
                        'scene_num': len(scenes) + 1,
                        'start_time': start_time,
                        'end_time': showcase_start_time,
                        'duration': before_duration,
                        'frame_count': showcase_idx_in_scene,
                        'semantic_interest': float(avg_interest),
                        'semantic_boring': float(avg_boring),
                        'duplicate_ratio': float(duplicate_ratio),
                        'repetitive_ratio': float(repetitive_ratio),
                        'feature_variance': float(feature_variance),
                        'quality_score': float(quality_score - 10),
                        'is_showcase': False,
                        'scene_hash': scene_hash(before_frames)
                    })
                    print(f"   Scene {len(scenes):02d}: {start_time:.1f}s-{showcase_start_time:.1f}s ({before_duration:.1f}s) "
                          f"[SPEEDUP-PRE] dup={duplicate_ratio:.0%}")
                
                # Part 2: Showcase - the selected global showcase
                showcase_real_duration = showcase_end_time - showcase_start_time
                showcase_frames = scene_frames[showcase_idx_in_scene:showcase_end_idx]
                scenes.append({
                    'scene_num': len(scenes) + 1,
                    'start_time': showcase_start_time,
                    'end_time': showcase_end_time,
                    'duration': showcase_real_duration,
                    'frame_count': subsection_frames,
                    'semantic_interest': float(avg_interest),
                    'semantic_boring': float(avg_boring),
                    'duplicate_ratio': 0.0,  # Override - this is showcase
                    'repetitive_ratio': 0.0,
                    'feature_variance': float(scene_showcase['visual_variance']),
                    'quality_score': float(scene_showcase['score']),
                    'is_showcase': True,
                    'scene_hash': scene_hash(showcase_frames)
                })
                
                print(f"   Scene {len(scenes):02d}: {showcase_start_time:.1f}s-{showcase_end_time:.1f}s ({showcase_real_duration:.1f}s) "
                      f"[SHOWCASE] score={scene_showcase['score']:.1f}, variance={scene_showcase['visual_variance']:.3f}")
                
                # Part 3: After showcase to end - speedup
                if end_time > showcase_end_time + 5:
                    after_frames = scene_frames[showcase_end_idx:]
                    after_duration = end_time - showcase_end_time
                    remaining_frames = len(scene_frames) - showcase_end_idx
                    scenes.append({
                        'scene_num': len(scenes) + 1,
                        'start_time': showcase_end_time,
                        'end_time': end_time,
                        'duration': after_duration,
                        'frame_count': remaining_frames,
                        'semantic_interest': float(avg_interest),
                        'semantic_boring': float(avg_boring),
                        'duplicate_ratio': float(duplicate_ratio),
                        'repetitive_ratio': float(repetitive_ratio),
                        'feature_variance': float(feature_variance),
                        'quality_score': float(quality_score - 10),
                        'is_showcase': False,
                        'scene_hash': scene_hash(after_frames)
                    })
                    print(f"   Scene {len(scenes):02d}: {showcase_end_time:.1f}s-{end_time:.1f}s ({after_duration:.1f}s) "
                          f"[SPEEDUP-POST] dup={duplicate_ratio:.0%}")
            
            continue
        
        # Regular scene (not split)
        scenes.append({
            'scene_num': len(scenes) + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frame_count': len(scene_frames),
            'semantic_interest': float(avg_interest),
            'semantic_boring': float(avg_boring),
            'duplicate_ratio': float(duplicate_ratio),
            'repetitive_ratio': float(repetitive_ratio),
            'feature_variance': float(feature_variance),
            'quality_score': float(quality_score),
            'is_showcase': False,
            'scene_hash': scene_hash(scene_frames)
        })
        
        print(f"   Scene {len(scenes):02d}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s) "
              f"quality={quality_score:.1f} (interest={avg_interest:.2f}, dup={duplicate_ratio:.0%}, rep={repetitive_ratio:.0%})")
    
    print(f"   ✓ Created {len(scenes)} scenes")
    
    # Merge small transitional scenes before/after showcases
    merged_scenes = []
    i = 0
    while i < len(scenes):
        scene = scenes[i]
        
        # If this is a showcase, look for small scenes before/after to merge
        if scene.get('is_showcase', False):
            # Check if previous scene is small (<30s) and not already merged
            if i > 0 and len(merged_scenes) > 0:
                prev = merged_scenes[-1]
                if prev['duration'] < 30 and not prev.get('is_showcase', False):
                    # Merge previous small scene with showcase
                    merged_scenes[-1] = {
                        **scene,
                        'start_time': prev['start_time'],
                        'duration': scene['end_time'] - prev['start_time'],
                        'frame_count': prev['frame_count'] + scene['frame_count'],
                        'scene_num': prev['scene_num']
                    }
                    print(f"   ↳ Merged scene {prev['scene_num']} (small) with showcase → {prev['start_time']:.1f}s-{scene['end_time']:.1f}s")
                    i += 1
                    continue
            
            # Check if next scene is small (<30s) - merge it too
            if i + 1 < len(scenes):
                next_scene = scenes[i + 1]
                if next_scene['duration'] < 30 and not next_scene.get('is_showcase', False):
                    # Merge showcase with next small scene
                    merged_scene = {
                        **scene,
                        'end_time': next_scene['end_time'],
                        'duration': next_scene['end_time'] - scene['start_time'],
                        'frame_count': scene['frame_count'] + next_scene['frame_count'],
                    }
                    merged_scenes.append(merged_scene)
                    print(f"   ↳ Merged showcase with scene {next_scene['scene_num']} (small) → {scene['start_time']:.1f}s-{next_scene['end_time']:.1f}s")
                    i += 2
                    continue
        
        merged_scenes.append(scene)
        i += 1
    
    # Renumber scenes
    for idx, scene in enumerate(merged_scenes):
        scene['scene_num'] = idx + 1
    
    print(f"   ✓ Final: {len(merged_scenes)} scenes after merging")
    return merged_scenes


def classify_scenes(scenes):
    """Classify scenes by quality with emphasis on duplication/repetition"""
    print(f"\n🎨 Classifying scenes by AI-detected quality...")
    
    if not scenes:
        return scenes
    
    scores = [s['quality_score'] for s in scenes]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"   Quality scores: mean={mean_score:.1f}, std={std_score:.1f}")
    
    for scene in scenes:
        score = scene['quality_score']
        dup_ratio = scene['duplicate_ratio']
        rep_ratio = scene['repetitive_ratio']
        interest = scene['semantic_interest']
        boring = scene['semantic_boring']
        duration = scene['duration']
        is_showcase = scene.get('is_showcase', False)
        
        # Showcase sections are always interesting (showing the activity)
        if is_showcase:
            scene['classification'] = 'interesting'
            scene['speed'] = 1.0
        # Strong boring signals
        elif (boring > 0.50 and interest < 0.22) or (dup_ratio > 0.80 or rep_ratio > 0.80):
            scene['classification'] = 'boring'
            scene['speed'] = 6.0
        elif dup_ratio > 0.60 or rep_ratio > 0.65 or boring > 0.35:
            # Repetitive or semantically boring
            scene['classification'] = 'low'
            scene['speed'] = 4.0
        elif interest > 0.34 and dup_ratio < 0.40 and boring < 0.30:
            # High interest and low duplication - interesting
            scene['classification'] = 'interesting'
            scene['speed'] = 1.0
        elif interest > 0.26:
            # Decent interest - moderate
            scene['classification'] = 'moderate'
            scene['speed'] = 2.0
        else:
            # Default - low
            scene['classification'] = 'low'
            scene['speed'] = 4.0
        
        print(f"   Scene {scene['scene_num']:02d}: {scene['classification']:12s} ({scene['speed']:.1f}x) "
              f"dup={dup_ratio:.0%}, rep={rep_ratio:.0%}, interest={interest:.2f}, boring={boring:.2f}")
    
    return scenes


def save_results(scenes, frames, output_file, video_name):
    """Save analysis results"""
    counts = {'interesting': 0, 'moderate': 0, 'low': 0, 'boring': 0, 'skip': 0}
    total_original = sum(s['duration'] for s in scenes)
    total_output = sum(s['duration'] / s['speed'] for s in scenes if s['speed'] > 0)
    
    for scene in scenes:
        counts[scene['classification']] += 1
    
    result = {
        'video': video_name,
        'method': 'advanced_ai_v3',
        'models': ['CLIP_ViT-B/32', 'ResNet50', 'Cosine_Similarity', 'Duplicate_Detection'],
        'total_frames': len(frames),
        'total_scenes': len(scenes),
        'scenes': scenes,
        'summary': {
            'interesting': counts['interesting'],
            'moderate': counts['moderate'],
            'low': counts['low'],
            'boring': counts['boring'],
            'skip': counts['skip'],
            'original_duration': total_original,
            'output_duration': total_output,
            'compression_ratio': (1 - total_output / total_original) * 100 if total_original > 0 else 0
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    return result


def save_keyframe_captions(frames, output_dir, video_stem, suffix=None):
    """Save per-keyframe captions for kept keyframes only."""
    suffix_part = f"_{suffix}" if suffix else ""
    captions_jsonl = output_dir / f"keyframe_captions_{video_stem}{suffix_part}.jsonl"
    captions_csv = output_dir / f"keyframe_captions_{video_stem}{suffix_part}.csv"

    with open(captions_jsonl, "w") as jf, open(captions_csv, "w", newline="") as cf:
        fieldnames = [
            "index",
            "frame_index",
            "timestamp",
            "caption",
            "caption_model",
            "caption_prompt",
            "frame_path",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, frame in enumerate(frames):
            entry = {
                "index": idx,
                "frame_index": frame.get("index"),
                "timestamp": round(float(frame.get("timestamp", 0.0)), 3),
                "caption": frame.get("caption"),
                "caption_model": frame.get("caption_model"),
                "caption_prompt": frame.get("caption_prompt"),
                "frame_path": frame.get("path"),
            }
            jf.write(json.dumps(entry) + "\n")
            writer.writerow(entry)

    print(f"\n🧭 Keyframe captions saved: {captions_jsonl}")
    print(f"🧭 Keyframe captions saved: {captions_csv}")
    return captions_jsonl, captions_csv


def filter_frames_by_scenes(frames, scenes):
    """Return frames that fall within kept scene time ranges."""
    if not frames or not scenes:
        return []
    ranges = []
    for scene in scenes:
        start = scene.get('start_time')
        end = scene.get('end_time')
        if start is None or end is None:
            continue
        if end < start:
            continue
        ranges.append((float(start), float(end)))
    if not ranges:
        return []
    ranges.sort(key=lambda r: r[0])

    kept = []
    range_idx = 0
    current_start, current_end = ranges[range_idx]
    for frame in frames:
        ts = float(frame.get('timestamp', 0.0))
        while range_idx < len(ranges) - 1 and ts > current_end:
            range_idx += 1
            current_start, current_end = ranges[range_idx]
        if current_start <= ts <= current_end:
            kept.append(frame)
    return kept


def adjust_speeds_to_target(scenes, target_output_ratio, max_speed):
    """Scale speeds of non-interesting scenes to hit target output ratio."""
    if not scenes:
        return scenes

    total_original = sum(s['duration'] for s in scenes)
    if total_original <= 0:
        return scenes

    def current_output_ratio():
        return sum(s['duration'] / s['speed'] for s in scenes if s['speed'] > 0) / total_original

    output_ratio = current_output_ratio()
    if output_ratio <= target_output_ratio:
        return scenes

    # Scale only non-interesting scenes
    adjustable = [s for s in scenes if s.get('classification') != 'interesting']
    if not adjustable:
        return scenes

    scale = min(3.0, output_ratio / target_output_ratio)

    for s in adjustable:
        s['speed'] = min(max_speed, s['speed'] * scale)

    # Recompute if still above target, apply one more gentle pass
    output_ratio = current_output_ratio()
    if output_ratio > target_output_ratio:
        scale = min(2.0, output_ratio / target_output_ratio)
        for s in adjustable:
            s['speed'] = min(max_speed, s['speed'] * scale)

    return scenes


def process_video(video_path, output_dir, clip_model, resnet_model, sample_interval, object_detection_cfg=None, caption_model=None, captioning_cfg=None):
    """Run full analysis for a single video and write results."""
    print("\n" + "=" * 70)
    print(f"🎬 Processing: {video_path.name}")
    print("=" * 70)
    
    # Extract frames
    frames = extract_frames_parallel(video_path, sample_interval)
    
    # AI analysis
    frames = analyze_with_clip(frames, clip_model)
    frames, features = extract_features(frames, resnet_model)
    frames = detect_duplicates_and_repetition(frames, features)

    # Object detection (optional)
    object_detection_cfg = object_detection_cfg or {}
    od_enabled = object_detection_cfg.get("enabled", False)
    od_threshold = float(object_detection_cfg.get("score_threshold", 0.5))
    od_max_objects = int(object_detection_cfg.get("max_objects", 5))
    detection_model = None
    if od_enabled:
        detection_model = load_object_detection_model()
        frames = analyze_with_object_detection(frames, detection_model, score_threshold=od_threshold, max_objects=od_max_objects)
    
    # Create and classify scenes
    scenes = create_intelligent_scenes(frames)
    scenes = classify_scenes(scenes)
    scenes = adjust_speeds_to_target(scenes, TARGET_OUTPUT_RATIO, MAX_SPEED_MULTIPLIER)

    # Caption kept keyframes only
    kept_frames = filter_frames_by_scenes(frames, scenes)
    captioning_cfg = captioning_cfg or {}
    cap_enabled = captioning_cfg.get("enabled", True)
    cap_max_length = int(captioning_cfg.get("max_length", 40))
    cap_num_beams = int(captioning_cfg.get("num_beams", 5))
    cap_min_length = captioning_cfg.get("min_length", None)
    cap_min_length = int(cap_min_length) if cap_min_length is not None else None
    cap_prompt = captioning_cfg.get(
        "prompt",
        "Describe the scene in detail, focusing on scale model car painting or polishing workflow, tools, materials, and actions."
    )
    cap_use_prompt_for_blip = bool(captioning_cfg.get("use_prompt_for_blip", False))
    cap_device = captioning_cfg.get("device", "cuda")
    cap_fallback_models = captioning_cfg.get("fallback_models", [])
    if isinstance(cap_fallback_models, str):
        cap_fallback_models = [cap_fallback_models]
    cap_quant = captioning_cfg.get("quantization", "")
    if cap_enabled:
        if GPU_DEVICE == "cuda":
            print("\n🧹 Releasing GPU models before captioning...")
            release_model_dict(clip_model)
            release_model_dict(resnet_model)
            if detection_model is not None:
                release_model_dict(detection_model)
            clear_cuda_cache()

        if caption_model is None:
            model_candidates = [captioning_cfg.get("model", "Salesforce/blip-image-captioning-large")] + cap_fallback_models
            for model_name in model_candidates:
                clear_cuda_cache()
                caption_model = load_caption_model(
                    model_name,
                    device=cap_device,
                    torch_dtype=torch.float16 if cap_device == "cuda" else None,
                    quantization=cap_quant,
                )
                if caption_model is not None:
                    break

        if caption_model is None:
            print("   ⚠️  Captioning skipped (CUDA only; model failed to load).")
        else:
            cap_max_image_size = captioning_cfg.get("max_image_size", 768)
            kept_frames = caption_keyframes(
                kept_frames,
                caption_model,
                max_length=cap_max_length,
                num_beams=cap_num_beams,
                prompt=cap_prompt,
                use_prompt_for_blip=cap_use_prompt_for_blip,
                min_length=cap_min_length,
                max_image_size=cap_max_image_size,
            )
            save_keyframe_captions(kept_frames, output_dir, video_path.stem, suffix="kept")
    
    # Save and report
    output_file = output_dir / f"scene_analysis_{video_path.stem}.json"
    result = save_results(scenes, frames, output_file, video_path.name)
    print_summary(result)
    
    return output_file


def print_summary(result):
    """Print analysis summary"""
    s = result['summary']
    print("\n" + "=" * 70)
    print("📊 Advanced AI Analysis v3 Report")
    print("=" * 70)
    print(f"  Models: {', '.join(result['models'])}")
    print(f"  Interesting (1.0x):  {s['interesting']} scenes")
    print(f"  Moderate (1.5x):     {s['moderate']} scenes")
    print(f"  Low (2.5x):          {s['low']} scenes")
    print(f"  Boring (4.0x):       {s['boring']} scenes")
    print(f"  Skip:                {s['skip']} scenes")
    print()
    print(f"  Original:    {s['original_duration']/60:.1f} min")
    print(f"  Compressed:  {s['output_duration']/60:.1f} min")
    print(f"  Saved:       {(s['original_duration']-s['output_duration'])/60:.1f} min ({s['compression_ratio']:.0f}%)")
    print("=" * 70)


def main():
    """Main analysis workflow"""
    parser = argparse.ArgumentParser(description="Batch analyze videos in a folder")
    parser.add_argument("--config", default="project_config.json", help="Project config JSON file")
    parser.add_argument("--input-dir", default=None, help="Folder with input videos")
    parser.add_argument("--video", default=None, help="Single video file to analyze")
    parser.add_argument("--output-dir", default=None, help="Folder for analysis JSON outputs")
    parser.add_argument("--sample-interval", type=int, default=None, help="Frame sample interval (seconds)")
    args = parser.parse_args()

    config = load_project_config(args.config)
    paths_cfg = config.get("paths", {})
    analysis_cfg = config.get("analysis", {})

    sample_interval = args.sample_interval if args.sample_interval is not None else analysis_cfg.get("sample_interval", SAMPLE_INTERVAL)

    global TARGET_OUTPUT_RATIO
    global MAX_SPEED_MULTIPLIER
    if "target_output_ratio" in analysis_cfg:
        TARGET_OUTPUT_RATIO = float(analysis_cfg["target_output_ratio"])
    if "max_speed_multiplier" in analysis_cfg:
        MAX_SPEED_MULTIPLIER = float(analysis_cfg["max_speed_multiplier"])
    
    output_dir = Path(args.output_dir or paths_cfg.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_arg = args.video or paths_cfg.get("video")
    if video_arg:
        video_paths = [Path(video_arg)]
    else:
        input_dir = Path(args.input_dir or paths_cfg.get("input_dir") or ".")
        video_paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".mkv"}
        ]
        video_paths.sort(key=lambda p: p.name.lower())
    
    if not video_paths:
        print("❌ No videos found to analyze.")
        return
    
    output_files = []
    object_detection_cfg = analysis_cfg.get("object_detection", {}) if isinstance(analysis_cfg, dict) else {}
    captioning_cfg = analysis_cfg.get("captioning", {}) if isinstance(analysis_cfg, dict) else {}

    for video_path in video_paths:
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            continue
        clip_model = load_clip_model()
        resnet_model = load_resnet_model()
        output_file = process_video(
            video_path,
            output_dir,
            clip_model,
            resnet_model,
            sample_interval,
            object_detection_cfg=object_detection_cfg,
            caption_model=None,
            captioning_cfg=captioning_cfg,
        )
        output_files.append(output_file)
    
    if output_files:
        print("\n💡 Next: python extract_scenes.py --analysis-dir", output_dir)


if __name__ == "__main__":
    main()
