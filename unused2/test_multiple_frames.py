#!/usr/bin/env python3
"""Test multiple vision models to compare quality"""
import base64
import subprocess
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
from pathlib import Path
import os
import glob

# Available vision models to test
models_to_test = [
    {
        "name": "Qwen2-VL-7B-Q4_K_M (7B 4-bit)",
        "type": "qwen2-vl",
        "path": "models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
    },
    {
        "name": "Qwen2-VL-7B-Q5_K_M (7B 5-bit)",
        "type": "qwen2-vl",
        "path": "models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q5_K_M.gguf"
    },
    {
        "name": "LLaVA-1.6-vicuna-13B-Q4_K_M (13B 4-bit)",
        "type": "llava",
        "path": "models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/llava-v1.6-vicuna-13b.Q4_K_M.gguf",
        "mmproj": "models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/mmproj-model-f16.gguf"
    }
]

cache_dir = Path.home() / ".cache/huggingface/hub"

# Find which models are actually available
available_models = []
for model_info in models_to_test:
    # Handle glob patterns for LLaVA models
    model_pattern = cache_dir / model_info["path"]
    model_matches = glob.glob(str(model_pattern))
    
    if model_matches:
        model_path = model_matches[0]
        model_dict = {**model_info, "full_path": model_path}
        
        # For LLaVA models, also find the mmproj file
        if model_info["type"] == "llava":
            mmproj_pattern = cache_dir / model_info["mmproj"]
            mmproj_matches = glob.glob(str(mmproj_pattern))
            if mmproj_matches:
                model_dict["mmproj_path"] = mmproj_matches[0]
                available_models.append(model_dict)
                print(f"✓ Found: {model_info['name']}")
            else:
                print(f"✗ Not found: {model_info['name']} (missing mmproj)")
        else:
            available_models.append(model_dict)
            print(f"✓ Found: {model_info['name']}")
    else:
        print(f"✗ Not found: {model_info['name']}")

if not available_models:
    print("\n❌ No models found!")
    exit(1)

print(f"\n📦 Testing {len(available_models)} model(s) sequentially (to avoid GPU memory conflicts)...")
# Test just one representative frame at 80 seconds
test_time = 80
test_desc = "middle of video - painting activity"

# Best prompt from previous tests
test_prompt = "Describe what you see in this image: the car part, tools, paint colors, and what the hands are doing."

print("\n" + "="*70)
print(f"Comparing model quality on frame at {test_time}s ({test_desc})")
print("="*70)

# Extract the test frame once
frame_path = f"/tmp/compare_models_frame.jpg"
print(f"\nExtracting frame at {test_time}s with rotation fix...")
subprocess.run(['ffmpeg', '-y', '-ss', str(test_time), '-i', '/home/mazsola/video/IMG_3520.MOV',
                '-vframes', '1', '-vf', 'transpose=2', '-q:v', '1', frame_path],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"   ✓ Frame extracted")

with open(frame_path, 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

# Test each model
for idx, model_info in enumerate(available_models, 1):
    print(f"\n{'='*70}")
    print(f"Model {idx}/{len(available_models)}: {model_info['name']}")
    print(f"{'='*70}")
    
    # Load model
    print(f"Loading model...")
    try:
        if model_info["type"] == "llava":
            chat_handler = Llava16ChatHandler(clip_model_path=model_info["mmproj_path"], verbose=False)
            model = Llama(
                model_path=model_info["full_path"],
                chat_handler=chat_handler,
                n_ctx=3072,
                n_gpu_layers=-1,
                verbose=False
            )
        else:
            model = Llama(
                model_path=model_info["full_path"],
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False
            )
        print("   ✓ Loaded")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        continue
    
    try:
        import time
        start = time.time()
        
        output = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes images of hobby workshops and scale model building. Always describe what you see."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": test_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.2
        )
        
        elapsed = time.time() - start
        caption = output['choices'][0]['message']['content'].strip()
        
        print(f"\nTime: {elapsed:.1f}s")
        print(f"\nCaption:\n{caption}\n")
        
        # Check for refusals
        if "I'm sorry" in caption or "I cannot" in caption or "I can't" in caption:
            print("⚠️  Contains refusal language")
        else:
            print("✓ Actual description provided")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Clean up model from GPU memory
    del model
    import gc
    gc.collect()

print("\n" + "="*70)
print("Model comparison complete!")
print("="*70)
print("\n💡 Results summary:")
print("   - Q4_K_M (7B): Fast but misses details")
print("   - Q5_K_M (7B): +20% better quality, balanced speed")  
print("   - 13B: Larger model, potentially better understanding")

print("\n" + "="*70)
print("Testing complete!")
print("="*70)
