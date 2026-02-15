#!/usr/bin/env python3
"""Test LLaVA-1.6-vicuna-13B with optimal context size"""
import glob
import base64
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

# Find the model files
model_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/llava-v1.6-vicuna-13b.Q4_K_M.gguf")
mmproj_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/mmproj-model-f16.gguf")

model_path = glob.glob(model_pattern)[0]
mmproj_path = glob.glob(mmproj_pattern)[0]

print("="*70)
print("Loading LLaVA-1.6-vicuna-13B Q4_K_M")
print("Model: 13B parameters, 7.4GB")
print("="*70)

# Try with 3072 context (should fit: 7.4GB model + ~3GB context = ~10.4GB)
print("\nAttempting to load with n_ctx=3072...")
try:
    chat_handler = Llava16ChatHandler(
        clip_model_path=mmproj_path,
        verbose=False
    )
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=3072,  # Should be enough for images (need ~2900 tokens)
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ Model loaded successfully with n_ctx=3072!")
    
    # Test with the frame
    print("\nTesting image captioning at 80s...")
    with open('/tmp/llava_test_frame.jpg', 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    import time
    start = time.time()
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image: the car part, tools, paint colors, and what the hands are doing."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }
        ],
        max_tokens=200,
        temperature=0.2
    )
    
    elapsed = time.time() - start
    caption = response['choices'][0]['message']['content']
    
    print(f"\n{'='*70}")
    print(f"LLaVA-1.6-vicuna-13B Q4_K_M Result")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nCaption:\n{caption}")
    print(f"\n{'='*70}")
    print("✓ SUCCESS! LLaVA 13B is working!")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"✗ FAILED with n_ctx=3072: {e}")
    print("\nTrying with n_ctx=4096 (model's native context)...")
    
    try:
        chat_handler = Llava16ChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print("✓ Loaded with n_ctx=4096")
    except Exception as e2:
        print(f"✗ Also failed with n_ctx=4096: {e2}")
        print("\n💡 Solution: Model requires too much GPU memory")
        print("   With 12GB GPU, the 13B model + context is too large")
        print("   Recommendation: Continue using Qwen2-VL-7B-Q5_K_M")
