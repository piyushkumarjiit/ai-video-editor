#!/usr/bin/env python3
"""Test MiniCPM-V 2.6 Q8_0 GGUF model"""
import base64
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

# Model paths
model_path = str(Path.home() / ".cache/huggingface/minicpm-v-2.6/MiniCPM-V-2_6-Q8_0.gguf")
mmproj_path = str(Path.home() / ".cache/huggingface/minicpm-v-2.6/mmproj-MiniCPM-V-2_6-f16.gguf")

print("="*70)
print("Testing MiniCPM-V 2.6 Q8_0 (~8-9GB)")
print("="*70)

print("\nLoading model with MiniCPMv26ChatHandler...")
try:
    # MiniCPM-V 2.6 uses a dedicated chat handler with mmproj
    chat_handler = MiniCPMv26ChatHandler(
        clip_model_path=mmproj_path,
        verbose=False
    )
    
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=4096,
        n_gpu_layers=-1,  # Offload all to GPU
        verbose=False
    )
    print("✓ Model loaded with MiniCPMv26ChatHandler!")
    
    # Load test image
    print("\nTesting image captioning...")
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
    print(f"MiniCPM-V 2.6 Q8_0 Result")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nCaption:\n{caption}")
    print(f"\n{'='*70}")
    print("✓ SUCCESS! MiniCPM-V 2.6 is working!")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"✗ Error with MiniCPMv26ChatHandler: {e}")
    print("\nTrying without chat handler...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print("✓ Model loaded (basic mode)")
        
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say 'Hello from MiniCPM-V!'"}],
            max_tokens=50
        )
        print(f"\nText test: {response['choices'][0]['message']['content']}")
        print("\n⚠️  Note: MiniCPM-V requires MiniCPMv26ChatHandler for vision features")
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
