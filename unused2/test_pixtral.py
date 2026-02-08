#!/usr/bin/env python3
"""Test Pixtral 12B vision-language model"""
import base64
from pathlib import Path
from llama_cpp import Llama

# Model path
model_path = str(Path.home() / ".cache/huggingface/pixtral-12b/mistral-community_pixtral-12b-Q6_K_L.gguf")

print("="*70)
print("Testing Pixtral 12B Q6_K_L (10.4GB)")
print("="*70)

# Test with the frame
print("\nLoading model with vision support...")
try:
    # Pixtral has built-in vision, try with chat_format
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # Pixtral supports longer context
        n_gpu_layers=-1,  # Offload all to GPU
        chat_format="pixtral",  # Try pixtral chat format
        verbose=True
    )
    print("✓ Model loaded with pixtral chat format!")
    
    # Load test image
    print("\nTesting image captioning...")
    with open('/tmp/llava_test_frame.jpg', 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    import time
    start = time.time()
    
    # Try with image
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
    print(f"Pixtral 12B Q6_K_L Result")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nCaption:\n{caption}")
    print(f"\n{'='*70}")
    print("✓ SUCCESS! Pixtral 12B is working!")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"✗ Error loading with pixtral format: {e}")
    print("\nTrying without chat format...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print("✓ Model loaded (text-only mode)")
        
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say 'Hello, I am Pixtral!'"}],
            max_tokens=50
        )
        print(f"\nText test: {response['choices'][0]['message']['content']}")
        print("\n⚠️  Vision features may not be available in current llama-cpp-python version")
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
