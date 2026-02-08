#!/usr/bin/env python3
"""Debug LLaVA-1.6-vicuna-13B loading issue"""
import glob
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

# Find the model files
model_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/llava-v1.6-vicuna-13b.Q4_K_M.gguf")
mmproj_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/mmproj-model-f16.gguf")

model_path = glob.glob(model_pattern)[0]
mmproj_path = glob.glob(mmproj_pattern)[0]

print("="*70)
print("Test 1: Load model without chat handler")
print("="*70)
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ SUCCESS: Model loads without chat handler")
    del llm
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*70)
print("Test 2: Create Llava16ChatHandler first")
print("="*70)
try:
    chat_handler = Llava16ChatHandler(
        clip_model_path=mmproj_path,
        verbose=False
    )
    print("✓ SUCCESS: Chat handler created")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test 3: Load model with chat handler")
print("="*70)
try:
    chat_handler = Llava16ChatHandler(
        clip_model_path=mmproj_path,
        verbose=False
    )
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ SUCCESS: Model loads with chat handler")
    
    # Try a simple test
    print("\nTesting image captioning...")
    import base64
    with open('/tmp/llava_test_frame.jpg', 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }
        ],
        max_tokens=100
    )
    print(f"Caption: {response['choices'][0]['message']['content']}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test 4: Try with n_ctx=2048 (larger context)")
print("="*70)
try:
    chat_handler = Llava16ChatHandler(
        clip_model_path=mmproj_path,
        verbose=False
    )
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ SUCCESS with n_ctx=2048")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*70)
print("Test 5: Load model first, then try to use chat format")
print("="*70)
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="llava-1-5"  # Try specifying chat format
    )
    print("✓ SUCCESS with chat_format parameter")
except Exception as e:
    print(f"✗ FAILED with chat_format: {e}")

print("\n" + "="*70)
print("Diagnosis complete")
print("="*70)
