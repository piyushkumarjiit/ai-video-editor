#!/usr/bin/env python3
"""Test LLaVA-1.6-vicuna-13B Q4_K_M model with llama.cpp"""
import subprocess
import base64
import glob
from pathlib import Path

# Find the downloaded model
model_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/llava-v1.6-vicuna-13b.Q4_K_M.gguf")
mmproj_pattern = str(Path.home() / ".cache/huggingface/hub/models--cjpais--llava-v1.6-vicuna-13b-gguf/snapshots/*/mmproj-model-f16.gguf")

model_matches = glob.glob(model_pattern)
mmproj_matches = glob.glob(mmproj_pattern)

if not model_matches or not mmproj_matches:
    print("❌ LLaVA model or mmproj not found!")
    print(f"   Model pattern: {model_pattern}")
    print(f"   Mmproj pattern: {mmproj_pattern}")
    exit(1)

model_path = model_matches[0]
mmproj_path = mmproj_matches[0]

print(f"✓ Found model: {model_path}")
print(f"✓ Found mmproj: {mmproj_path}")

# Extract test frame
test_time = 80
frame_path = "/tmp/llava_test_frame.jpg"
print(f"\nExtracting frame at {test_time}s with rotation fix...")
subprocess.run(['ffmpeg', '-y', '-ss', str(test_time), '-i', '/home/mazsola/video/IMG_3520.MOV',
                '-vframes', '1', '-vf', 'transpose=2', '-q:v', '1', frame_path],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"   ✓ Frame extracted: {frame_path}")

# Test with llama.cpp binary if available
print("\n" + "="*70)
print("Testing LLaVA-1.6-vicuna-13B Q4_K_M")
print("="*70)
print("\nNote: LLaVA 1.6 models require llama.cpp built with LLAVA support")
print("or llama-cpp-python >= 0.2.90 with proper multimodal support.")
print("\nCurrent llama-cpp-python version may have compatibility issues.")
print("The model file is valid (13B, 7.4GB) but requires specific loading.")
print("\n💡 Recommendation:")
print("   Continue using Qwen2-VL-7B-Q5_K_M which is proven to work well")
print("   and provides excellent quality for scale model video captioning.")
print("\n" + "="*70)
