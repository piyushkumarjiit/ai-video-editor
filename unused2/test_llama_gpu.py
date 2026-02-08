#!/usr/bin/env python3
"""Test llama-cpp-python GPU usage and memory allocation."""

import subprocess
import time
from pathlib import Path

def run_nvidia_smi():
    """Get GPU memory usage from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        used, total = result.stdout.strip().split(',')
        return int(used.strip()), int(total.strip())
    return None, None

def find_gguf_model():
    """Auto-detect downloaded GGUF model."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    patterns = [
        "models--bartowski--Qwen2-VL-7B-Instruct-GGUF/**/Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        "models--Qwen--Qwen2-VL-7B-Instruct-GGUF/**/qwen2-vl-7b-instruct-q4_k_m.gguf",
    ]
    
    for pattern in patterns:
        matches = list(cache_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return None

print("=" * 70)
print("Testing llama-cpp-python GPU Usage")
print("=" * 70)

# Check GPU before loading
mem_before_used, mem_before_total = run_nvidia_smi()
print(f"\n📊 GPU Memory BEFORE loading model:")
print(f"   Used: {mem_before_used} MB / {mem_before_total} MB")

# Find model
model_path = find_gguf_model()
if not model_path:
    print("\n❌ GGUF model not found!")
    exit(1)

print(f"\n📦 Model: {Path(model_path).name}")
print(f"   Size on disk: {Path(model_path).stat().st_size / 1024**3:.2f} GB")

# Load with GPU
print("\n🔄 Loading model with n_gpu_layers=-1 (all layers on GPU)...")
from llama_cpp import Llama

start = time.time()
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # -1 = all layers to GPU
    n_ctx=4096,
    verbose=True,  # Enable verbose to see layer offloading
)
load_time = time.time() - start

print(f"\n✅ Model loaded in {load_time:.1f}s")

# Check GPU after loading
time.sleep(1)  # Give GPU time to allocate
mem_after_used, mem_after_total = run_nvidia_smi()
mem_delta = mem_after_used - mem_before_used

print(f"\n📊 GPU Memory AFTER loading model:")
print(f"   Used: {mem_after_used} MB / {mem_after_total} MB")
print(f"   Increase: {mem_delta} MB")

# Check if model is actually on GPU
if mem_delta < 1000:
    print("\n⚠️  WARNING: GPU memory increase is very small!")
    print("   Model might be on CPU or only partially on GPU")
    print("\n   Possible reasons:")
    print("   1. llama-cpp-python was compiled without CUDA support")
    print("   2. Model layers aren't being offloaded to GPU")
    print("   3. Model is using CPU memory instead")
else:
    print(f"\n✅ Model appears to be on GPU (using ~{mem_delta} MB)")

# Test inference to see if GPU is used
print("\n🧪 Testing inference speed (GPU should be faster)...")
start = time.time()
output = llm('Q: What is 2+2? A:', max_tokens=10, echo=False)
gen_time = time.time() - start
tokens = output['usage']['completion_tokens']
tokens_per_sec = tokens / gen_time if gen_time > 0 else 0

print(f"   Generated {tokens} tokens in {gen_time:.2f}s")
print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
print(f"   Output: {output['choices'][0]['text'].strip()}")

if tokens_per_sec < 5:
    print("\n⚠️  SLOW! This suggests CPU inference")
    print("   GPU inference should be 10-50+ tokens/sec")
elif tokens_per_sec < 15:
    print("\n⚡ Moderate speed - possible hybrid CPU/GPU")
else:
    print("\n🚀 Fast! GPU is being used")

# Check GPU memory during inference
mem_infer_used, _ = run_nvidia_smi()
print(f"\n📊 GPU Memory DURING inference: {mem_infer_used} MB")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"Model file size: {Path(model_path).stat().st_size / 1024**3:.2f} GB")
print(f"GPU memory used: {mem_delta} MB ({mem_delta/1024:.2f} GB)")
print(f"Inference speed: {tokens_per_sec:.1f} tokens/sec")
print()
if mem_delta > 3000:
    print("✅ Model is FULLY on GPU")
elif mem_delta > 1000:
    print("⚡ Model is PARTIALLY on GPU (hybrid)")
else:
    print("❌ Model is on CPU")
print("=" * 70)
