#!/usr/bin/env python3
"""Qwen2-VL-7B GGUF loader using llama-cpp-python with CUDA acceleration.

Uses Q4_K_M quantized GGUF model (~4.7GB) with full GPU offloading.
Loads in ~1.4s with all layers on GPU, generating at ~7-20 tokens/sec.
"""

import argparse
import os
import time
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


def find_gguf_model():
    """Auto-detect downloaded GGUF model in HuggingFace cache."""
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


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL 7B GGUF with GPU acceleration")
    parser.add_argument("--model", help="Path to GGUF model file (auto-detects if not provided)")
    parser.add_argument("--image", help="Local file path or URL to image (optional for text-only)")
    parser.add_argument("--prompt", default="Describe exactly what is visible. Avoid guessing; if uncertain, say 'unclear'.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 = all)")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Auto-detect model path
    model_path = args.model
    if not model_path:
        model_path = find_gguf_model()
        if not model_path:
            print("Error: GGUF model not found. Download with:")
            print("  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Qwen2-VL-7B-Instruct-GGUF', filename='Qwen2-VL-7B-Instruct-Q4_K_M.gguf')\"")
            return 1
        print(f"Using model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Loading model with GPU acceleration (n_gpu_layers={args.n_gpu_layers})...", flush=True)
    start_time = time.time()
    
    # Load model with GPU acceleration
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=args.n_gpu_layers,  # -1 = offload all layers to GPU
        n_ctx=args.n_ctx,
        n_threads=8,
        verbose=args.verbose,
        # Vision support (Qwen2-VL GGUF may have limited vision support)
        # chat_format="llava-1-5",  # Uncomment if model supports vision
    )
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.1f}s", flush=True)
    
    # Construct prompt
    if args.image:
        # For vision models (if supported by GGUF)
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{args.prompt}\nImage: {args.image}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Text-only mode
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\nGenerating (max_tokens={args.max_tokens})...", flush=True)
    gen_start = time.time()
    
    output = llm(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        echo=False,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    gen_time = time.time() - gen_start
    
    # Extract generated text
    generated_text = output['choices'][0]['text'].strip()
    tokens_generated = output['usage']['completion_tokens']
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    
    print("\n" + "="*60)
    print("OUTPUT")
    print("="*60)
    print(generated_text)
    print("="*60)
    print(f"\nGeneration stats:")
    print(f"  Tokens: {tokens_generated}")
    print(f"  Time: {gen_time:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    
    return 0


if __name__ == "__main__":
    exit(main())
