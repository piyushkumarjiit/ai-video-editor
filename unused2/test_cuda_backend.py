#!/usr/bin/env python3
"""Test if llama-cpp-python has CUDA backend support."""

import sys

print("=" * 60)
print("Testing llama-cpp-python CUDA Backend Support")
print("=" * 60)

# Test 1: Check llama-cpp-python version
try:
    import llama_cpp
    print(f"\n✓ llama-cpp-python version: {llama_cpp.__version__}")
except Exception as e:
    print(f"\n✗ Failed to import llama_cpp: {e}")
    sys.exit(1)

# Test 2: Try to access backend functions
print("\n" + "=" * 60)
print("Checking for CUDA backend in llama_cpp library...")
print("=" * 60)

try:
    from llama_cpp import llama_cpp
    
    # Check for CUDA-related attributes
    cuda_attrs = [
        'ggml_backend_cuda_init',
        'ggml_cuda_host_malloc',
        'llama_supports_gpu_offload',
        'ggml_backend_is_cuda',
    ]
    
    found_cuda = False
    for attr in cuda_attrs:
        has_attr = hasattr(llama_cpp, attr)
        status = "✓ FOUND" if has_attr else "✗ NOT FOUND"
        print(f"{status}: {attr}")
        if has_attr:
            found_cuda = True
    
    if not found_cuda:
        print("\n⚠️  NO CUDA functions found in llama_cpp library!")
        print("This is a CPU-ONLY build.")
    else:
        print("\n✓ Some CUDA functions found - may have GPU support")
        
except Exception as e:
    print(f"✗ Error checking backend: {e}")

# Test 3: Check available backends
print("\n" + "=" * 60)
print("Attempting to query available backends...")
print("=" * 60)

try:
    from llama_cpp import Llama
    
    # Try to create a model instance and check what happens with GPU layers
    print("\nAttempting to load model with verbose output...")
    print("This will show which backends are actually available.\n")
    
    # Use the existing GGUF model
    import os
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--bartowski--Qwen2-VL-*-GGUF"))
    
    if not model_dirs:
        model_dirs = list(cache_dir.glob("models--*--*GGUF*"))
    
    if model_dirs:
        model_dir = model_dirs[0]
        gguf_files = list(model_dir.rglob("*.gguf"))
        
        if gguf_files:
            model_path = str(gguf_files[0])
            print(f"Using model: {model_path}")
            
            # Try loading with n_gpu_layers=1 to see if GPU is available
            print("\nTest A: Loading with n_gpu_layers=1...")
            try:
                llm = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_gpu_layers=1,  # Try to offload just 1 layer
                    verbose=True
                )
                print("\n✓ Model loaded with n_gpu_layers=1")
                del llm
            except Exception as e:
                print(f"\n✗ Failed with n_gpu_layers=1: {e}")
            
            print("\n" + "-" * 60)
            print("Test B: Loading with n_gpu_layers=0 (CPU only)...")
            try:
                llm = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_gpu_layers=0,  # CPU only
                    verbose=False
                )
                print("✓ Model loaded with n_gpu_layers=0 (CPU)")
                del llm
            except Exception as e:
                print(f"✗ Failed with n_gpu_layers=0: {e}")
        else:
            print("✗ No GGUF files found in cache")
    else:
        print("✗ No GGUF model found in HuggingFace cache")
        
except Exception as e:
    print(f"✗ Error during backend test: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check shared library dependencies
print("\n" + "=" * 60)
print("Checking shared library CUDA dependencies...")
print("=" * 60)

try:
    import llama_cpp
    import os
    from pathlib import Path
    
    # Find the llama_cpp shared library
    llama_cpp_dir = Path(llama_cpp.__file__).parent
    so_files = list(llama_cpp_dir.glob("*.so*"))
    
    if so_files:
        for so_file in so_files:
            print(f"\nChecking: {so_file.name}")
            result = os.popen(f"ldd {so_file} 2>/dev/null | grep -i 'cuda\\|cublas\\|cublasLt'").read()
            if result.strip():
                print("✓ CUDA libraries linked:")
                print(result)
            else:
                print("✗ No CUDA libraries linked (CPU-only build)")
    else:
        print("✗ No shared libraries found")
        
except Exception as e:
    print(f"✗ Error checking dependencies: {e}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("""
If you see:
- ✗ NO CUDA functions found → CPU-ONLY build, needs rebuild with CUDA
- ✓ CUDA functions found BUT all layers go to CPU → Configuration issue
- ✓ CUDA libraries linked → Has GPU support compiled in
""")
