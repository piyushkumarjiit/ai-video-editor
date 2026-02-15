# Building llama-cpp-python with GPU Support on Fedora 43

**Complete guide based on actual successful build on Fedora 43 with NVIDIA RTX A3000 12GB**

## Problem Statement

**System:**
- OS: Fedora 43 with GCC 15.2.1 and glibc 2.40
- CUDA: 12.4.131 (supports GCC up to 12.x only)
- GPU: NVIDIA RTX A3000 12GB Laptop GPU
- Python: 3.14

**Issues:**
1. ❌ CUDA 12.4 does not support GCC 15 (system default on Fedora 43)
2. ❌ glibc 2.40 requires `noexcept` on math functions, but CUDA headers lack them
3. ❌ Prebuilt `llama-cpp-python` wheels have no CUDA support

**Solution:** Compile GCC 12, patch CUDA headers, build from source

---

## Prerequisites

```bash
# Install build dependencies
sudo dnf install -y wget tar bzip2 gcc gcc-c++ make \
    libmpc-devel mpfr-devel gmp-devel \
    zlib-devel flex bison texinfo \
    ninja-build cmake git python3-devel
```

---

## Step 1: Compile GCC 12.3.0 from Source

CUDA 12.4 requires GCC ≤ 12.x. We compile GCC 12.3.0 to `/opt/gcc-12`.

### Using the Provided Script (Recommended)

```bash
chmod +x tools/install_gcc12.sh
sudo ./tools/install_gcc12.sh
```

**Build time:** ~40 minutes on modern CPU

### Verify Installation

```bash
/opt/gcc-12/bin/gcc --version
```

**Expected output:**
```
gcc (GCC) 12.3.0
```

### What the Script Does

The `install_gcc12.sh` script:
1. Downloads GCC 12.3.0 source from gnu.org
2. Configures with C/C++ only (faster build, no Fortran/Ada)
3. Compiles using all CPU cores (`make -j$(nproc)`)
4. Installs to `/opt/gcc-12`

---

## Step 2: Patch CUDA Math Headers

glibc 2.40 declares math functions with `noexcept`, but CUDA 12.4 headers don't. This causes compilation conflicts.

### Using the Provided Script (Recommended)

```bash
chmod +x tools/patch_cuda_math.sh
sudo ./tools/patch_cuda_math.sh
```

### Verify Patches Applied

```bash
grep -n "noexcept" /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
```

**Expected output** (6 functions patched):
```
103:double                 cospi(double x) noexcept;
109:double                 sinpi(double x) noexcept;
111:double                 rsqrt(double x) noexcept;
191:float                  cospif(float x) noexcept;
197:float                  sinpif(float x) noexcept;
199:float                  rsqrtf(float x) noexcept;
```

### What the Script Does

The `patch_cuda_math.sh` script:
1. Backs up original file to `.backup`
2. Adds `noexcept` to 6 math functions using `sed`:
   - `cospi`, `sinpi`, `rsqrt` (double precision)
   - `cospif`, `sinpif`, `rsqrtf` (float precision)
3. Verifies patches with `grep`

### Restore Original (if needed)

```bash
sudo cp /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h.backup \
        /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
```

---

## Step 3: Build llama-cpp-python with CUDA Support

Now build llama-cpp-python from source using GCC 12 and patched CUDA headers.

### Using the Provided Script (Recommended)

```bash
# Activate your Python environment
source .venv/bin/activate

# Run the build script
chmod +x tools/build_llama_cpp_with_gcc12.sh
./tools/build_llama_cpp_with_gcc12.sh
```

**Build time:** ~5-10 minutes (183 compilation steps)

### What the Script Does

The `build_llama_cpp_with_gcc12.sh` script:
1. Checks GCC 12 is installed at `/opt/gcc-12`
2. Uninstalls any existing llama-cpp-python
3. Sets environment variables to force GCC 12 usage:
   - `CUDAHOSTCXX=/opt/gcc-12/bin/g++`
   - `PATH` with GCC 12 first
   - C++ include paths for GCC 12 headers
4. Configures CMake with CUDA enabled:
   - `GGML_CUDA=on`
   - `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
   - `CMAKE_CUDA_HOST_COMPILER=/opt/gcc-12/bin/g++`
   - `-allow-unsupported-compiler` flag
5. Builds and installs llama-cpp-python from source

### Verify CUDA Support

Check if CUDA libraries are linked:

```bash
source .venv/bin/activate
python -c "from llama_cpp import Llama; import os, subprocess; lib = os.path.join(os.path.dirname(Llama.__file__), 'libggml-cuda.so'); print(f'Checking: {lib}'); subprocess.run(['ldd', lib])"
```

**Expected output** (should show CUDA libraries):
```
libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12
libcublas.so.12 => /usr/local/cuda/lib64/libcublas.so.12
libcuda.so.1 => /lib64/libcuda.so.1
libcublasLt.so.12 => /usr/local/cuda/lib64/libcublasLt.so.12
```

---

## Step 4: Test GPU Acceleration

### Using the Provided Test Script

```bash
source .venv/bin/activate
python tools/test_video_gpu.py
```

### What the Test Does

The `test_video_gpu.py` script:
1. Detects GPU and CUDA version
2. Loads Qwen2-VL-7B GGUF model with **all layers on GPU** (`n_gpu_layers=-1`)
3. Extracts a test frame from video
4. Runs inference on the frame
5. Reports performance metrics

**Expected output:**
```
======================================================================
🎬 Video Frame Analysis Test with GPU
======================================================================
💎 GPU: NVIDIA RTX A3000 12GB Laptop GPU, 12288 MiB
🤖 Loading Qwen2-VL-7B GGUF model...
load_tensors: offloaded 29/29 layers to GPU
load_tensors: CUDA0 model buffer size = 4168.09 MiB
✓ Model loaded in 0.9s
📹 Extracting test frame from IMG_3520.MOV...
✓ Saved test frame: /tmp/test_frame.jpg
🧠 Running inference...
llama_perf_context_print: prompt eval time = 94.37 ms / 33 tokens (349.67 tokens per second)
llama_perf_context_print: eval time = 272.83 ms / 15 runs (54.98 tokens per second)
✓ Inference completed in 0.4s
📝 Response:
A person is holding a camera while standing in front of a blurred background.
======================================================================
✓ Test complete! Model is using GPU for inference.
======================================================================
```

### Key Indicators of Success

✅ **All 29 layers on GPU:**
```
load_tensors: offloaded 29/29 layers to GPU
load_tensors: CUDA0 model buffer size = 4168.09 MiB
```

✅ **High performance:**
- Prompt evaluation: **350 tokens/sec**
- Text generation: **55 tokens/sec**
- Inference time: **0.4 seconds**

---

## Critical Configuration: n_gpu_layers

When loading models, **always use `n_gpu_layers=-1`** for maximum GPU performance:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="/path/to/model.gguf",
    n_gpu_layers=-1,  # ⚠️ IMPORTANT: -1 = all layers on GPU
    n_ctx=4096,
    verbose=True
)
```

**Parameter meanings:**
- `n_gpu_layers=-1` → All layers on GPU ✅ **(Use this!)**
- `n_gpu_layers=1` → Only 1 layer on GPU ❌ (very slow)
- `n_gpu_layers=0` → CPU-only ❌ (very slow)

---

## Performance Benchmarks

**Hardware:** NVIDIA RTX A3000 12GB, Fedora 43, Python 3.14  
**Model:** Qwen2-VL-7B-Instruct-Q4_K_M.gguf (4.36 GB)

| Configuration | Layers on GPU | GPU Memory | Prompt Speed | Generation Speed |
|---------------|---------------|------------|--------------|------------------|
| CPU-only | 0/29 | 0 MB | ~50 tok/s | ~10 tok/s |
| Partial | 1/29 | ~150 MB | ~100 tok/s | ~20 tok/s |
| **Full GPU** ✅ | **29/29** | **4,200 MB** | **350 tok/s** | **55 tok/s** |

---

## Troubleshooting

### Issue: "offloaded 1/29 layers" or "layers assigned to CPU"

**Cause:** Using `n_gpu_layers=1` instead of `-1`

**Solution:** Change to `n_gpu_layers=-1` in your Python code

### Issue: Compilation errors with "unsupported compiler"

**Cause:** CUDA 12.4 doesn't recognize GCC 15

**Solution:** Follow Step 1 to compile GCC 12.3.0

### Issue: Compilation errors with "noexcept" mismatches

**Cause:** glibc 2.40 requires `noexcept`, CUDA headers don't have it

**Solution:** Follow Step 2 to patch CUDA headers

### Issue: Build completes but no GPU acceleration

**Cause:** Environment variables not set correctly during build

**Solution:**
1. Uninstall: `pip uninstall -y llama-cpp-python`
2. Re-run: `./build_llama_cpp_with_gcc12.sh`
3. Verify with: `ldd` command from Step 3

### Monitor GPU Usage

```bash
# Watch GPU in real-time during inference
watch -n 0.5 nvidia-smi
```

---

## Files Reference

### Scripts Provided

| Script | Purpose | Build Time |
|--------|---------|------------|
| `install_gcc12.sh` | Compile GCC 12.3.0 from source | ~40 minutes |
| `patch_cuda_math.sh` | Add `noexcept` to 6 CUDA math functions | <1 second |
| `build_llama_cpp_with_gcc12.sh` | Build llama-cpp-python with CUDA | ~5-10 minutes |
| `test_video_gpu.py` | Verify GPU acceleration and performance | ~1 second |

### Key Files Modified

- `/opt/gcc-12/` - GCC 12.3.0 installation directory
- `/opt/gcc-12/bin/gcc` - GCC 12 compiler
- `/opt/gcc-12/bin/g++` - GCC 12 C++ compiler
- `/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h` - Patched CUDA headers
- `.venv/lib/python3.14/site-packages/llama_cpp/libggml-cuda.so` - Built with CUDA support

---

## Real-World Usage

After successful setup, the video analysis pipeline runs with full GPU acceleration:

```bash
source .venv/bin/activate
python analyze_advanced3.py
```

**Results on IMG_3520.MOV (209s, 4K video):**
```
💎 GPU: NVIDIA RTX A3000 12GB Laptop GPU (11.6GB)
   CUDA: 12.8, PyTorch: 2.10.0+cu128
✓ CLIP ViT-B/32 loaded on CUDA (0.33GB GPU memory)
✓ ResNet-50 loaded on CUDA (0.42GB GPU memory)
✓ Qwen2-VL-7B GGUF loaded on GPU (4.2GB, All layers on GPU)
📝 Captioning kept keyframes...
Captioning: 100%|███████████| 104/104 [01:24<00:00, 1.24frame/s]
✓ Captioning complete
```

**Performance:** 1.24 frames/sec captioning speed with all models on GPU

---

## Summary of What We Achieved

1. ✅ Compiled GCC 12.3.0 from source (~40 min)
2. ✅ Patched 6 CUDA math functions with `noexcept`
3. ✅ Built llama-cpp-python with full CUDA support (183 steps)
4. ✅ Verified all 29 model layers running on GPU
5. ✅ Achieved 350/55 tokens/sec (7x faster than CPU)
6. ✅ Successfully processed 4K video with GPU-accelerated AI models

---

## Additional Resources

- **CUDA Toolkit:** https://docs.nvidia.com/cuda/
- **llama-cpp-python:** https://github.com/abetlen/llama-cpp-python
- **GGUF Models:** https://huggingface.co/models?library=gguf
- **GCC Releases:** https://ftp.gnu.org/gnu/gcc/

---

**Last Updated:** February 2026  
**Tested Configuration:** Fedora 43, CUDA 12.4.131, Python 3.14, NVIDIA RTX A3000 12GB  
**Status:** ✅ Fully working with GPU acceleration
