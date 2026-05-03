import torch
import tensorrt as trt
import os
import glob

# --- CONFIGURATION ---
MODELS_DIR = "/home/pk/ai-video-editor/models"
# Define the resolutions we want to target
RESOLUTIONS = {
    "720p": (1, 3, 720, 1280),
    "1080p": (1, 3, 1080, 1920)
}

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def convert_models():
    pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    
    if not pt_files:
        print("No .pt files found.")
        return

    for pt_path in pt_files:
        base_name = os.path.basename(pt_path).replace(".pt", "")
        
        # Load the model once
        print(f"\n--- Loading Model: {base_name} ---")
        try:
            model = torch.load(pt_path)
            model.eval().cuda()
        except Exception as e:
            print(f"❌ Failed to load {pt_path}: {e}")
            continue

        for res_name, shape in RESOLUTIONS.items():
            print(f"\nTargeting {res_name} ({shape[2]}x{shape[3]})...")
            
            onnx_path = os.path.join(MODELS_DIR, f"{base_name}_{res_name}.onnx")
            engine_path = os.path.join(MODELS_DIR, f"{base_name}_{res_name}_1080ti.engine")

            # 1. Export to ONNX for this resolution
            try:
                dummy_input = torch.randn(*shape).cuda()
                torch.onnx.export(model, dummy_input, onnx_path, ops_version=11, 
                                  do_constant_folding=True, 
                                  input_names=['input'], output_names=['output'])
                print(f"✅ ONNX Exported: {res_name}")
            except Exception as e:
                print(f"❌ ONNX Export Failed: {e}")
                continue

            # 2. Build TensorRT Engine
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print("❌ Parser failed")
                    continue

            # Optimize for FP32 (Best for 1080 Ti)
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine:
                with open(engine_path, 'wb') as f:
                    f.write(serialized_engine)
                print(f"🚀 Engine Created: {os.path.basename(engine_path)}")
            
            # Cleanup ONNX to keep the folder tidy
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

if __name__ == "__main__":
    convert_models()