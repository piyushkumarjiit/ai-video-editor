from llama_cpp import Llama
import sys

print("🎬 Initializing AI Video Editor Hello World...")

try:
    # We set n_gpu_layers to -1 to offload EVERYTHING to your 1080 Ti
    llm = Llama(
        model_path="models/test-model.gguf",
        n_gpu_layers=-1, 
        verbose=True # This is key: it shows the GPU logs
    )

    prompt = "Q: What is the best way to edit a video? A:"
    
    print("\n🤖 Thinking...")
    output = llm(prompt, max_tokens=50, echo=True)
    
    print("\n--- AI RESPONSE ---")
    print(output["choices"][0]["text"])
    print("-------------------\n")
    
    print("✅ Hardware Test Passed!")

except Exception as e:
    print(f"\n❌ Test Failed: {e}")
    sys.exit(1)