#!/usr/bin/env python3
"""Test simple prompt on one frame"""
import base64
from llama_cpp import Llama

# Load model
print("Loading model...")
model = Llama(
    model_path="/home/mazsola/.cache/huggingface/hub/models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

# Test frame
frame_path = "/home/mazsola/video/tmp_advanced3_frames/IMG_3520/frame_0017.jpg"

print(f"Testing: {frame_path}")

# Load image
with open(frame_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Force 'I see' start, specific to scale modeling
prompt = "Start with 'I see' and describe this scale model building scene: which model car part (body, seat, chassis, wheels, interior pieces), modeling tools (hobby knife, tweezers, brush, file, airbrush), materials (paint, glue, decals), what hands are doing"

# Generate
output = model.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ],
    max_tokens=100,
    temperature=0.1,
)

caption = output['choices'][0]['message']['content'].strip()
print(f"\nCaption:\n{caption}")
