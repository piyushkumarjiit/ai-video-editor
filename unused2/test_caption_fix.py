#!/usr/bin/env python3
"""Test caption with fixed rotation and better prompt"""
import base64
from llama_cpp import Llama

print("Loading Qwen2-VL model...")
model = Llama(
    model_path="/home/mazsola/.cache/huggingface/hub/models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

test_image = "/home/mazsola/video/tmp_advanced3_frames/IMG_3520_test/frame_test_rotated.jpg"

# Extract frame from middle of video where painting should be visible (around 80 seconds)
import subprocess
subprocess.run(['ffmpeg', '-y', '-ss', '80', '-i', '/home/mazsola/video/IMG_3520.MOV', 
                '-vframes', '1', '-vf', 'transpose=2', '-q:v', '1', test_image], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Simple prompt - don't force "I see", just ask directly
prompt = "Describe what you see in this image: the car part being worked on, any tools visible, paint colors, and what the hands are doing."

print(f"\nTesting frame at 80 seconds (properly rotated, high quality):")
print(f"Prompt: {prompt}\n")

with open(test_image, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

output = model.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant that describes images of hobbyworkshops and scale model building. Always describe what you see."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ],
    max_tokens=200,
    temperature=0.2
)

caption = output['choices'][0]['message']['content'].strip()
print(f"Caption:\n{caption}\n")
