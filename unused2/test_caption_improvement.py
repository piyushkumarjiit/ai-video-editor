#!/usr/bin/env python3
"""Test improved caption recognition on sample frames."""

import sys
import base64
from llama_cpp import Llama
from PIL import Image

def test_caption(model, image_path, frame_num):
    """Test caption for a single frame."""
    print(f"\n{'='*70}")
    print(f"🖼️  Testing Frame {frame_num}: {image_path}")
    print('='*70)
    
    # Enhanced prompt for model building workflow recognition
    prompt = """Analyze this scale model building image and describe:

1. **Workflow Stage**: Identify which stage (preparation/cleanup, assembly/gluing, painting/priming, detailing/weathering, or final touches)
2. **Central Object**: What model part or tool is in focus? (body, chassis, wheels, suspension, engine, interior, decals, paint bottles, brushes, airbrush, sprue cutters, files, sandpaper, etc.)
3. **Action/Process**: What is being done? (cutting parts from sprue, test fitting, gluing, sanding seams, masking, priming, base coating, detail painting, applying decals, weathering, clear coating, polishing)
4. **Hands/Tools Visible**: Describe any hands holding parts/tools and what they're manipulating
5. **Materials**: Identify materials visible (plastic parts, photo-etch metal, resin, paint, glue, putty, tape)

Be specific and technical. If the image is blurry or unclear, state "Image unclear - cannot identify details" and stop."""
    
    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Generate caption
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
        max_tokens=200,
        temperature=0.3,
    )
    
    caption = output['choices'][0]['message']['content'].strip()
    
    print(f"\n📝 Caption:\n{caption}\n")
    return caption

def main():
    print("="*70)
    print("🧪 Testing Improved Model Building Caption Recognition")
    print("="*70)
    
    # Load model
    print("\n🤖 Loading Qwen2-VL-7B GGUF model...")
    model_path = "/home/mazsola/.cache/huggingface/hub/models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
    
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="qwen2-vl"
    )
    print("✓ Model loaded\n")
    
    # Test frames that had issues or generic descriptions
    test_frames = [
        ("tmp_advanced3_frames/IMG_3520/frame_0001.jpg", 1, "Previous: Generic assembly description"),
        ("tmp_advanced3_frames/IMG_3520/frame_0008.jpg", 8, "Previous: Generic car + tools description"),
        ("tmp_advanced3_frames/IMG_3520/frame_0016.jpg", 16, "Previous: Screwdriver work description"),
        ("tmp_advanced3_frames/IMG_3520/frame_0017.jpg", 17, "Previous: Vintage model car body"),
        ("tmp_advanced3_frames/IMG_3520/frame_0041.jpg", 41, "Previous: Yellow plastic car"),
    ]
    
    results = []
    for frame_path, frame_num, prev_caption in test_frames:
        print(f"\n📋 Previous caption: {prev_caption}")
        caption = test_caption(model, frame_path, frame_num)
        results.append({
            "frame": frame_num,
            "path": frame_path,
            "previous": prev_caption,
            "new": caption
        })
    
    print("\n" + "="*70)
    print("📊 Summary of Improvements")
    print("="*70)
    for r in results:
        print(f"\nFrame {r['frame']}:")
        print(f"  Before: {r['previous'][:60]}...")
        print(f"  After:  {r['new'][:60]}...")
    
    print("\n✓ Test complete!")

if __name__ == "__main__":
    main()
