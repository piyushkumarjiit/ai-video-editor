#!/usr/bin/env python3
"""Quick test of improved caption prompt."""

from llama_cpp import Llama
import base64

print("Loading model...")
model = Llama(
    model_path='/home/mazsola/.cache/huggingface/hub/models--bartowski--Qwen2-VL-7B-Instruct-GGUF/snapshots/3088669af444bb2b86da6272694edd905f9c5a5b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf',
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False
)

prompt = """Describe this scale model car building scene with maximum detail and specificity:

**1. IDENTIFY THE MAIN CAR PART** (be very specific):
- Exterior: body shell, hood, fender, door, trunk, roof, bumper, grille, headlight, taillight, side mirror
- Wheels/Suspension: wheel, tire, rim, brake disc, caliper, suspension arm, shock absorber, axle
- Interior: dashboard, steering wheel, seat (driver/passenger), center console, door panel, gear shifter, instrument cluster, roll cage
- Engine/Mechanical: engine block, transmission, exhaust pipe, radiator, battery, wiring harness
- Small parts: decal sheet, photo-etch piece, clear parts (windows), chrome parts

**2. DESCRIBE THE CURRENT ACTIVITY**:
- Part preparation: removing from sprue with cutters, cleaning mold lines, test fitting
- Surface work: sanding with sandpaper/file, filling gaps with putty, scribing details
- Assembly: applying glue, clamping parts together, aligning components
- Pre-paint: masking areas with tape, applying primer coat, cleaning surface
- Painting: airbrushing color, brush painting details, applying base/top coat
- Detailing: positioning decals, applying panel line wash, dry brushing, weathering with pigments
- Finishing: applying clear coat, polishing, final assembly and inspection

**3. WHAT'S IN THE HANDS** (if hands visible):
Describe exactly what the hands are holding and manipulating

**4. TOOLS & MATERIALS VISIBLE**:
List any tools (airbrush, brushes, knives, files, tweezers, clamps) and materials (paint bottles, glue, tape, putty)

Keep it factual and technical. If image is blurry/unclear, state "Image unclear - cannot identify specific details" and stop."""

# Test two frames
test_frames = [
    ('tmp_advanced3_frames/IMG_3520/frame_0001.jpg', 'Frame 1 (2s)'),
    ('tmp_advanced3_frames/IMG_3520/frame_0017.jpg', 'Frame 17 (34s)'),
]

for frame_path, label in test_frames:
    print(f"\n{'='*70}")
    print(f"Testing {label}")
    print('='*70)
    
    with open(frame_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    output = model.create_chat_completion(
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_data}'}}
        ]}],
        max_tokens=200,
        temperature=0.3,
    )
    
    print(output['choices'][0]['message']['content'])

print("\n" + "="*70)
print("✓ Test complete")
