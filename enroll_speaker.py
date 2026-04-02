import os
import torch
import numpy as np
import shutil
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FOLDER = "samples/to_enroll"
LIBRARY_FOLDER = "speaker_library"
ARCHIVE_FOLDER = "samples/enrolled_archive"

# Ensure directories exist
for folder in [INPUT_FOLDER, LIBRARY_FOLDER, ARCHIVE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def process_enrollment_queue():
    """Loops through the input folder and generates .npy embeddings."""
    
    # Load model once for the whole batch
    print(f"📦 Loading Embedding Model on {DEVICE}...")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
    inference = Inference(model, window="whole", device=DEVICE)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not files:
        print("📭 No new samples found in samples/to_enroll.")
        return

    for filename in files:
        speaker_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(LIBRARY_FOLDER, f"{speaker_name}.npy")

        print(f"🧬 Extracting fingerprint for: {speaker_name}...")
        
        try:
            # We use the whole file to get the best possible average 'voice print'
            embedding = inference(input_path)
            
            # Save as a mathematical vector (.npy)
            np.save(output_path, embedding)
            
            # Move original file to archive so we don't re-process it
            shutil.move(input_path, os.path.join(ARCHIVE_FOLDER, filename))
            print(f"✅ Success! Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    process_enrollment_queue()