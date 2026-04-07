"""
FILE: denoise_worker.py
ROLE: DeepFilterNet Execution Engine
-------------------------------------------------------------------------
DESCRIPTION:
The actual processing core for audio cleanup. It runs the DeepFilterNet3 
model to strip environmental noise while preserving speech. Designed to 
run in a dedicated venv to prevent dependency hell.

INPUT: temp_raw.wav (High-fidelity raw audio).
OUTPUT: temp_clean.wav (Enhanced audio).
-------------------------------------------------------------------------
"""

import sys
from df.enhance import enhance, init_df, load_audio, save_audio

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"

def run_denoise(input_file, output_file):
    model, state, _ = init_df()
    audio, _ = load_audio(input_file, sr=state.sr())
    enhanced = enhance(model, state, audio)
    save_audio(output_file, enhanced, state.sr())

if __name__ == "__main__":
    run_denoise(sys.argv[1], sys.argv[2])