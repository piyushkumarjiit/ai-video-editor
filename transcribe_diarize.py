import os
import gc
import torch
import whisperx
import subprocess
import json
import warnings
import numpy as np
import pandas as pd
from scipy.io import wavfile
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from dotenv import load_dotenv
from scipy.spatial.distance import cdist

# --- CONFIGURATION ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda"
COMPUTE_TYPE = "float32" 
VIDEO_DIR = "samples/sanitized"
DENOISED_DIR = "samples/denoised"  # New Directory
OUTPUT_DIR = "transcripts"
DENOISE_PYTHON = "/home/piyush/.virtualenvs/denoise-env/bin/python"
DELETE_DENOISED_FILES = False

# --- ENVIRONMENT STABILIZATION ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet" 
warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DENOISED_DIR, exist_ok=True) # Ensure denoised folder exists

def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def identify_speakers(audio_file, diarize_df, hf_token):
    print("🔍 Phase 3.5: Matching speakers against library...")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
    inference = Inference(model, window="whole", device="cpu")
    
    library = {}
    if os.path.exists("speaker_library"):
        for f in os.listdir("speaker_library"):
            if f.endswith(".npy"):
                name = os.path.splitext(f)[0]
                library[name] = np.load(os.path.join("speaker_library", f))

    if not library:
        return {spk: spk for spk in diarize_df['speaker'].unique()}

    detected_speakers = diarize_df['speaker'].unique()
    refined_map = {}

    for spk_id in detected_speakers:
        spk_segments = diarize_df[diarize_df['speaker'] == spk_id]
        best_seg = spk_segments.iloc[(spk_segments['end'] - spk_segments['start']).argmax()]
        excerpt = Segment(best_seg.start, min(best_seg.start + 10, best_seg.end))
        live_embedding = inference.crop(audio_file, excerpt)

        best_match = spk_id
        min_dist = 0.5 
        for name, ref_embedding in library.items():
            dist = cdist(live_embedding.reshape(1, -1), ref_embedding.reshape(1, -1), metric="cosine")[0][0]
            if dist < min_dist:
                min_dist = dist
                best_match = name
        refined_map[spk_id] = best_match
    return refined_map

def run_isolated_denoise(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_wav = os.path.join(VIDEO_DIR, f"{base_name}_raw_tmp.wav")
    
    # 1. Extract audio
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path, 
        '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', 
        raw_wav
    ], check=True, capture_output=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" 
    
    try:
        print(f"🧹 Phase 0: Denoising {base_name} (Isolated CPU Mode)...")
        # Removed capture_output so you see the progress bar
        subprocess.run([
            DENOISE_PYTHON, "-m", "df.enhance", 
            raw_wav, "--output-dir", DENOISED_DIR
        ], env=env, check=True) 

        # 2. MATCH THE EXACT FILENAME SEEN IN YOUR LS COMMAND
        # DeepFilterNet adds the model name as a suffix
        enhanced_file = None
        expected_suffix = "_raw_tmp_DeepFilterNet3.wav"
        potential_path = os.path.join(DENOISED_DIR, f"{base_name}{expected_suffix}")

        if os.path.exists(potential_path):
            enhanced_file = potential_path
        else:
            # Fallback fuzzy search if the model version changes
            for file in os.listdir(DENOISED_DIR):
                if base_name in file and "DeepFilterNet" in file:
                    enhanced_file = os.path.join(DENOISED_DIR, file)
                    break
        
        if enhanced_file:
            # Cleanup raw temp only on success
            if os.path.exists(raw_wav):
                os.remove(raw_wav)
            return enhanced_file
        
        return raw_wav 
    except Exception as e:
        print(f"⚠️ Denoising failed: {e}")
        return raw_wav

def annotation_to_df(annotation):
    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": segment.start, "end": segment.end, "speaker": speaker})
    return pd.DataFrame(segments)

def process_video(video_path):
    video_file = os.path.basename(video_path)
    audio_file = run_isolated_denoise(video_path)
    
    try:
        print(f"📝 Phase 1 & 2: Transcribing {video_file}...")
        model = whisperx.load_model("medium", DEVICE, compute_type=COMPUTE_TYPE)
        audio_data = whisperx.load_audio(audio_file)
        result = model.transcribe(audio_data, batch_size=4)
        
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, DEVICE)
        
        del model, model_a
        flush_gpu()

        print("🧬 Phase 3: Diarizing...")
        pipeline = DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        pipeline.to(torch.device(DEVICE))
        
        sr, data = wavfile.read(audio_file)
        waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0) / 32768.0
        diarize_output = pipeline({"waveform": waveform, "sample_rate": sr})
        diarize_df = annotation_to_df(diarize_output)
        
        del pipeline
        flush_gpu()

        speaker_map = identify_speakers(audio_file, diarize_df, HF_TOKEN)
        diarize_df['speaker'] = diarize_df['speaker'].map(speaker_map)

        print("🏷️ Phase 4: Finalizing labels...")
        result = whisperx.assign_word_speakers(diarize_df, result)
        return result

    finally:
        # 1. Clean up the denoised file based on your new flag
        if os.path.exists(audio_file) and DELETE_DENOISED_FILES:
            print(f"🗑️ Deleting denoised file: {audio_file}")
            os.remove(audio_file)
        elif os.path.exists(audio_file):
            print(f"💾 Keeping denoised file at: {audio_file}")

        # 2. Safety: Always clean up the raw_tmp file if it somehow survived
        # (DeepFilterNet usually handles this, but this is a good backup)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_tmp = os.path.join(VIDEO_DIR, f"{base_name}_raw_tmp.wav")
        if os.path.exists(raw_tmp):
            os.remove(raw_tmp)

def save_transcript(result, video_name):
    base_name = os.path.splitext(video_name)[0]
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            f.write(f"[{speaker}]: {text}\n")
    print(f"✅ Saved JSON and TXT to: {OUTPUT_DIR}")

if __name__ == "__main__":
    videos = [v for v in os.listdir(VIDEO_DIR) if v.endswith(('.mp4', '.mkv'))]
    for video in videos:
        try:
            final_result = process_video(os.path.join(VIDEO_DIR, video))
            save_transcript(final_result, video)
        except Exception as e:
            print(f"❌ Error on {video}: {e}")