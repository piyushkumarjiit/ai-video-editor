import os
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- CONFIGURATION ---
SANITIZED_DIR = "samples/sanitized"
TRANSCRIPT_DIR = "transcripts" # Store them here to keep 'sanitized' clean
MODEL_SIZE = "base"            # 'base' is a great speed/accuracy balance
DEVICE = "cuda"                # Change to 'cpu' if Qwen is running simultaneously
COMPUTE_TYPE = "int8"       # Optimized for GTX 1080 Ti

def run_whisper_pipeline():
    # Ensure directories exist
    if not os.path.exists(SANITIZED_DIR):
        print(f"❌ Error: {SANITIZED_DIR} folder not found.")
        return
    
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    # 1. Initialize Whisper once to keep it in VRAM
    print(f"🤖 Loading Whisper ({MODEL_SIZE}) on {DEVICE}...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    # 2. Find videos that need transcription
    videos = [f for f in os.listdir(SANITIZED_DIR) if f.endswith(('.mp4', '.mkv', '.mov'))]
    
    for video_file in videos:
        base_name = os.path.splitext(video_file)[0]
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")

        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            continue

        print(f"\n🎵 Processing: {video_file}")
        video_path = os.path.join(SANITIZED_DIR, video_file)
        audio_temp = f"temp_{base_name}.mp3"

        try:
            # 3. Extract Audio
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_temp)
            video.close()

            # 4. Transcribe
            segments, _ = model.transcribe(audio_temp, beam_size=5)
            full_transcript = " ".join([segment.text for segment in segments]).strip()

            # 5. Save Transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(full_transcript)
            
            print(f"✅ Saved transcript to: {transcript_path}")

        except Exception as e:
            print(f"⚠️ Failed to process {video_file}: {e}")

        finally:
            if os.path.exists(audio_temp):
                os.remove(audio_temp)

    print("\n✨ Transcription sync complete.")

if __name__ == "__main__":
    run_whisper_pipeline()