import json
import os
import math
from openai import OpenAI

# --- CONFIGURATION ---
client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama", 
)

INPUT_DIR = "transcripts"
FINAL_DIR = "transcripts/final_edits"
CHUNK_SIZE_SEC = 900  # 15 minutes
OVERLAP_SEC = 60      # 1 minute overlap to maintain context
os.makedirs(FINAL_DIR, exist_ok=True)

def process_chunk(segments, chunk_index):
    """Sends a specific block of segments to the LLM."""
    chunk_text = ""
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        # Including timestamps helps the LLM understand the flow
        time_str = f"[{math.floor(seg['start']/60):02d}:{int(seg['start']%60):02d}]"
        chunk_text += f"{time_str} [{spk}]: {text}\n"

    print(f"🧠 Processing Chunk {chunk_index}...")

    prompt = f"""
    You are a professional script editor. Clean this 15-minute transcript block.
    
    TASKS:
    1. IDENTIFY ROLES: Rename generic 'SPEAKER_XX' to roles (OFFICER, DRIVER, SUSPECT, EMT, DOCTOR, PASSENGER, PASSERBY, WITNESS etc.) based on dialogue.
    2. CONSISTENCY: Ensure speaker names match across the dialogue.
    3. FIX HALLUCINATIONS: Correct obvious AI audio errors (e.g., 'glass of corn' -> 'glass partition').
    4. FORMAT: Return only the cleaned dialogue script. Do not add intro/outro text.

    TRANSCRIPT BLOCK {chunk_index}:
    {chunk_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error in chunk {chunk_index}: {e}"

def clean_large_transcript(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_segments = data.get("segments", [])
    if not all_segments:
        return "No segments found."

    total_duration = all_segments[-1]['end']
    num_chunks = math.ceil(total_duration / CHUNK_SIZE_SEC)
    
    full_cleaned_script = []

    for i in range(num_chunks):
        start_t = i * CHUNK_SIZE_SEC
        end_t = start_t + CHUNK_SIZE_SEC
        
        # Filter segments for this time window
        chunk_segments = [s for s in all_segments if s['start'] >= start_t and s['start'] < end_t]
        
        if chunk_segments:
            cleaned_block = process_chunk(chunk_segments, i + 1)
            full_cleaned_script.append(cleaned_block)

    return "\n\n--- NEXT SEGMENT ---\n\n".join(full_cleaned_script)

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    print(f"📂 Found {len(files)} JSON files in {INPUT_DIR}") # Debug line 1

    for file in files:
        print(f"🔎 Checking file: {file}") # Debug line 2
        output_name = f"{os.path.splitext(file)[0]}_CLEANED.txt"
        final_output_path = os.path.join(FINAL_DIR, output_name)
        
        if os.path.exists(final_output_path):
            print(f"⏭️ Skipping {file} (Already exists)")
            continue

        result = clean_large_transcript(os.path.join(INPUT_DIR, file))
        
        with open(final_output_path, "w") as f:
            f.write(result)
        print(f"✅ Full script finalized: {output_name}")