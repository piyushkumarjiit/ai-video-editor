"""
FILE: clean_transcript.py
ROLE: AI-Driven Script Polishing & Speaker Profiling
-------------------------------------------------------------------------
DESCRIPTION:
The final refinement stage. It cleans raw transcripts by removing verbal 
fillers (um/uh), fixing grammar, and mapping generic 'SPEAKER_XX' tags 
to logical roles (OFFICER, SUSPECT, etc.) based on dialogue context.

INPUT: transcripts/*.json (Output from transcribe_diarize.py).
OUTPUT: transcripts/final_edits/*_FINAL.txt (Polished human-readable script).

HARDWARE COMPATIBILITY:
- Connects to local Ollama API (qwen3.5:9b).
- Uses chunk-based processing to handle very long videos without OOM.
-------------------------------------------------------------------------
"""

import json
import os
import math
import re
import inspect
from openai import OpenAI

# --- CONFIGURATION ---
INPUT_DIR = "transcripts"
FINAL_DIR = "transcripts/final_edits"
CHUNK_SIZE_SEC = 120  # 7.5 minutes

MODEL_CONFIG = {
    "name": "qwen3.5:9b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "num_ctx": 2048,        
    "num_predict": 1024,     
    "temperature_janitor": 0.1,
    "temperature_profiler": 0.3,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.5,
    "think": False,
    "top_k": 40,
    "top_p": 0.9
}

client = OpenAI(
    base_url=MODEL_CONFIG["base_url"], 
    api_key=MODEL_CONFIG["api_key"], 
    timeout=300.0  # Increase timeout to 5 minutes
)

os.makedirs(FINAL_DIR, exist_ok=True)

def strip_thinking(text):
    if not text: 
        return ""
    
    # 1. Normalize tags to handle potential model case-drifting
    # This ensures <THINK> or <Think> are caught by the regex
    has_tags = re.search(rf"<think>", text, re.IGNORECASE) and re.search(rf"</think>", text, re.IGNORECASE)
    
    # 2. Extract content OUTSIDE of tags (The standard behavior)
    # Using re.IGNORECASE to catch variations
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 3. SAFETY: Handle "Unclosed Tags"
    # If the model crashed or hit a token limit, it might leave an open <think> with no end.
    # This strips everything after a lone <think> tag.
    if "<think" in clean_text.lower():
        clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 4. FALLBACK: If we are left with nothing, the model put the answer INSIDE the tags.
    if not clean_text and has_tags:
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            clean_text = match.group(1).strip()
    
    # 5. FINAL SAFETY: Remove literal tag strings and strip markdown code blocks
    # Sometimes Ollama wraps the output in ```text ... ```
    if not clean_text:
        clean_text = text.replace("<think>", "").replace("</think>", "").strip()
    
    # Final Polish: Remove leading/trailing markdown code fences if the model added them
    clean_text = re.sub(r'^```[a-zA-Z]*\n?', '', clean_text)
    clean_text = re.sub(r'\n?```$', '', clean_text)
        
    return clean_text.strip()


def process_chunk(segments, chunk_index):
    """Pass 1: Clean grammar and remove fillers."""
    chunk_text = ""
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        time_str = f"[{math.floor(seg['start']/60):02d}:{int(seg['start']%60):02d}]"
        chunk_text += f"{time_str} [{spk}]: {text}\n"

    print(f"🧹 Janitor Pass: Cleaning Chunk {chunk_index}...")

    # Simplified, non-indented prompt to prevent the model from getting "lost"
    #prompt = f"TASK: Clean the following transcript. Fix grammar, remove 'um/uh', and keep tags exactly as they are. Output ONLY the cleaned transcript lines.\n\nDATA:\n{chunk_text}\n\nCLEANED OUTPUT:\n["
    prompt = inspect.cleandoc(f"""
    TASK: Clean this transcript. Fix grammar and remove filler words.
    CRITICAL: If a speaker repeats a command, keep it, but DO NOT loop the text indefinitely.
    If the transcript ends, STOP writing immediately. Do not invent new speakers like 'SAI' or 'response'.

    DATA:
    {chunk_text}

    CLEANED OUTPUT:
    """)
    # Old messages content --> {"role": "user", "content": prompt}
    try:
        #print(f"Text being passed: {chunk_text}")
        response = client.chat.completions.create(
            model=MODEL_CONFIG["name"],
            messages=[                
                {"role": "system", "content": "You are a transcript editor. Output only the cleaned text, preserving all [HH:MM] and [SPEAKER_XX] tags. CRITICAL: If the transcript ends, STOP writing immediately. Do not invent new speakers"},
                {"role": "user", "content": f"Clean this transcript by removing fillers and fixing grammar:\n\n{chunk_text}"}
                ],

            temperature=0.1,
            extra_body={
                "num_ctx": 2048,       # Dropping context slightly for better speed/focus
                "num_predict": 1024,
                "num_thread": 16,    
                "think": False
            }
        )
        content = response.choices[0].message.content
        
        # We forced an opening bracket '[' in the prompt, so we add it back if it's missing
        #if content and not content.startswith('['):
        #    content = '[' + content
            
        print(f"--- DEBUG CHUNK {chunk_index} ---")
        print(f"RAW CONTENT FROM MODEL: '{content}'")
        print(f"DEBUG RAW OUTPUT (First 50 chars): {content[:50].replace('\\n', ' ')}...")
        
        cleaned = strip_thinking(content)
        return cleaned
    except Exception as e:
        print(f"❌ Error in chunk {chunk_index}: {e}")
        return ""


def generate_speaker_map(full_cleaned_text):
    """Pass 2: Identify roles using situational logic."""
    if not full_cleaned_text:
        return {}
        
    print("🧠 Profiler Pass: Analyzing speaker roles...")
    text_len = len(full_cleaned_text)
    snip_size = min(4000, text_len // 3)
    
    start_snip = full_cleaned_text[:snip_size]
    mid_start = text_len // 2
    mid_snip = full_cleaned_text[mid_start : mid_start + snip_size]
    end_snip = full_cleaned_text[-snip_size:]
    
    combined_context = f"{start_snip}\n...[CLIP]...\n{mid_snip}\n...[CLIP]...\n{end_snip}"
    
    prompt = f"""
### TASK ###
Map each SPEAKER_ID to a specific ROLE. 
Roles: NARRATOR, OFFICER_n, DISPATCHER, SUSPECT_n, VICTIM_n, WITNESS_n, EMS_n.

### SNIPPETS ###
{combined_context}

### OUTPUT FORMAT ###
Reasoning: [Your thoughts]
JSON:
{{ "SPEAKER_XX": "ROLE" }}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_CONFIG["name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_CONFIG["temperature_profiler"],
            extra_body={
                "num_ctx": MODEL_CONFIG["num_ctx"],
                "num_predict": 2048,
                "think": MODEL_CONFIG["think"]
            }
        )
        raw_output = strip_thinking(response.choices[0].message.content)
        print(f"DEBUG REASONING:\n{raw_output}")
        
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        print(f"⚠️ Failed to generate dynamic map: {e}")
        return {}

def apply_tags(text, speaker_map):
    """Pass 3: Global replacement of IDs with Roles."""
    if not speaker_map:
        return text
        
    print("🏷️ Tagger Pass: Applying roles...")
    sorted_keys = sorted(speaker_map.keys(), key=len, reverse=True)
    
    for spk_id in sorted_keys:
        role = speaker_map[spk_id].strip().upper().replace(" ", "_")
        base_id = spk_id.replace("SPEAKER_", "").lstrip("0") or "0"
        # Regex to handle [SPEAKER_01], [SPEAKER_1], etc.
        pattern = rf"(?i)\[SPEAKER_0*{base_id}\]"
        text = re.sub(pattern, f"[{role}]", text)
        
    return text

def clean_large_transcript(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_segments = data.get("segments", [])
    if not all_segments: return "No segments found."

    total_duration = all_segments[-1]['end']
    num_chunks = math.ceil(total_duration / CHUNK_SIZE_SEC)
    
    full_cleaned_list = []
    for i in range(num_chunks):
        start_t = i * CHUNK_SIZE_SEC
        chunk_segments = [s for s in all_segments if start_t <= s['start'] < (start_t + CHUNK_SIZE_SEC)]
        if chunk_segments:
            cleaned_chunk = process_chunk(chunk_segments, i + 1)
            if cleaned_chunk:
                full_cleaned_list.append(cleaned_chunk)

    if not full_cleaned_list:
        return "ERROR: All chunks returned empty. Check model output and think tags."

    full_cleaned_text = "\n\n".join(full_cleaned_list)
    speaker_map = generate_speaker_map(full_cleaned_text)
    final_script = apply_tags(full_cleaned_text, speaker_map)

    return final_script

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    for file in files:
        print(f"🚀 STARTING: {file}")
        output_name = f"{os.path.splitext(file)[0]}_FINAL.txt"
        final_output_path = os.path.join(FINAL_DIR, output_name)

        result = clean_large_transcript(os.path.join(INPUT_DIR, file))
        
        with open(final_output_path, "w", encoding='utf-8') as f:
            f.write(result)
        print(f"✅ SAVED: {output_name}\n")