import json
import os
import re
from presidio_analyzer import AnalyzerEngine

# 1. Initialize the Analyzer
analyzer = AnalyzerEngine()

# CONFIGURATION
TRANSCRIPT_DIR = "transcripts"
REDACTION_DIR = "redactions"

# Ensure directories exist
os.makedirs(REDACTION_DIR, exist_ok=True)

def clean_for_match(text):
    """Deep clean: remove all non-alphanumeric characters and lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def map_pii_to_timestamps(filename):
    file_path = os.path.join(TRANSCRIPT_DIR, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # WhisperX JSON structure: 'text' for full transcript, 'segments' for words
    full_text = data.get("text", "")
    # Fallback: if 'text' is missing, reconstruct it from segments
    if not full_text:
        full_text = " ".join([s.get("text", "") for s in data.get("segments", [])])

    # Run Presidio Analysis
    results = analyzer.analyze(text=full_text, language='en', 
                                entities=["PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS"])

    mute_manifest = []

    for pii in results:
        original_pii_text = full_text[pii.start:pii.end]
        pii_clean = clean_for_match(original_pii_text)
        
        if not pii_clean:
            continue
        
        pii_start_time = None
        pii_end_time = None
        
        # Search through segments and words
        for segment in data.get("segments", []):
            for word_info in segment.get("words", []):
                word_raw = word_info.get('word', '')
                word_clean = clean_for_match(word_raw)
                
                if not word_clean:
                    continue

                # FUZZY MATCH: Checks if PII is in word or word is in PII
                # This handles "selling," vs "selling" and "691" vs "6-9-1-8-9"
                if pii_clean in word_clean or word_clean in pii_clean:
                    w_start = word_info.get('start')
                    w_end = word_info.get('end')
                    
                    if w_start is not None:
                        if pii_start_time is None or w_start < pii_start_time:
                            pii_start_time = w_start
                    if w_end is not None:
                        if pii_end_time is None or w_end > pii_end_time:
                            pii_end_time = w_end

        if pii_start_time is not None:
            mute_manifest.append({
                "entity": pii.entity_type,
                "text": original_pii_text,
                "start": round(pii_start_time, 3),
                "end": round(pii_end_time, 3),
                "confidence": round(pii.score, 2)
            })

    return mute_manifest

def get_pending_files():
    """Finds metadata files that don't have a manifest yet."""
    all_files = os.listdir(TRANSCRIPT_DIR)
    metadata_files = [f for f in all_files if f.endswith("_metadata.json")]
    
    pending = []
    for meta in metadata_files:
        manifest_name = meta.replace("_metadata.json", "_mute_manifest.json")
        # Check if already processed
        if not os.path.exists(os.path.join(REDACTION_DIR, manifest_name)):
            pending.append(meta)
    return pending

if __name__ == "__main__":
    pending = get_pending_files()
    
    if not pending:
        print(f"📭 No new files to process in '{TRANSCRIPT_DIR}'.")
    else:
        for meta_file in pending:
            print(f"🕵️  Scanning {meta_file} for PII...")
            try:
                manifest = map_pii_to_timestamps(meta_file)
                
                output_filename = meta_file.replace("_metadata.json", "_mute_manifest.json")
                output_path = os.path.join(REDACTION_DIR, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(manifest, f, indent=4)
                
                print(f"✅ Success! Created {output_path} with {len(manifest)} redactions.")
            except Exception as e:
                print(f"❌ Error processing {meta_file}: {e}")