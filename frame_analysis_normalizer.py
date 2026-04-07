"""
FILE: frame_analysis_normalizer.py
ROLE: VLM Output Sanitization & Standardization
-------------------------------------------------------------------------
DESCRIPTION:
A 'Data Janitor' script that converts unpredictable VLM JSON outputs into 
a strict schema. It handles common AI errors like stringified JSON 
nesting and varied key naming (bbox vs bbox_2d).

INPUT: video_analysis.json (Raw AI output).
OUTPUT: redaction_manifest.json (Clean, standardized schema).
-------------------------------------------------------------------------
"""

import json
import re

def normalize_detections(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    clean_manifest = {}

    for video_name, frames in data.items():
        clean_manifest[video_name] = []
        
        for entry in frames:
            frame_name = entry['frame']
            raw_detections = entry['detections']
            normalized = []

            # 1. Handle Stringified JSON (Frame 001 case)
            if isinstance(raw_detections, str):
                try:
                    # Fix single quotes to double quotes for valid JSON
                    json_ready = raw_detections.replace("'", '"')
                    raw_detections = json.loads(json_ready)
                except:
                    raw_detections = []

            # 2. Handle Nested Dicts (Frame 003 case)
            if isinstance(raw_detections, dict):
                raw_detections = raw_detections.get('result', raw_detections.get('detections', []))

            # 3. Standardize the List
            if isinstance(raw_detections, list):
                for item in raw_detections:
                    # Map 'bbox' or 'bbox_2d' to a single standard key
                    bbox = item.get('bbox_2d') or item.get('bbox')
                    label = item.get('label', 'unknown')
                    id_tag = item.get('id', label)

                    if bbox and len(bbox) == 4:
                        normalized.append({
                            "id": id_tag,
                            "label": label,
                            "bbox": bbox  # Always [ymin, xmin, ymax, xmax]
                        })

            clean_manifest[video_name].append({
                "frame": frame_name,
                "targets": normalized
            })

    with open(output_file, 'w') as f:
        json.dump(clean_manifest, f, indent=4)
    print(f"✅ Normalized manifest saved to {output_file}")

if __name__ == "__main__":
    normalize_detections('video_analysis.json', 'redaction_manifest.json')