import json

def generate_ui_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    ui_summary = {}

    for video_name, frames in data.items():
        entities = {} # Stores first appearance and metadata
        
        for entry in frames:
            frame_name = entry['frame']
            for target in entry['targets']:
                uid = target['id']
                label = target['label']
                
                if uid not in entities:
                    # First time we see this person/doc
                    entities[uid] = {
                        "category": label,
                        "first_seen": frame_name,
                        "total_frames": 1,
                        "sample_bbox": target['bbox']
                    }
                else:
                    entities[uid]["total_frames"] += 1

        ui_summary[video_name] = entities

    # Save for your UI to read
    with open('ui_manifest.json', 'w') as f:
        json.dump(ui_summary, f, indent=4)
    
    return ui_summary

# --- EXAMPLE OF HOW YOUR UI USES THIS ---
if __name__ == "__main__":
    summary = generate_ui_data('tracked_manifest.json')
    
    for video, items in summary.items():
        print(f"\n📺 Video: {video}")
        print("-" * 30)
        for uid, info in items.items():
            print(f"[ ] Redact {uid} ({info['category']}) - Seen in {info['total_frames']} frames")