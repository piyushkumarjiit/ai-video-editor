import json
import math

def calculate_distance(box1, box2):
    # Calculate center points (ymin, xmin, ymax, xmax)
    c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def track_entities(input_file, output_file, dist_threshold=150):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    final_data = {}

    for video_name, frames in data.items():
        active_entities = {} # { 'global_id': last_known_bbox }
        next_id_counter = 1
        tracked_frames = []

        for frame_entry in frames:
            current_targets = frame_entry['targets']
            new_frame_targets = []
            
            used_global_ids = set()

            for target in current_targets:
                current_bbox = target['bbox']
                current_label = target['label'].split('_')[0] # get 'face' or 'doc'
                
                matched_id = None
                min_dist = float('inf')

                # Try to match with existing active entities
                for gid, last_bbox in active_entities.items():
                    if gid in used_global_ids: continue
                    
                    dist = calculate_distance(current_bbox, last_bbox)
                    if dist < dist_threshold and dist < min_dist:
                        min_dist = dist
                        matched_id = gid

                if matched_id:
                    target['id'] = matched_id
                    active_entities[matched_id] = current_bbox
                    used_global_ids.add(matched_id)
                else:
                    # Assign a new Global ID
                    new_id = f"{current_label}_{next_id_counter}"
                    target['id'] = new_id
                    active_entities[new_id] = current_bbox
                    next_id_counter += 1
                    used_global_ids.add(new_id)

                new_frame_targets.append(target)
            
            tracked_frames.append({
                "frame": frame_entry['frame'],
                "targets": new_frame_targets
            })

        final_data[video_name] = tracked_frames

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"🚀 Grouping complete! Check {output_file}")

if __name__ == "__main__":
    track_entities('redaction_manifest.json', 'tracked_manifest.json')