import json
import cv2
import numpy as np

def interpolate_coordinates(start_coords, end_coords, current_frame, start_frame, end_frame):
    """Calculates the estimated bounding box between two AI-identified keyframes."""
    fraction = (current_frame - start_frame) / (end_frame - start_frame)
    
    x = int(start_coords['x'] + (end_coords['x'] - start_coords['x']) * fraction)
    y = int(start_coords['y'] + (end_coords['y'] - start_coords['y']) * fraction)
    w = int(start_coords['w'] + (end_coords['w'] - start_coords['w']) * fraction)
    h = int(start_coords['h'] + (end_coords['h'] - start_coords['h']) * fraction)
    
    return {'x': x, 'y': y, 'w': w, 'h': h}

def generate_full_tracking(video_path, redaction_manifest):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    full_tracked_data = {}

    for entity_id, instances in redaction_manifest.items():
        # instances is a list of {frame_number, x, y, w, h}
        instances = sorted(instances, key=lambda x: x['frame_number'])
        
        for i in range(len(instances) - 1):
            start_node = instances[i]
            end_node = instances[i+1]
            
            for f in range(start_node['frame_number'], end_node['frame_number']):
                coords = interpolate_coordinates(
                    start_node, end_node, f, 
                    start_node['frame_number'], end_node['frame_number']
                )
                
                if f not in full_tracked_data:
                    full_tracked_data[f] = []
                
                full_tracked_data[f].append({
                    'entity_id': entity_id,
                    'bbox': coords
                })
    
    return full_tracked_data

# Save this to tracked_manifest.json for the renderer to consume