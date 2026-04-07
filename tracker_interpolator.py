"""
FILE: tracker_interpolator.py
ROLE: Path Smoothing & Frame Gap Filling
-------------------------------------------------------------------------
DESCRIPTION:
Calculates the linear trajectory of bounding boxes between AI-processed 
keyframes. This ensures the final redaction blur doesn't 'jump' but 
moves smoothly with the target's motion.
-------------------------------------------------------------------------
"""

import json
import cv2
import numpy as np

def interpolate_coordinates(start_coords, end_coords, current_frame, start_frame, end_frame):
    """Calculates the estimated bounding box between two AI-identified keyframes."""
    # Safety check to avoid division by zero
    if end_frame == start_frame:
        return start_coords
        
    fraction = (current_frame - start_frame) / (end_frame - start_frame)
    
    x = int(start_coords['x'] + (end_coords['x'] - start_coords['x']) * fraction)
    y = int(start_coords['y'] + (end_coords['y'] - start_coords['y']) * fraction)
    w = int(start_coords['w'] + (end_coords['w'] - start_coords['w']) * fraction)
    h = int(start_coords['h'] + (end_coords['h'] - start_coords['h']) * fraction)
    
    return {'x': x, 'y': y, 'w': w, 'h': h}

def generate_full_tracking(video_path, redaction_manifest):
    cap = cv2.VideoCapture(video_path)
    # Get actual dimensions to scale from the AI's 0-1000 range
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    full_tracked_data = {}

    for entity_id, instances in redaction_manifest.items():
        # Sort by frame to ensure linear progression
        instances = sorted(instances, key=lambda x: x.get('frame_number', 0))
        
        for i in range(len(instances) - 1):
            start_node = instances[i]
            end_node = instances[i+1]
            
            # Fill the gap between two keyframes
            for f in range(start_node['frame_number'], end_node['frame_number'] + 1):
                coords = interpolate_coordinates(
                    start_node, end_node, f, 
                    start_node['frame_number'], end_node['frame_number']
                )
                
                # SCALE MAPPING: Convert 0-1000 to actual Pixels
                scaled_coords = {
                    'x': int(coords['x'] * width / 1000),
                    'y': int(coords['y'] * height / 1000),
                    'w': int(coords['w'] * width / 1000),
                    'h': int(coords['h'] * height / 1000)
                }
                
                if f not in full_tracked_data:
                    full_tracked_data[f] = []
                
                # Check for duplicates to prevent double-blurring same entity
                if not any(d['entity_id'] == entity_id for d in full_tracked_data[f]):
                    full_tracked_data[f].append({
                        'entity_id': entity_id,
                        'bbox': scaled_coords
                    })
    
    return full_tracked_data