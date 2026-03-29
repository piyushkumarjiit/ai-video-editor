import os
from scenedetect import open_video, SceneManager, ContentDetector, save_images

# Configuration
VIDEO_DIR = 'samples/sanitized'
BASE_OUTPUT_DIR = 'keyframes'

def analyze_all_samples():
    # Loop through every mp4 in your samples folder
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        # Create a unique folder for this specific video
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(BASE_OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"\n🎬 Analyzing: {video_file}")

        # 1. Initialize Scene Manager
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))

        # 2. Perform Scene Detection
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        print(f"✅ Found {len(scene_list)} scenes. Saving to {video_output_dir}...")

        # 3. Save keyframes into the specific subfolder
        save_images(
            scene_list=scene_list,
            video=video,
            num_images=1,
            image_name_template='$SCENE_NUMBER',
            output_dir=video_output_dir  # Points to the specific subfolder
        )

if __name__ == "__main__":
    analyze_all_samples()