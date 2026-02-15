import csv
import subprocess
from pathlib import Path

# Input video and CSV
video_file = "IMG_3839.MOV"
csv_file = "scenes.csv/IMG_3839-Scenes.csv"

# Output directory
output_dir = Path("clips")
output_dir.mkdir(exist_ok=True)

# Read the CSV and split video
with open(csv_file, newline="") as f:
    # Skip the first row (Timecode List)
    next(f)
    reader = csv.DictReader(f)
    for row in reader:
        scene_number = row["Scene Number"]
        start_time = row["Start Time (seconds)"]
        end_time = row["End Time (seconds)"]
        output_file = output_dir / f"scene_{scene_number.zfill(3)}.mp4"

        # FFmpeg command to preserve 10-bit HEVC using libx265
        cmd = [
            "ffmpeg",
            "-y",                   # overwrite output
            "-i", video_file,       # input file
            "-ss", start_time,      # start time
            "-to", end_time,        # end time
            "-c:v", "libx265",      # HEVC software encoder
            "-pix_fmt", "yuv420p10le", # preserve 10-bit
            "-crf", "20",           # quality (lower = better)
            "-preset", "fast",      # speed/efficiency
            "-c:a", "copy",         # copy audio
            str(output_file)
        ]

        print(f"→ {output_file}")
        subprocess.run(cmd, check=True)

print("All scenes processed successfully!")
