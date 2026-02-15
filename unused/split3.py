import csv
import subprocess
import os

INPUT = "IMG_3839.MOV"
CSV = "scenes.csv/IMG_3839-Scenes.csv"
OUTDIR = "clips"

os.makedirs(OUTDIR, exist_ok=True)

def clean_float(x):
    return float(x.strip().replace("\ufeff", ""))

with open(CSV, newline="") as f:
    reader = csv.reader(f)

    for row in reader:
        if not row:
            continue

        # Skip non-data rows
        if not row[0].strip().isdigit():
            continue

        scene = int(row[0])
        start = clean_float(row[3])
        end = clean_float(row[6])
        duration = end - start

        if duration < 3:
            continue   # drop micro scenes

        out = f"{OUTDIR}/scene_{scene:03d}.mkv"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", INPUT,
            "-t", str(duration),
            "-map", "0:v",  # video stream
            "-map", "0:a",  # audio stream
            "-c", "copy",
            out
        ]

        print(f"→ scene {scene}: {start:.3f}s → {end:.3f}s")
        subprocess.run(cmd, check=True)

print("\n✅ Scene splitting complete — 10-bit preserved.")
