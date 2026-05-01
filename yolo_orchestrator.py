"""
FILE: yolo_orchestrator.py
ROLE: Multi-Process Execution & Resource Management
-------------------------------------------------------------------------
DESCRIPTION:
Coordinates parallel execution of YOLOv8 tracking workers. 
Each worker writes JSON progress lines to
stdout; this process reads them, filters all other noise, and drives tqdm
bars directly — giving us clean, live progress without a TTY pipe problem.

Protocol (worker → orchestrator, one JSON object per line):
  {"type": "init",     "total": <int>}   # sent once before processing
  {"type": "progress", "frame": <int>}   # sent every frame
  {"type": "done"}                        # sent when finished

OUTPUT STRUCTURE:
tracking_results/
└── [video_name]/
    ├── [video_name]_tracking.json
    └── previews/
        ├── latest_heartbeat.jpg
        └── latest_detection.jpg
-------------------------------------------------------------------------
"""

import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import tqdm  # orchestrator owns ALL tqdm bars

# ─── CONFIGURATION ────────────────────────────────────────────────────────
VIDEO_SOURCE_DIR  = "samples/sanitized"
OUTPUT_BASE_DIR   = "tracking_results"
WORKER_COUNT      = 3
TRACKER_SCRIPT    = "yolo_tracker.py"
DEFAULT_CONF      = 0.45
THUMBNAIL_INTERVAL = 30


def run_worker(task_info):
    """
    Runs in a multiprocessing.Pool worker.
    Launches the tracker as a subprocess, reads its JSON stdout,
    and sends (worker_id, msg_dict) tuples back via a shared queue.
    """
    video_path, worker_id, queue = task_info

    #time.sleep(worker_id * 10)   # staggered GPU warm-up

    queue.put((worker_id, {"type": "starting", "name": video_path.name}))

    cmd = [
        "python3", TRACKER_SCRIPT,
        "--source",             str(video_path),
        "--output_dir",         str(Path(OUTPUT_BASE_DIR) / video_path.stem),
        "--worker_id",          str(worker_id),
        "--conf",               str(DEFAULT_CONF),
        "--thumbnail_interval", str(THUMBNAIL_INTERVAL),
        "--show_thumbnails",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # C++/TRT stderr noise goes here
        text=True,
        bufsize=1,
    )

    for raw_line in proc.stdout:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = json.loads(raw_line)          # only forward valid JSON
            queue.put((worker_id, msg))
        except json.JSONDecodeError:
            pass                                # silently drop any non-JSON noise

    proc.wait()
    rc = proc.returncode
    queue.put((worker_id, {"type": "exited", "code": rc, "name": video_path.name}))


def orchestrate(tasks, queue):
    """
    Reads from the shared queue and manages one tqdm bar per worker slot.
    Runs in the main process so it has full TTY access.
    """
    # Pre-create one bar per worker slot (position=0,1,2,...)
    bars = {
        wid: tqdm.tqdm(
            total=0,
            desc=f"  Worker {wid}  (waiting)",
            position=wid,
            leave=True,
            unit="f",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ascii=" ▂▃▄▅▆▇█",
            dynamic_ncols=True,
        )
        for wid in range(WORKER_COUNT)
    }

    # Print header above the bars (below position 0)
    tqdm.tqdm.write(f"🚀 Processing {len(tasks)} videos with {WORKER_COUNT} workers...\n")

    active = len(tasks)

    while active > 0:
        try:
            worker_id, msg = queue.get(timeout=30)
        except Exception:
            continue    # timeout — workers still alive, keep waiting

        bar  = bars[worker_id]
        kind = msg.get("type")

        if kind == "starting":
            name = msg["name"][:28]
            bar.set_description(f"⏳ {name}")
            bar.reset(total=0)

        elif kind == "init":
            bar.reset(total=msg["total"])
            bar.set_description(f"🔍 {bar.desc[2:]}")   # swap ⏳ → 🔍

        elif kind == "progress":
            bar.n = msg["frame"]
            bar.refresh()

        elif kind == "done":
            bar.n = bar.total
            bar.refresh()

        elif kind == "exited":
            name = msg["name"]
            code = msg["code"]
            if code == 0:
                bar.set_description(f"✅ {name[:28]}")
            else:
                bar.set_description(f"❌ {name[:28]} (exit {code})")
            bar.refresh()
            active -= 1

    for bar in bars.values():
        bar.close()

    tqdm.tqdm.write("\n✅ ALL VIDEOS PROCESSED.")


def main():
    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(VIDEO_SOURCE_DIR).glob("*.mp4"))
    if not videos:
        print(f"No videos found in {VIDEO_SOURCE_DIR}")
        return

    # Use 'spawn' so child processes don't inherit the parent's tqdm state
    ctx   = multiprocessing.get_context("spawn")
    queue = ctx.Manager().Queue()

    tasks = [(vid, i % WORKER_COUNT, queue) for i, vid in enumerate(videos)]

    # Launch pool workers (they feed the queue) …
    pool = ctx.Pool(processes=WORKER_COUNT)
    pool.map_async(run_worker, tasks)
    pool.close()

    # … while the main process drains the queue and draws bars
    orchestrate(tasks, queue)

    pool.join()


if __name__ == "__main__":
    main()