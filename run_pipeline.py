#!/usr/bin/env python3
"""
End-to-end workflow: analyze -> extract clips -> export Resolve timeline.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_stage(label, cmd, cwd):
	print(f"\n▶ {label}")
	print("-" * 72)
	print(" ".join(cmd))
	print("-" * 72)
	subprocess.run(cmd, cwd=cwd, check=True)
	print(f"✔ {label} completed")


def list_videos(input_dir):
	return sorted(
		[p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".mkv"}],
		key=lambda p: p.name.lower(),
	)


def analysis_output_path(video_path, output_dir):
	return output_dir / f"scene_analysis_{video_path.stem}.json"


def analysis_exists(video_path, output_dir):
	primary = analysis_output_path(video_path, output_dir)
	if primary.exists():
		return True
	pattern = f"scene_analysis_{video_path.stem}*.json"
	return any(output_dir.glob(pattern))


def expected_clip_exists(clips_dir, video_stem, scene_num, classification):
	pattern = f"{video_stem}_scene_{scene_num:02d}_{classification}_*x.mkv"
	return any((clips_dir / video_stem).glob(pattern))


def clips_complete(analysis_file, clips_dir):
	import json
	with open(analysis_file, "r") as f:
		analysis = json.load(f)

	video_name = analysis.get("video")
	if not video_name:
		return False

	video_stem = Path(video_name).stem
	output_dir = clips_dir / video_stem
	if not output_dir.exists():
		return False

	for scene in analysis.get("scenes", []):
		classification = scene.get("classification", "unknown")
		if classification == "skip":
			continue
		if not expected_clip_exists(clips_dir, video_stem, scene["scene_num"], classification):
			return False

	return True


def load_project_config(config_path):
	if not config_path:
		return {}
	path = Path(config_path)
	if not path.exists():
		return {}
	try:
		with open(path, "r") as f:
			data = json.load(f)
		if isinstance(data, dict):
			return data
	except Exception:
		return {}
	return {}


def main():
	parser = argparse.ArgumentParser(description="End-to-end AI video pipeline")
	parser.add_argument("--config", default="project_config.json", help="Project config JSON file")
	parser.add_argument("--input-dir", default=None, help="Folder with input videos")
	parser.add_argument("--video", default=None, help="Single video file to analyze")
	parser.add_argument("--output-dir", default=None, help="Folder for analysis JSON outputs")
	parser.add_argument("--clips-dir", default=None, help="Base output directory for clips")
	parser.add_argument("--timeline", default=None, help="Output FCPXML file")
	parser.add_argument("--sample-interval", type=int, default=None, help="Frame sample interval (seconds)")
	parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=None, help="Deduplicate similar scenes across videos")
	parser.add_argument("--hash-threshold", type=int, default=None, help="Hamming distance threshold for dedupe")
	parser.add_argument("--use-rendered", action=argparse.BooleanOptionalAction, default=None, help="Use pre-rendered clips when available")
	parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
	parser.add_argument("--skip-extract", action="store_true", help="Skip clip extraction stage")
	parser.add_argument("--skip-export", action="store_true", help="Skip timeline export stage")
	parser.add_argument("--force-analysis", action="store_true", help="Re-run analysis even if JSON exists")
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent
	python = sys.executable

	config = load_project_config(args.config)
	paths_cfg = config.get("paths", {})
	analysis_cfg = config.get("analysis", {})
	pipeline_cfg = config.get("pipeline", {})

	input_dir = Path(args.input_dir or paths_cfg.get("input_dir") or ".").resolve()
	output_dir = Path(args.output_dir or paths_cfg.get("output_dir") or ".").resolve()
	clips_dir = Path(args.clips_dir or paths_cfg.get("clips_dir") or "ai_clips").resolve()
	timeline_path = Path(args.timeline or paths_cfg.get("timeline") or "timeline_davinci_resolve.fcpxml").resolve()

	video_arg = args.video or paths_cfg.get("video")
	if video_arg:
		video_path = Path(video_arg).resolve()
		video_dir = video_path.parent
	else:
		video_path = None
		video_dir = input_dir

	sample_interval = args.sample_interval if args.sample_interval is not None else analysis_cfg.get("sample_interval", 2)
	dedupe = args.dedupe if args.dedupe is not None else pipeline_cfg.get("dedupe", False)
	hash_threshold = args.hash_threshold if args.hash_threshold is not None else pipeline_cfg.get("hash_threshold", 6)
	use_rendered = args.use_rendered if args.use_rendered is not None else pipeline_cfg.get("use_rendered", False)
	skip_analysis = args.skip_analysis or pipeline_cfg.get("skip_analysis", False)
	skip_extract = args.skip_extract or pipeline_cfg.get("skip_extract", False)
	skip_export = args.skip_export or pipeline_cfg.get("skip_export", False)

	print("\n=== AI Video Pipeline ===")
	print(f"Input dir:   {input_dir}")
	print(f"Output dir:  {output_dir}")
	print(f"Clips dir:   {clips_dir}")
	print(f"Timeline:    {timeline_path}")
	print(f"Sample step: {sample_interval}s")
	print(f"Dedupe:      {dedupe} (threshold {hash_threshold})")
	print(f"Rendered:    {use_rendered}")

	try:
		if not skip_analysis:
			videos_to_analyze = []
			if video_path:
				if args.force_analysis or not analysis_exists(video_path, output_dir):
					videos_to_analyze = [video_path]
			else:
				for vid in list_videos(input_dir):
					if args.force_analysis or not analysis_exists(vid, output_dir):
						videos_to_analyze.append(vid)

			if not videos_to_analyze:
				print("\n▶ [1/3] Analyze videos (skipped - JSON exists)")
			else:
				# Run batch analysis with analyze_advanced5.py
				cmd = [
					python,
					str(base_dir / "analyze_advanced5.py"),
					"--sample-interval",
					str(sample_interval),
					"--skip-duplicate-captions",
					"--max-scene-length",
					"40",
				]
				if video_path:
					cmd.extend(["--video", str(video_path)])
				run_stage("[1/3] Analyze videos with Qwen2.5-VL-7B", cmd, base_dir)
		else:
			print("\n▶ [1/3] Analyze videos (skipped)")

		if not skip_extract:
			# Run batch extraction with extract_scenes.py
			cmd = [
				python,
				str(base_dir / "extract_scenes.py"),
				"--analysis-dir",
				str(output_dir),
				"--output-dir",
				str(clips_dir),
			]
			if video_dir != input_dir:
				cmd.extend(["--video-dir", str(video_dir)])
			run_stage("[2/3] Extract scenes and showcase moments", cmd, base_dir)
		else:
			print("\n▶ [2/3] Extract scenes (skipped)")

		if not skip_export:
			cmd = [
				python,
				str(base_dir / "export_resolve.py"),
				"--config",
				str(Path(args.config).resolve()),
				"--analysis",
				str(output_dir),
				"--video-dir",
				str(video_dir),
				"--clips-dir",
				str(clips_dir),
				"--output",
				str(timeline_path),
			]
			if use_rendered:
				cmd.append("--use-rendered")
			if dedupe:
				cmd.append("--dedupe")
				cmd.extend(["--hash-threshold", str(hash_threshold)])
			run_stage("[3/3] Export Resolve timeline", cmd, base_dir)
		else:
			print("\n▶ [3/3] Export Resolve timeline (skipped)")

		print("\n✅ Pipeline complete")
		print("Import all clips to Resolve Media Pool before importing the XML.")

	except subprocess.CalledProcessError as exc:
		print(f"\n❌ Pipeline failed at: {exc.cmd}")
		sys.exit(exc.returncode)


if __name__ == "__main__":
	main()
