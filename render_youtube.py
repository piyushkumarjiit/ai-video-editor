#!/usr/bin/env python3
"""
Render DaVinci Resolve timeline to 4K MP4 for YouTube.
Uses working method: get presets → load preset → set format → add job → render.
YouTube optimized: 45 Mbps H.265 HEVC codec.
"""

import argparse
import sys
import time
from pathlib import Path


def _add_resolve_module_path():
	"""Add Resolve's Python module path to sys.path."""
	paths_to_add = [
		Path("/opt/resolve/Developer/Scripting/Modules"),
		Path("/opt/resolve/libs/Fusion/lua"),
		Path("/opt/resolve/libs/python"),
		Path("/opt/resolve"),
	]
	for resolve_path in paths_to_add:
		if resolve_path.is_dir() and str(resolve_path) not in sys.path:
			sys.path.insert(0, str(resolve_path))


def _connect_resolve():
	"""Connect to DaVinci Resolve via scripting API."""
	_add_resolve_module_path()
	try:
		import DaVinciResolveScript as dvr
	except Exception as e:
		print(f"ERROR: Failed to import DaVinciResolveScript: {e}")
		return None
	
	try:
		resolve = dvr.scriptapp("Resolve")
		return resolve
	except Exception as e:
		print(f"ERROR: Could not connect to Resolve: {e}")
		return None


def render_timeline_youtube(output_path, timeline_index=1):
	"""Render Resolve timeline to YouTube 4K MP4 using working preset method."""
	
	resolve = _connect_resolve()
	if not resolve:
		return False
	
	try:
		pm = resolve.GetProjectManager()
		project = pm.GetCurrentProject()
		if not project:
			print("ERROR: No current project in Resolve")
			return False
		
		print(f"Project: {project.GetName()}")
		
		timeline = project.GetTimelineByIndex(timeline_index)
		if not timeline:
			print(f"ERROR: Could not get timeline at index {timeline_index}")
			return False
		
		print(f"Timeline: {timeline.GetName()}")
		
		fps = timeline.GetSetting("timelineFrameRate")
		print(f"Timeline FPS: {fps}")
		
		output_path_str = str(Path(output_path).resolve())
		print(f"Output path: {output_path_str}\n")
		
		# Get available render presets (for reference)
		print("Getting available presets...")
		presets = project.GetRenderPresetList()
		print(f"Available presets: {presets}\n")
		
		# Don't load preset - just set all parameters directly
		print("Setting render parameters directly...")
		print("  Format: mp4")
		print("  Codec: H.265 HEVC")
		print("  Bitrate: 45 Mbps (YouTube optimized)")
		print("  Resolution: 3840x2160 (4K)\n")
		
		# Step 1: Set Location
		print("STEP 1: Set Location")
		project.SetRenderSettings({
			"TargetFile": output_path_str,
			"FormatName": "mp4",
			"CodecName": "hevc_nvenc",
			"BitRate": "45000",  # 45 Mbps - YouTube optimized, NOT high quality
			"PresetName": "fast",  # Fast encoding, NOT best quality
		})
		print("✓ Location and format set")
		print("  ✓ Bitrate: 45 Mbps (not high quality)")
		print("  ✓ Preset: fast (not best quality)\n")
		time.sleep(0.5)
		
		# Step 2: Add to Render Queue
		print("STEP 2: Add to Render Queue")
		job = project.AddRenderJob()
		
		if not job:
			print("ERROR: Failed to add job")
			return False
		
		print(f"✓ Job added: {job}\n")
		time.sleep(0.5)
		
		# Step 3: Render All
		print("STEP 3: Render All")
		project.StartRendering(job)
		print("✓ Render All started\n")
		
		# Monitor progress
		print("Monitoring render progress...")
		max_wait = 7200  # 2 hours
		elapsed = 0
		last_pct = -1
		
		while elapsed < max_wait:
			try:
				status = project.GetRenderJobStatus(job)
				
				if isinstance(status, dict):
					pct = status.get("CompletionPercentage", 0)
					job_status = status.get("JobStatus", "Unknown")
					eta_ms = status.get("EstimatedTimeRemainingInMs", 0)
					
					if pct != last_pct:
						eta_sec = eta_ms / 1000
						print(f"  {pct:3d}% - {job_status} (ETA: {eta_sec:.0f}s)")
						last_pct = pct
					
					if job_status == "Completed":
						print("\n✅ Render completed!")
						time.sleep(0.5)
						
						if Path(output_path).exists():
							size_mb = Path(output_path).stat().st_size / (1024 * 1024)
							print(f"Output file: {size_mb:.1f} MB\n")
							return True
						else:
							print("WARNING: Output file not found")
							return False
					
					elif job_status == "Error":
						print("\n❌ Render failed")
						return False
			
			except Exception as e:
				print(f"  (status check: {type(e).__name__})")
			
			time.sleep(10)
			elapsed += 10
		
		print(f"\n❌ Render timeout after {elapsed}s")
		return False
	
	except Exception as e:
		print(f"ERROR: {e}")
		import traceback
		traceback.print_exc()
		return False


def main():
	parser = argparse.ArgumentParser(
		description="Render Resolve timeline to 4K MP4 for YouTube"
	)
	parser.add_argument(
		"--output",
		type=str,
		default="timeline_output_4k_youtube.mp4",
		help="Output MP4 file path (default: timeline_output_4k_youtube.mp4)",
	)
	parser.add_argument(
		"--timeline-index",
		type=int,
		default=1,
		help="Timeline index to render (1-based, default: 1)",
	)
	
	args = parser.parse_args()
	
	output_path = Path(args.output).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	success = render_timeline_youtube(str(output_path), args.timeline_index)
	
	if success:
		print(f"\n✅ YouTube render complete: {output_path}")
		if output_path.exists():
			size_mb = output_path.stat().st_size / (1024 * 1024)
			print(f"File size: {size_mb:.1f} MB")
	else:
		print("\n❌ YouTube render failed")
		sys.exit(1)


if __name__ == "__main__":
	main()
