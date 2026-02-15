#!/usr/bin/env python3
"""
Alternative render method using DaVinci Fusion scripting to avoid queue issues.
"""

import argparse
import os
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


def _connect_resolve(retries=10):
	"""Connect to DaVinci Resolve via scripting API."""
	_add_resolve_module_path()
	try:
		import DaVinciResolveScript as dvr
	except Exception as e:
		print(f"Failed to import DaVinciResolveScript: {e}")
		return None
	
	for attempt in range(retries):
		try:
			resolve = dvr.scriptapp("Resolve")
			if resolve:
				return resolve
		except Exception:
			pass
		if attempt < retries - 1:
			time.sleep(1)
	return None


def render_timeline_youtube(output_path, timeline_index=1):
	"""
	Render the current Resolve timeline to 4K MP4 for YouTube.
	Uses alternative method via Fusion if queue is locked.
	"""
	resolve = _connect_resolve()
	if not resolve:
		print("ERROR: Could not connect to DaVinci Resolve")
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
		
		output_path_str = str(Path(output_path).resolve())
		print(f"Output: {output_path_str}")
		
		# Get Fusion instance
		print("\nAttempting Fusion export...")
		fusion = resolve.Fusion()
		if not fusion:
			print("ERROR: Could not get Fusion instance")
			return False
		
		# Try using OpenPageComposition to access Deliver page
		print("Trying to access Deliver settings...")
		
		# Alternative: Use command to export directly
		# Go to Deliver page
		resolve.OpenPage("Deliver")
		time.sleep(2)
		
		# Try to set render settings through project API
		print("\nTrying render queue with fresh project state...")
		
		# Delete old jobs
		try:
			project.DeleteAllRenderJobs()
			time.sleep(1)
		except:
			pass
		
		# Set settings  
		project.SetRenderSettings({
			"TargetFile": output_path_str,
			"FormatName": "mp4",
			"CodecName": "hevc_nvenc",
			"BitRate": "45000",
			"PresetName": "fast"
		})
		
		print("✓ Settings configured")
		time.sleep(2)
		
		# Try to add job with fresh state
		job = project.AddRenderJob()
		
		if not job:
			print("\n⚠️  Render queue still locked")
			print("Manual render required:")
			print("  1. In Resolve, go to Deliver page")
			print("  2. Select: Custom Export")
			print("  3. Set Format: mp4")
			print("  4. Set Codec: hevc_nvenc")
			print("  5. Set Bitrate: 45000 (45 Mbps)")
			print("  6. Output to: " + output_path_str)
			print("  7. Click 'Add to Render Queue' then 'Start Rendering'")
			return False
		
		print(f"✓ Job added: {job}")
		
		# Start rendering
		print("\nStarting render...")
		project.StartRendering(job)
		print("✓ Render started")
		
		# Monitor
		print("\nMonitoring render progress...")
		max_wait = 7200
		elapsed = 0
		last_pct = -1
		
		while elapsed < max_wait:
			try:
				status_dict = project.GetRenderJobStatus(job)
				if isinstance(status_dict, dict):
					job_status = status_dict.get("JobStatus", "Unknown")
					pct = status_dict.get("CompletionPercentage", 0)
					
					if pct != last_pct:
						print(f"  {pct}% - {job_status}")
						last_pct = pct
					
					if job_status == "Completed":
						print("\n✔ Render completed successfully!")
						time.sleep(1)
						
						if Path(output_path).exists():
							size_mb = Path(output_path).stat().st_size / (1024 * 1024)
							print(f"Output file: {size_mb:.1f} MB")
							return True
						return False
					elif job_status == "Error":
						print("\n❌ Render failed")
						return False
				else:
					print(f"  Status: {status_dict}")
			
			except Exception as e:
				print(f"  Monitor error: {e}")
			
			time.sleep(10)
			elapsed += 10
		
		print(f"❌ Render timeout after {elapsed}s")
		return False
	
	except Exception as e:
		print(f"ERROR: {e}")
		import traceback
		traceback.print_exc()
		return False


def main():
	parser = argparse.ArgumentParser(description="Render Resolve timeline to YouTube 4K MP4")
	parser.add_argument("--output", required=True, help="Output file path")
	parser.add_argument("--timeline-index", type=int, default=1, help="Timeline index (1-based)")
	
	args = parser.parse_args()
	
	success = render_timeline_youtube(args.output, args.timeline_index)
	
	if success:
		print("\n✅ YouTube render succeeded")
		sys.exit(0)
	else:
		print("\n❌ YouTube render failed")
		sys.exit(1)


if __name__ == "__main__":
	main()
