#!/usr/bin/env python3
"""Query all available render settings from DaVinci Resolve."""

import sys
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

def query_render_settings():
	"""Query and display all available render settings."""
	_add_resolve_module_path()
	try:
		import DaVinciResolveScript as dvr
	except Exception as e:
		print(f"ERROR: Failed to import DaVinciResolveScript: {e}")
		return False
	
	try:
		resolve = dvr.scriptapp("Resolve")
		pm = resolve.GetProjectManager()
		project = pm.GetCurrentProject()
		
		if not project:
			print("ERROR: No current project")
			return False
		
		print("=" * 80)
		print("AVAILABLE RENDER SETTINGS")
		print("=" * 80)
		
		# Get current render settings
		current_settings = project.GetRenderSettings()
		print("\nCurrent Render Settings:")
		print("-" * 80)
		if isinstance(current_settings, dict):
			for key in sorted(current_settings.keys()):
				value = current_settings[key]
				print(f"  {key:40s}: {value}")
		else:
			print(f"Type: {type(current_settings)}")
			print(f"Value: {current_settings}")
		
		print("\n" + "=" * 80)
		print("AVAILABLE FORMAT NAMES")
		print("=" * 80)
		
		# Get available formats
		try:
			formats = project.GetRenderFormatList()
			if formats:
				print("Formats:", formats)
			else:
				print("(no formats returned)")
		except Exception as e:
			print(f"Error getting format list: {e}")
		
		print("\n" + "=" * 80)
		print("AVAILABLE CODECS FOR EACH FORMAT")
		print("=" * 80)
		
		# Try to get codec list for mp4
		try:
			codecs = project.GetRenderCodecList("mp4")
			if codecs:
				print("MP4 Codecs:", codecs)
			else:
				print("(no codecs returned for mp4)")
		except Exception as e:
			print(f"Error getting codec list: {e}")
		
		print("\n" + "=" * 80)
		print("RENDER PRESET LIST")
		print("=" * 80)
		
		# Get presets
		try:
			presets = project.GetRenderPresetList()
			if presets:
				print("Presets:", presets)
			else:
				print("(no presets returned)")
		except Exception as e:
			print(f"Error getting preset list: {e}")
		
		print("\n" + "=" * 80)
		print("TESTING SetRenderSettings WITH DIFFERENT PARAMETERS")
		print("=" * 80)
		
		# Try setting with minimal params
		print("\nAttempting to set with minimal parameters...")
		result = project.SetRenderSettings({
			"TargetFile": "/tmp/test.mp4",
			"FormatName": "mp4",
			"CodecName": "hevc_nvenc",
		})
		print(f"Result: {result}")
		
		# Read back what was set
		settings = project.GetRenderSettings()
		print("\nSettings after SetRenderSettings:")
		if isinstance(settings, dict):
			for key in sorted(settings.keys()):
				value = settings[key]
				print(f"  {key:40s}: {value}")
		
		return True
	
	except Exception as e:
		print(f"ERROR: {e}")
		import traceback
		traceback.print_exc()
		return False

if __name__ == "__main__":
	query_render_settings()
