#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Optional


def _add_resolve_module_path() -> None:
    candidates = [
        "/opt/resolve/Developer/Scripting/Modules",
        "/opt/resolve/Developer/Scripting/Modules/Resolve",
        "/opt/resolve/Developer/Scripting/Modules/DaVinciResolveScript",
        os.path.expanduser("~/DaVinciResolve/Developer/Scripting/Modules"),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def _import_resolve():
    _add_resolve_module_path()
    try:
        import DaVinciResolveScript as dvr  # type: ignore
        return dvr
    except Exception as exc:  # noqa: BLE001
        print("Failed to import DaVinciResolveScript.")
        print("Make sure DaVinci Resolve is installed and its scripting modules are on sys.path.")
        print("Tried common paths under /opt/resolve/Developer/Scripting/Modules.")
        print(f"Error: {exc}")
        sys.exit(1)


def _collect_media_pool_clips(folder) -> List:
    clips = []
    try:
        clips.extend(folder.GetClipList() or [])
    except Exception:  # noqa: BLE001
        pass
    try:
        for sub in folder.GetSubFolderList() or []:
            clips.extend(_collect_media_pool_clips(sub))
    except Exception:  # noqa: BLE001
        pass
    return clips


def _build_lut_values(lut_path: str) -> List[str]:
    values: List[str] = []
    if lut_path:
        values.append(lut_path)
    lut_root = "/opt/resolve/LUT/"
    if lut_path.startswith(lut_root):
        rel = lut_path[len(lut_root) :]
        values.append(rel)
        rel_no_ext, _ = os.path.splitext(rel)
        if rel_no_ext:
            values.append(rel_no_ext)
    else:
        rel_no_ext, _ = os.path.splitext(lut_path)
        if rel_no_ext and rel_no_ext != lut_path:
            values.append(rel_no_ext)
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _set_clip_lut(clip, lut_path: str, property_key: Optional[str] = None) -> Optional[str]:
    keys = [property_key] if property_key else ["Input LUT", "3D Input LUT"]
    values = _build_lut_values(lut_path)
    for key in keys:
        for value in values:
            try:
                if clip.SetClipProperty(key, value):
                    return key
            except Exception:  # noqa: BLE001
                pass
    try:
        props = clip.GetClipProperty() or {}
        for key in props:
            if "LUT" in key:
                try:
                    if clip.SetClipProperty(key, lut_path):
                        return key
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    return None


def _set_timeline_item_lut(item, lut_path: str, property_key: Optional[str] = None) -> Optional[str]:
    keys = [property_key] if property_key else ["Input LUT", "3D Input LUT"]
    values = _build_lut_values(lut_path)
    for key in keys:
        for value in values:
            try:
                if item.SetProperty(key, value):
                    return key
            except Exception:  # noqa: BLE001
                pass
    try:
        props = item.GetProperty() or {}
        for key in props:
            if "LUT" in key:
                try:
                    if item.SetProperty(key, lut_path):
                        return key
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    try:
        clip = item.GetMediaPoolItem()
        if clip:
            return _set_clip_lut(clip, lut_path, property_key=property_key)
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a LUT to Resolve Media Pool and/or Timeline items.")
    parser.add_argument(
        "--config",
        default="./project_config.json",
        help="Path to project config JSON (used for default LUT)",
    )
    parser.add_argument(
        "--lut",
        default=None,
        help="Path to LUT .cube file",
    )
    parser.add_argument(
        "--mode",
        choices=["mediapool", "timeline", "both"],
        default="mediapool",
        help="Where to apply the LUT",
    )
    parser.add_argument(
        "--timeline-name",
        default=None,
        help="Target timeline name (optional; defaults to current timeline)",
    )
    parser.add_argument(
        "--clip-name",
        default=None,
        help="Media Pool clip name to target/inspect (optional)",
    )
    parser.add_argument(
        "--property-key",
        default=None,
        help="Force a specific LUT property key (e.g., 'Input LUT')",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print LUT-related properties for a sample clip/item and exit",
    )
    parser.add_argument(
        "--dump-props",
        action="store_true",
        help="Print all properties for a sample clip/item and exit",
    )
    parser.add_argument(
        "--search-props",
        default=None,
        help="Search all properties for a value containing this text and print the first match",
    )
    parser.add_argument(
        "--find-applied",
        action="store_true",
        help="Find a clip/item with a non-empty LUT value and print it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many clips/items to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without applying",
    )
    args = parser.parse_args()

    lut_path = args.lut
    if not lut_path and args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            lut_path = cfg.get("resolve", {}).get("input_lut")
        except Exception:  # noqa: BLE001
            pass
    if not lut_path:
        lut_path = (
            "/opt/resolve/LUT/Filmic_Pro_deLOG_LUT_Pack_May_2022/"
            "FiLMiC_Pro_deLOG_LUT_Pack_May_2022/10-bit_SDR_LUTs_iOS_only/"
            "FiLMiC_deLog_V3.cube"
        )
    if not os.path.exists(lut_path):
        print(f"LUT not found: {lut_path}")
        return 2

    dvr = _import_resolve()
    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        print("Resolve scripting app not available. Is DaVinci Resolve running?")
        return 3

    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        print("No current project open in Resolve.")
        return 4

    if args.mode in ("mediapool", "both"):
        media_pool = project.GetMediaPool()
        root = media_pool.GetRootFolder()
        clips = _collect_media_pool_clips(root)
        if args.clip_name:
            clips = [c for c in clips if (c.GetName() == args.clip_name)]
        if args.limit is not None:
            clips = clips[: max(args.limit, 0)]
        if args.inspect or args.find_applied or args.dump_props or args.search_props:
            if not clips:
                print("No Media Pool clips found to inspect.")
                return 6
            if args.search_props:
                needle = str(args.search_props)
                for clip in clips:
                    props = clip.GetClipProperty() or {}
                    for key, value in props.items():
                        if value is None:
                            continue
                        if needle in str(value):
                            print("Found matching property on Media Pool clip:")
                            print(f"  Clip: {clip.GetName()}")
                            print(f"  {key}: {value}")
                            return 0
                print("No Media Pool clip properties matched the search text.")
                return 0
            if args.find_applied:
                for clip in clips:
                    props = clip.GetClipProperty() or {}
                    lut_keys = [k for k in props.keys() if "LUT" in k]
                    for key in lut_keys:
                        value = props.get(key)
                        if value:
                            print("Found applied LUT on Media Pool clip:")
                            print(f"  Clip: {clip.GetName()}")
                            print(f"  {key}: {value}")
                            return 0
                print("No Media Pool clips with non-empty LUT values found.")
                return 0
            props = clips[0].GetClipProperty() or {}
            if args.dump_props:
                print("Clip properties:")
                for key in sorted(props.keys()):
                    print(f"  {key}: {props.get(key)}")
                return 0
            lut_keys = [k for k in props.keys() if "LUT" in k]
            print("LUT-related clip properties:")
            for key in lut_keys:
                print(f"  {key}: {props.get(key)}")
            if not lut_keys:
                print("No LUT-related keys found. Available keys:")
                for key in props.keys():
                    print(f"  {key}")
            return 0
        updated = 0
        for clip in clips:
            if args.dry_run:
                updated += 1
                continue
            key = _set_clip_lut(clip, lut_path, property_key=args.property_key)
            if key:
                updated += 1
        print(f"Media Pool clips processed: {len(clips)}, LUT applied: {updated}")

    if args.mode in ("timeline", "both"):
        timeline = project.GetCurrentTimeline()
        if args.timeline_name:
            timeline = project.GetTimelineByName(args.timeline_name)
        if not timeline:
            print("No timeline available.")
            return 5
        video_tracks = timeline.GetTrackCount("video")
        items = []
        for track_index in range(1, video_tracks + 1):
            items.extend(timeline.GetItemListInTrack("video", track_index) or [])
        if args.limit is not None:
            items = items[: max(args.limit, 0)]
        if args.inspect or args.find_applied or args.dump_props or args.search_props:
            if not items:
                print("No timeline items found to inspect.")
                return 7
            if args.search_props:
                needle = str(args.search_props)
                for item in items:
                    props = item.GetProperty() or {}
                    for key, value in props.items():
                        if value is None:
                            continue
                        if needle in str(value):
                            name = None
                            try:
                                clip = item.GetMediaPoolItem()
                                if clip:
                                    name = clip.GetName()
                            except Exception:  # noqa: BLE001
                                pass
                            print("Found matching property on timeline item:")
                            if name:
                                print(f"  Clip: {name}")
                            print(f"  {key}: {value}")
                            return 0
                print("No timeline item properties matched the search text.")
                return 0
            if args.find_applied:
                for item in items:
                    props = item.GetProperty() or {}
                    lut_keys = [k for k in props.keys() if "LUT" in k]
                    for key in lut_keys:
                        value = props.get(key)
                        if value:
                            name = None
                            try:
                                clip = item.GetMediaPoolItem()
                                if clip:
                                    name = clip.GetName()
                            except Exception:  # noqa: BLE001
                                pass
                            print("Found applied LUT on timeline item:")
                            if name:
                                print(f"  Clip: {name}")
                            print(f"  {key}: {value}")
                            return 0
                print("No timeline items with non-empty LUT values found.")
                return 0
            props = items[0].GetProperty() or {}
            if args.dump_props:
                print("Timeline item properties:")
                for key in sorted(props.keys()):
                    print(f"  {key}: {props.get(key)}")
                return 0
            lut_keys = [k for k in props.keys() if "LUT" in k]
            print("LUT-related timeline item properties:")
            for key in lut_keys:
                print(f"  {key}: {props.get(key)}")
            if not lut_keys:
                print("No LUT-related keys found. Available keys:")
                for key in props.keys():
                    print(f"  {key}")
            return 0
        updated = 0
        for item in items:
            if args.dry_run:
                updated += 1
                continue
            key = _set_timeline_item_lut(item, lut_path, property_key=args.property_key)
            if key:
                updated += 1
        print(f"Timeline items processed: {len(items)}, LUT applied: {updated}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
