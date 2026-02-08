#!/usr/bin/env python3
"""
Export DaVinci Resolve Timeline (FCP XML)
References original video with speed changes - no re-encoding needed
"""

import json
import os
import sys
import subprocess
import shutil
import random
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
from fractions import Fraction

DEFAULT_TIMELINE_WIDTH = 3840
DEFAULT_TIMELINE_HEIGHT = 2160
DEFAULT_WATERMARK_MARGIN = 80


def get_media_dimensions(media_path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x', media_path
        ], capture_output=True, text=True, check=True, timeout=2)
        value = (result.stdout or '').strip()
        if 'x' in value:
            width_str, height_str = value.split('x', 1)
            return int(width_str), int(height_str)
    except Exception:
        return None
    return None


def normalize_transparency(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1:
        value = value / 100.0
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return value


def compute_watermark_position(position, image_size, margin=DEFAULT_WATERMARK_MARGIN,
                               timeline_width=DEFAULT_TIMELINE_WIDTH,
                               timeline_height=DEFAULT_TIMELINE_HEIGHT):
    if isinstance(position, dict) and 'x' in position and 'y' in position:
        return f"{position['x']} {position['y']}"
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        return f"{position[0]} {position[1]}"

    image_width, image_height = image_size if image_size else (400, 400)
    half_w = timeline_width / 2
    half_h = timeline_height / 2
    offset_x = half_w - (image_width / 2) - margin
    offset_y = half_h - (image_height / 2) - margin

    pos = (position or '').lower().strip()
    if pos in ('top-left', 'topleft'):
        return f"{-offset_x} {offset_y}"
    if pos in ('top-right', 'topright'):
        return f"{offset_x} {offset_y}"
    if pos in ('bottom-left', 'bottomleft'):
        return f"{-offset_x} {-offset_y}"
    if pos in ('center', 'middle'):
        return "0 0"
    return f"{offset_x} {-offset_y}"

def get_video_duration_frac(video_path, fps=24):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0', video_path
    ], capture_output=True, text=True, check=True)
    total_frames = int(result.stdout.strip().rstrip(','))
    return Fraction(total_frames, fps)


def get_audio_info(audio_path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=duration,channels',
            '-show_entries', 'format=duration',
            '-of', 'json', audio_path
        ], capture_output=True, text=True, check=True, timeout=2)
        data = json.loads(result.stdout or '{}')
        streams = data.get('streams') or []
        channels = None
        duration = None
        if streams:
            stream = streams[0]
            channels = stream.get('channels')
            duration = stream.get('duration')
        if duration is None:
            duration = (data.get('format') or {}).get('duration')
        duration_sec = float(duration) if duration is not None else None
        if duration_sec and duration_sec > 0:
            duration_frac = Fraction(duration_sec).limit_denominator(10000)
        else:
            duration_frac = None
        if channels is None:
            channels = 2
        return duration_frac, int(channels)
    except Exception:
        return None, 2


def get_video_rotation_degrees(video_path):
    """Return rotation in degrees from video metadata (0, 90, 180, 270)."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-read_intervals', '0%+0.1',
            '-show_entries', 'stream_tags=rotate:side_data_list',
            '-of', 'json', video_path
        ], capture_output=True, text=True, check=True, timeout=2)
        data = json.loads(result.stdout or '{}')
        streams = data.get('streams') or []
        if not streams:
            return 0

        stream = streams[0]
        rotate_tag = None
        tags = stream.get('tags') or {}
        if 'rotate' in tags:
            rotate_tag = tags.get('rotate')

        side_data = stream.get('side_data_list') or []
        rotate_side = None
        for item in side_data:
            if 'rotation' in item:
                rotate_side = item.get('rotation')
                break

        rotate_value = rotate_tag if rotate_tag is not None else rotate_side
        if rotate_value is None:
            return 0

        rotation = int(round(float(rotate_value)))
        rotation = rotation % 360
        if rotation in (0, 90, 180, 270):
            return rotation
        return 0
    except Exception:
        return 0


def load_project_config(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def load_analyses(analysis_path, video_dir=None, video_file=None):
    analysis_path = Path(analysis_path)
    analyses = []
    
    if analysis_path.is_dir():
        analysis_files = sorted(analysis_path.glob('scene_analysis_*.json'), key=lambda p: p.name.lower())
    else:
        analysis_files = [analysis_path]
    
    for analysis_file in analysis_files:
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        video_name = data.get('video')
        if not video_name and video_file:
            video_name = os.path.basename(video_file)
        if not video_name:
            continue
        
        if video_file and not analysis_path.is_dir():
            video_path = Path(video_file)
        elif video_dir:
            video_path = Path(video_dir) / video_name
        else:
            video_path = analysis_file.parent / video_name
        
        analyses.append({
            'data': data,
            'analysis_file': analysis_file,
            'video_name': video_name,
            'video_path': video_path
        })
    
    analyses.sort(key=lambda x: x['video_name'].lower())
    return analyses


def hash_distance(a, b):
    if not a or not b:
        return None
    try:
        return bin(int(a, 16) ^ int(b, 16)).count('1')
    except ValueError:
        return None


def dedupe_clip_infos(clip_infos, threshold):
    """Remove near-duplicate scenes across videos using scene_hash."""
    selected = []
    for info in clip_infos:
        scene_hash = info['scene'].get('scene_hash')
        if not scene_hash:
            selected.append(info)
            continue
        
        duplicate_idx = None
        for idx, kept in enumerate(selected):
            dist = hash_distance(scene_hash, kept['scene'].get('scene_hash'))
            if dist is not None and dist <= threshold:
                duplicate_idx = idx
                break
        
        if duplicate_idx is None:
            selected.append(info)
        else:
            current_score = info['scene'].get('quality_score', 0)
            kept_score = selected[duplicate_idx]['scene'].get('quality_score', 0)
            if current_score > kept_score:
                selected[duplicate_idx] = info
    return selected


def to_file_uri(path_str):
    path = Path(path_str).expanduser().resolve()
    return f"file://{path.as_posix()}"


def find_rendered_clip(rendered_dir, video_stem, scene_num, classification, speed):
    prefix = f"{video_stem}_" if video_stem else ""
    candidates = [
        f"{prefix}scene_{scene_num:02d}_{classification}_{speed:.2f}x.mkv",
        f"{prefix}scene_{scene_num:02d}_{classification}_{speed:.1f}x.mkv",
        f"{prefix}scene_{scene_num:02d}_{classification}_{speed:g}x.mkv",
    ]
    for name in candidates:
        path = rendered_dir / name
        if path.exists():
            return path

    pattern = f"{prefix}scene_{scene_num:02d}_{classification}_*x.mkv"
    matches = sorted(rendered_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]

    def parse_speed(path):
        try:
            speed_str = path.stem.split('_')[-1].replace('x', '')
            return float(speed_str)
        except (ValueError, IndexError):
            return None

    for match in matches:
        match_speed = parse_speed(match)
        if match_speed is not None and abs(match_speed - speed) <= 0.01:
            return match

    return None


def create_fcpxml_timeline(analysis_path, video_dir, output_file, clip_base_dir='ai_clips', dedupe=False, hash_threshold=6, use_rendered=True, resolve_format=True, exclude_boring=False, config=None):
    """Create FCP XML timeline for DaVinci Resolve"""
    config = config or {}
    timeline_config = config.get('timeline', config)
    analyses = load_analyses(analysis_path, video_dir=video_dir, video_file=video_dir if video_dir and Path(video_dir).is_file() else None)
    if not analyses:
        raise ValueError("No analysis files found.")
    
    fps = 24
    clip_infos = []
    video_rotations = {}
    for analysis in analyses:
        scenes = analysis['data']['scenes']
        video_path = analysis['video_path']
        video_name = analysis['video_name']
        video_stem = Path(video_name).stem
        rendered_dir = Path(clip_base_dir) / video_stem

        if video_name not in video_rotations:
            video_rotations[video_name] = get_video_rotation_degrees(str(video_path))
        
        video_duration_frac = get_video_duration_frac(str(video_path), fps=fps)
        
        for i, scene in enumerate(scenes, 1):
            speed = scene['speed']
            duration_sec = scene['duration']
            output_duration_sec = duration_sec / speed
            output_duration_frames = round(output_duration_sec * fps)
            output_duration_frac = Fraction(output_duration_frames, fps)
            
            classification = scene.get('classification', 'unknown')
            rendered_path = find_rendered_clip(rendered_dir, video_stem, i, classification, speed)
            use_rendered_clip = use_rendered and rendered_path is not None
            
            clip_infos.append({
                'scene': scene,
                'video_name': video_name,
                'video_path': str(video_path),
                'output_duration_frac': output_duration_frac,
                'use_rendered': use_rendered_clip,
                'src_path': str(rendered_path) if use_rendered_clip else str(video_path),
                'src_name': rendered_path.name if use_rendered_clip else video_name,
                'src_duration_frac': output_duration_frac if use_rendered_clip else video_duration_frac,
                'rotation': video_rotations.get(video_name, 0)
            })

    def build_static_clip_info(clip_path, label, clips_dir, preferred_dir=None, copy_to_clips_dir=True):
        clip_path = Path(clip_path).expanduser().resolve()
        if not clip_path.exists():
            print(f"⚠️  Missing {label} clip: {clip_path}")
            return None
        video_name = clip_path.name
        preferred_path = clip_path
        if clips_dir and copy_to_clips_dir:
            clips_dir_path = Path(clips_dir).expanduser().resolve()
            target_dir = Path(preferred_dir).expanduser().resolve() if preferred_dir else clips_dir_path
            candidate = target_dir / video_name
            if not candidate.exists() or candidate.is_symlink():
                try:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    if candidate.is_symlink():
                        candidate.unlink()
                    shutil.copy2(clip_path, candidate)
                except Exception as exc:
                    print(f"⚠️  Could not copy {label} clip to {candidate}: {exc}")
            if candidate.exists():
                preferred_path = candidate
        rotation = get_video_rotation_degrees(str(preferred_path))
        video_rotations[video_name] = rotation
        duration_frac = get_video_duration_frac(str(preferred_path), fps=fps)
        duration_sec = float(duration_frac)
        scene = {
            'start_time': 0.0,
            'end_time': duration_sec,
            'duration': duration_sec,
            'speed': 1.0,
            'classification': label
        }
        return {
            'scene': scene,
            'video_name': video_name,
            'video_path': str(preferred_path),
            'output_duration_frac': duration_frac,
            'use_rendered': True,
            'src_path': str(preferred_path),
            'src_name': video_name,
            'src_duration_frac': duration_frac,
            'rotation': rotation
        }
    
    if dedupe:
        before = len(clip_infos)
        clip_infos = dedupe_clip_infos(clip_infos, hash_threshold)
        after = len(clip_infos)
        if after < before:
            print(f"🔁 Deduped scenes: {before} → {after} (threshold={hash_threshold})")

    if exclude_boring:
        before = len(clip_infos)
        clip_infos = [info for info in clip_infos if info['scene'].get('classification') != 'boring']
        after = len(clip_infos)
        if after < before:
            print(f"🚫 Excluded boring scenes: {before} → {after}")

    intro_path = timeline_config.get('intro_clip')
    outro_path = timeline_config.get('outro_clip')
    copy_intro_outro = timeline_config.get('copy_intro_outro_to_clips_dir', True)
    intro_used_path = None
    outro_used_path = None
    primary_render_dir = None
    if clip_base_dir and analyses:
        first_video_stem = Path(analyses[0]['video_name']).stem
        candidate_dir = Path(clip_base_dir) / first_video_stem
        primary_render_dir = candidate_dir if candidate_dir.exists() else Path(clip_base_dir)
    if intro_path:
        intro_info = build_static_clip_info(
            intro_path,
            'intro',
            clip_base_dir,
            preferred_dir=primary_render_dir,
            copy_to_clips_dir=copy_intro_outro
        )
        if intro_info:
            intro_used_path = intro_info['src_path']
            clip_infos.insert(0, intro_info)
    if outro_path:
        outro_info = build_static_clip_info(
            outro_path,
            'outro',
            clip_base_dir,
            preferred_dir=primary_render_dir,
            copy_to_clips_dir=copy_intro_outro
        )
        if outro_info:
            outro_used_path = outro_info['src_path']
            clip_infos.append(outro_info)
    
    use_rendered_any = any(info['use_rendered'] for info in clip_infos)
    use_rendered_all = all(info['use_rendered'] for info in clip_infos)

    # Calculate timeline duration
    timeline_duration_sec = sum(float(info['output_duration_frac']) for info in clip_infos)
    timeline_duration_frac = Fraction(timeline_duration_sec).limit_denominator(10000)

    watermark_config = timeline_config.get('watermark') or config.get('watermark')
    audio_config = config.get('audio') if isinstance(config.get('audio'), dict) else {}
    background_music_config = (
        timeline_config.get('background_music')
        or audio_config.get('background_music')
        or config.get('background_music')
    )
    audio_role = timeline_config.get('audio_role', 'dialogue')
    watermark_asset_id = None
    watermark_position = None
    watermark_opacity = None
    watermark_path = None
    if watermark_config and isinstance(watermark_config, dict):
        watermark_path = watermark_config.get('path')
        transparency = normalize_transparency(watermark_config.get('transparency'))
        if transparency is None:
            transparency = 0.3  # Default 30% transparent = 70% opaque
        watermark_opacity = 1.0 - transparency  # Convert transparency to opacity
        watermark_position = watermark_config.get('position', 'bottom-right')
    
    # FCP XML structure (version 1.13 - DaVinci Resolve format)
    fcpxml = Element('fcpxml', version='1.13')
    
    # Resources section
    resources = SubElement(fcpxml, 'resources')
    
    # Format r0: Timeline format - 4K 24fps
    SubElement(resources, 'format', {
        'name': 'FFVideoFormat3840x2160p2398',
        'height': '2160',
        'id': 'r0',
        'frameDuration': '1001/24000s',
        'width': '3840'
    })
    
    # Format r2: Source video format
    SubElement(resources, 'format', {
        'name': 'FFVideoFormatRateUndefined',
        'height': '2160',
        'id': 'r2',
        'frameDuration': '1/24s',
        'width': '3840'
    })
    
    # Cross Dissolve effect (r1)
    SubElement(resources, 'effect', {
        'name': 'Cross Dissolve',
        'id': 'r1',
        'uid': 'FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265'
    })
    
    asset_counter = 3
    media_by_video = {}

    if resolve_format:
        # Create media resources per source video (Resolve-style)
        for video_name in sorted({info['video_name'] for info in clip_infos if not info['use_rendered']}):
            video_path = next(info['video_path'] for info in clip_infos if info['video_name'] == video_name)
            duration_frac = next(info['src_duration_frac'] for info in clip_infos if info['video_name'] == video_name)

            video_asset_id = f'r{asset_counter}'
            asset_counter += 1
            audio_asset_id = f'r{asset_counter}'
            asset_counter += 1
            media_id = f'r{asset_counter}'
            asset_counter += 1

            asset_video = SubElement(resources, 'asset', {
                'name': video_name,
                'start': '0/1s',
                'hasAudio': '0',
                'id': video_asset_id,
                'uid': to_file_uri(video_path),
                'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s",
                'hasVideo': '1',
                'audioSources': '0',
                'format': 'r2',
                'audioChannels': '0'
            })
            SubElement(asset_video, 'media-rep', {
                'src': to_file_uri(video_path),
                'kind': 'original-media'
            })

            asset_audio = SubElement(resources, 'asset', {
                'name': video_name,
                'start': '0/1s',
                'hasAudio': '1',
                'id': audio_asset_id,
                'uid': to_file_uri(video_path),
                'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s",
                'hasVideo': '0',
                'audioSources': '1',
                'audioChannels': '1'
            })
            SubElement(asset_audio, 'media-rep', {
                'src': to_file_uri(video_path),
                'kind': 'original-media'
            })

            media = SubElement(resources, 'media', {
                'id': media_id,
                'name': video_name
            })
            media_seq = SubElement(media, 'sequence', {
                'tcStart': '0/1s',
                'tcFormat': 'NDF',
                'format': 'r2',
                'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s"
            })
            media_spine = SubElement(media_seq, 'spine')
            media_clip = SubElement(media_spine, 'clip', {
                'enabled': '1',
                'tcFormat': 'NDF',
                'start': '0/1s',
                'format': 'r2',
                'offset': '0/1s',
                'name': video_name,
                'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s"
            })
            media_transform = {
                'position': '0 0',
                'anchor': '0 0',
                'scale': '1 1'
            }
            rotation = video_rotations.get(video_name, 0)
            if rotation:
                media_transform['rotation'] = str(rotation)
                zoom = timeline_config.get('rotation_zoom', 1.78)
                media_transform['scale'] = f"{zoom} {zoom}"
            SubElement(media_clip, 'adjust-transform', media_transform)
            SubElement(media_clip, 'video', {
                'ref': video_asset_id,
                'start': '0/1s',
                'offset': '0/1s',
                'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s"
            })
            media_by_video[video_name] = {
                'media_id': media_id,
                'video_asset_id': video_asset_id,
                'audio_asset_id': audio_asset_id
            }

    # Create separate asset for rendered clips (or legacy mode)
    legacy_asset_map = {}
    for clip_info in clip_infos:
        if resolve_format and not clip_info['use_rendered'] and not clip_info.get('force_asset_clip'):
            continue
        asset_id = f'r{asset_counter}'
        asset_counter += 1
        asset = SubElement(resources, 'asset', {
            'name': clip_info['src_name'],
            'start': '0/1s',
            'hasAudio': '1',
            'id': asset_id,
            'uid': to_file_uri(clip_info['src_path']),
            'duration': f"{clip_info['src_duration_frac'].numerator}/{clip_info['src_duration_frac'].denominator}s",
            'hasVideo': '1',
            'audioSources': '1',
            'format': 'r2',
            'audioChannels': '2'
        })
        SubElement(asset, 'media-rep', {
            'src': to_file_uri(clip_info['src_path']),
            'kind': 'original-media'
        })
        legacy_asset_map[id(clip_info)] = asset_id

    if watermark_path:
        watermark_path = str(Path(watermark_path).expanduser().resolve())
        if Path(watermark_path).exists():
            watermark_asset_id = f'r{asset_counter}'
            asset_counter += 1
            watermark_asset = SubElement(resources, 'asset', {
                'name': Path(watermark_path).name,
                'start': '0/1s',
                'hasAudio': '0',
                'id': watermark_asset_id,
                'uid': to_file_uri(watermark_path),
                'duration': f"{timeline_duration_frac.numerator}/{timeline_duration_frac.denominator}s",
                'hasVideo': '1',
                'audioSources': '0',
                'format': 'r2',
                'audioChannels': '0'
            })
            SubElement(watermark_asset, 'media-rep', {
                'src': to_file_uri(watermark_path),
                'kind': 'original-media'
            })
        else:
            print(f"⚠️  Missing watermark image: {watermark_path}")

    music_assets = []
    if background_music_config and isinstance(background_music_config, dict):
        music_folder = background_music_config.get('folder')
        if music_folder:
            music_dir = Path(music_folder).expanduser().resolve()
            if music_dir.exists() and music_dir.is_dir():
                music_files = sorted(music_dir.glob('*.wav'))
                if not music_files:
                    print(f"⚠️  No WAV files found in music folder: {music_dir}")
                else:
                    for music_path in music_files:
                        duration_frac, channels = get_audio_info(str(music_path))
                        if not duration_frac:
                            print(f"⚠️  Skipping audio (unknown duration): {music_path}")
                            continue
                        asset_id = f'r{asset_counter}'
                        asset_counter += 1
                        asset = SubElement(resources, 'asset', {
                            'name': music_path.name,
                            'start': '0/1s',
                            'hasAudio': '1',
                            'id': asset_id,
                            'uid': to_file_uri(str(music_path)),
                            'duration': f"{duration_frac.numerator}/{duration_frac.denominator}s",
                            'hasVideo': '0',
                            'audioSources': '1',
                            'audioChannels': str(channels)
                        })
                        SubElement(asset, 'media-rep', {
                            'src': to_file_uri(str(music_path)),
                            'kind': 'original-media'
                        })
                        music_assets.append({
                            'asset_id': asset_id,
                            'path': str(music_path),
                            'duration': duration_frac
                        })
            else:
                print(f"⚠️  Music folder not found: {music_dir}")
    
    # Library structure
    library = SubElement(fcpxml, 'library')
    event = SubElement(library, 'event', name='Timeline 1 (Resolve)')
    project = SubElement(event, 'project', name='Timeline 1 (Resolve)')
    
    sequence = SubElement(project, 'sequence', {
        'tcStart': '0/1s',
        'duration': f'{timeline_duration_frac.numerator}/{timeline_duration_frac.denominator}s',
        'format': 'r0',
        'tcFormat': 'NDF'
    })
    
    spine = SubElement(sequence, 'spine')

    watermark_lane = None
    main_video_lane = None
    try:
        watermark_lane = int(
            watermark_config.get('lane', timeline_config.get('watermark_lane', 2))
            if watermark_config else timeline_config.get('watermark_lane', 2)
        )
    except (TypeError, ValueError):
        watermark_lane = 1
    try:
        main_video_lane = int(timeline_config.get('main_video_lane', 1))
    except (TypeError, ValueError):
        main_video_lane = 2

    # Add clips to timeline
    print(f'🎬 Creating DaVinci Resolve Timeline...')
    video_names = sorted({info['video_name'] for info in clip_infos})
    if len(video_names) == 1:
        print(f'   Original video: {video_names[0]}')
    else:
        print(f'   Videos: {len(video_names)} ({", ".join(video_names)})')
    print(f'   Timeline: {len(clip_infos)} clips')
    print()
    
    timeline_pos_sec = Fraction(0, 1)
    
    transition_seconds = timeline_config.get('transition_duration', None)
    transition_duration = None
    if transition_seconds is None:
        transition_duration = Fraction(1001, 1000)
    else:
        try:
            transition_seconds = float(transition_seconds)
        except (TypeError, ValueError):
            transition_seconds = None
        if transition_seconds and transition_seconds > 0:
            transition_duration = Fraction(transition_seconds).limit_denominator(10000)
    transition_half = transition_duration / 2 if transition_duration else None
    clip_timeline_ranges = []
    timeline_cursor = Fraction(0, 1)
    for idx, info in enumerate(clip_infos):
        clip_start = timeline_cursor
        clip_end = timeline_cursor + info['output_duration_frac']
        clip_timeline_ranges.append((clip_start, clip_end))
        if transition_half and idx < len(clip_infos) - 1:
            timeline_cursor = clip_end - transition_half
        else:
            timeline_cursor = clip_end
    snippet_volume_db = timeline_config.get(
        'snippet_audio_volume_db',
        audio_config.get('snippet_audio_volume_db', None)
    )
    try:
        snippet_volume_db = float(snippet_volume_db) if snippet_volume_db is not None else None
    except (TypeError, ValueError):
        snippet_volume_db = None
    
    for i, clip_info in enumerate(clip_infos, 1):
        scene = clip_info['scene']
        start_sec = scene['start_time']
        end_sec = scene['end_time']
        duration_sec = scene['duration']
        speed = scene['speed']
        classification = scene.get('classification', 'unknown')
        output_duration_frac = clip_info['output_duration_frac']
        output_duration_sec = float(output_duration_frac)
        
        # Round to frame boundaries (24fps) to avoid fractional frames
        fps = 24
        duration_frames = round(duration_sec * fps)
        duration_frac = Fraction(duration_frames, fps)
        
        clip_start_frac = Fraction(0, 1) if clip_info['use_rendered'] else Fraction(start_sec).limit_denominator(10000)
        clip_name = clip_info['src_name'] if clip_info['use_rendered'] else clip_info['video_name']

        clip_offset = timeline_pos_sec
        audio_start_frac = clip_start_frac
        audio_duration_frac = output_duration_frac
        if resolve_format and not clip_info['use_rendered'] and not clip_info.get('force_asset_clip'):
            media_ref = media_by_video[clip_info['video_name']]['media_id']
            audio_ref = media_by_video[clip_info['video_name']]['audio_asset_id']
            clip = SubElement(spine, 'ref-clip', {
                'name': clip_name,
                'ref': media_ref,
                'start': f'{clip_start_frac.numerator}/{clip_start_frac.denominator}s',
                'offset': f'{clip_offset.numerator}/{clip_offset.denominator}s',
                'duration': f'{output_duration_frac.numerator}/{output_duration_frac.denominator}s',
                'lane': str(main_video_lane),
                'audioStart': f'{audio_start_frac.numerator}/{audio_start_frac.denominator}s',
                'audioDuration': f'{audio_duration_frac.numerator}/{audio_duration_frac.denominator}s'
            })
            SubElement(clip, 'conform-rate', {'srcFrameRate': '24'})
            if abs(speed - 1.0) > 0.01:
                frame_epsilon = Fraction(1, fps)
                handle_sec = 45
                handle_frac = Fraction(round(handle_sec * fps), fps)
                available_source = clip_info['src_duration_frac'] - clip_start_frac
                max_source = available_source - handle_frac
                effective_duration_frac = duration_frac
                if max_source > frame_epsilon and duration_frac >= max_source:
                    effective_duration_frac = max_source
                elif available_source > frame_epsilon and duration_frac >= available_source:
                    effective_duration_frac = available_source - frame_epsilon

                timemap = SubElement(clip, 'timeMap', {'frameSampling': 'floor'})
                # Start point
                SubElement(timemap, 'timept', {
                    'time': '0/1s', 'interp': 'linear', 'value': '0/1s'
                })

                # End point
                SubElement(timemap, 'timept', {
                    'time': f'{output_duration_frac.numerator}/{output_duration_frac.denominator}s',
                    'interp': 'linear',
                    'value': f'{effective_duration_frac.numerator}/{effective_duration_frac.denominator}s'
                })
            SubElement(clip, 'adjust-conform', {'type': 'fit'})
        else:
            asset_ref = legacy_asset_map[id(clip_info)]
            clip = SubElement(spine, 'asset-clip', {
                'name': clip_name,
                'ref': asset_ref,
                'start': f'{clip_start_frac.numerator}/{clip_start_frac.denominator}s',
                'offset': f'{clip_offset.numerator}/{clip_offset.denominator}s',
                'duration': f'{output_duration_frac.numerator}/{output_duration_frac.denominator}s',
                'format': 'r2',
                'enabled': '1',
                'tcFormat': 'NDF',
                'lane': str(main_video_lane),
                'audioStart': f'{audio_start_frac.numerator}/{audio_start_frac.denominator}s',
                'audioDuration': f'{audio_duration_frac.numerator}/{audio_duration_frac.denominator}s'
            })
            SubElement(clip, 'conform-rate', {'srcFrameRate': '24'})
            if not clip_info['use_rendered'] and abs(speed - 1.0) > 0.01:
                timemap = SubElement(clip, 'timeMap', {'frameSampling': 'floor'})
                SubElement(timemap, 'timept', {
                    'time': '0/1s',
                    'interp': 'linear',
                    'value': '0/1s'
                })
                SubElement(timemap, 'timept', {
                    'time': f'{output_duration_frac.numerator}/{output_duration_frac.denominator}s',
                    'interp': 'linear',
                    'value': f'{duration_frac.numerator}/{duration_frac.denominator}s'
                })
        
        if snippet_volume_db is not None:
            clip_class = scene.get('classification')
            if clip_class not in ('intro', 'outro'):
                SubElement(clip, 'adjust-volume', {
                    'amount': f"{snippet_volume_db}dB"
                })

        # Add adjust-transform
        clip_transform = {
            'position': '0 0',
            'scale': '1 1',
            'anchor': '0 0'
        }
        if clip_info.get('rotation'):
            clip_transform['rotation'] = str(clip_info['rotation'])
            zoom = timeline_config.get('rotation_zoom', 1.78)
            clip_transform['scale'] = f"{zoom} {zoom}"
        SubElement(clip, 'adjust-transform', clip_transform)

        
        print(f'   Clip {i:02d}: {clip_info["video_name"]} {start_sec:7.1f}s-{end_sec:7.1f}s ({duration_sec:6.1f}s) @ {speed:.1f}x ({speed*100:.0f}%) → {output_duration_sec:6.1f}s [{classification}]')
        
        # Move timeline position forward
        timeline_pos_sec += output_duration_frac
        
        # Add transition to next clip (except for last clip) with overlap
        if transition_duration and i < len(clip_infos):
            transition_offset = timeline_pos_sec - transition_half
            transition = SubElement(spine, 'transition', {
                'name': 'Transition',
                'offset': f'{transition_offset.numerator}/{transition_offset.denominator}s',
                'duration': f'{transition_duration.numerator}/{transition_duration.denominator}s'
            })
            SubElement(transition, 'filter-video', {'name': 'Transition', 'ref': 'r1'})
            SubElement(transition, 'filter-audio', {'name': 'Transition', 'ref': 'r1'})
            timeline_pos_sec -= transition_half

    if watermark_asset_id:
        watermark_size = get_media_dimensions(watermark_path)
        watermark_margin = watermark_config.get('margin', timeline_config.get('watermark_margin', DEFAULT_WATERMARK_MARGIN))
        watermark_width = watermark_config.get('timeline_width', timeline_config.get('timeline_width', DEFAULT_TIMELINE_WIDTH))
        watermark_height = watermark_config.get('timeline_height', timeline_config.get('timeline_height', DEFAULT_TIMELINE_HEIGHT))
        watermark_scale = watermark_config.get('scale', timeline_config.get('watermark_scale', 1.0))
        position_scale_with_zoom = watermark_config.get(
            'position_scale_with_zoom',
            timeline_config.get('watermark_position_scale_with_zoom', True)
        )
        position_multiplier = watermark_config.get(
            'position_multiplier',
            timeline_config.get('watermark_position_multiplier', 1.0)
        )
        if position_scale_with_zoom and isinstance(watermark_position, dict):
            if 'x' in watermark_position and 'y' in watermark_position:
                try:
                    watermark_position = {
                        'x': float(watermark_position['x']) * watermark_scale,
                        'y': float(watermark_position['y']) * watermark_scale
                    }
                except (TypeError, ValueError):
                    pass
        if isinstance(watermark_position, dict) and 'x' in watermark_position and 'y' in watermark_position:
            try:
                watermark_position = {
                    'x': float(watermark_position['x']) * position_multiplier,
                    'y': float(watermark_position['y']) * position_multiplier
                }
            except (TypeError, ValueError):
                pass
        if watermark_size:
            scaled_size = (int(watermark_size[0] * watermark_scale), int(watermark_size[1] * watermark_scale))
        else:
            scaled_size = None
        watermark_position_str = compute_watermark_position(
            watermark_position,
            scaled_size,
            margin=watermark_margin,
            timeline_width=watermark_width,
            timeline_height=watermark_height
        )
        watermark_clip = SubElement(spine, 'asset-clip', {
            'name': Path(watermark_path).name,
            'ref': watermark_asset_id,
            'start': '0/1s',
            'offset': '0/1s',
            'duration': f"{timeline_duration_frac.numerator}/{timeline_duration_frac.denominator}s",
            'format': 'r2',
            'enabled': '1',
            'tcFormat': 'NDF',
            'lane': str(watermark_lane)
        })
        SubElement(watermark_clip, 'adjust-transform', {
            'position': watermark_position_str,
            'scale': f"{watermark_scale} {watermark_scale}",
            'anchor': '0 0'
        })
        SubElement(watermark_clip, 'adjust-blend', {
            'amount': f"{watermark_opacity:.3f}",
            'opacity': f"{watermark_opacity:.3f}"
        })

    if music_assets:
        try:
            music_lane = int(background_music_config.get('audio_lane', 2))
        except (TypeError, ValueError):
            music_lane = 2
        fade_seconds = background_music_config.get('fade_duration', 1.0)
        fade_in_seconds = background_music_config.get('fade_in_duration', fade_seconds)
        fade_out_seconds = background_music_config.get('fade_out_duration', fade_seconds)
        fade_silence_db = background_music_config.get('fade_silence_db', -96.0)

        def _to_duration(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if value <= 0:
                return None
            return Fraction(value).limit_denominator(10000)

        fade_duration = _to_duration(fade_seconds)
        fade_in_duration = _to_duration(fade_in_seconds)
        fade_out_duration = _to_duration(fade_out_seconds)
        seed = background_music_config.get('random_seed')
        rng = random.Random(seed) if seed is not None else random.Random()
        shuffled_music = music_assets[:]
        rng.shuffle(shuffled_music)
        def _fmt_time(value):
            if value is None:
                return None
            return f"{value.numerator}/{value.denominator}s"
        if shuffled_music:
            music_start = Fraction(0, 1)
            music_end = timeline_duration_frac
            intro_end = None
            outro_start = None
            for info, (clip_start, clip_end) in zip(clip_infos, clip_timeline_ranges):
                classification = info['scene'].get('classification')
                if classification == 'intro' and intro_end is None:
                    intro_end = clip_end
                if classification == 'outro' and outro_start is None:
                    outro_start = clip_start
            if intro_end is not None:
                music_start = intro_end
            if outro_start is not None:
                music_end = outro_start
            if music_end < music_start:
                music_end = music_start

            music_pos = music_start
            music_index = 0
            while music_pos < music_end:
                track = shuffled_music[music_index % len(shuffled_music)]
                remaining = music_end - music_pos
                clip_duration = track['duration'] if track['duration'] <= remaining else remaining
                if clip_duration <= 0:
                    break
                clip_start = music_pos
                attributes = {
                    'name': Path(track['path']).name,
                    'ref': track['asset_id'],
                    'start': '0/1s',
                    'offset': f"{clip_start.numerator}/{clip_start.denominator}s",
                    'duration': f"{clip_duration.numerator}/{clip_duration.denominator}s",
                    'enabled': '1',
                    'tcFormat': 'NDF',
                    'lane': str(music_lane),
                    'audioStart': '0/1s',
                    'audioDuration': f"{clip_duration.numerator}/{clip_duration.denominator}s"
                }
                if fade_in_duration:
                    fade_in = fade_in_duration if fade_in_duration <= clip_duration else clip_duration
                    attributes['audioFadeIn'] = _fmt_time(fade_in)
                if fade_out_duration:
                    fade_out = fade_out_duration if fade_out_duration <= clip_duration else clip_duration
                    attributes['audioFadeOut'] = _fmt_time(fade_out)
                music_clip = SubElement(spine, 'asset-clip', attributes)
                SubElement(music_clip, 'adjust-volume', {
                    'amount': '0dB'
                })

                if fade_in_duration or fade_out_duration:
                    fade_in = fade_in_duration if fade_in_duration else None
                    fade_out = fade_out_duration if fade_out_duration else None
                    if fade_in and fade_in > clip_duration:
                        fade_in = clip_duration
                    if fade_out and fade_out > clip_duration:
                        fade_out = clip_duration
                    fade_out_start = clip_duration - fade_out if fade_out else None
                    if fade_out_start is not None and fade_out_start < Fraction(0, 1):
                        fade_out_start = Fraction(0, 1)

                    try:
                        fade_silence_value = f"{float(fade_silence_db)}dB"
                    except (TypeError, ValueError):
                        fade_silence_value = "-96.0dB"
                    fade_full_value = "0dB"

                    audio_automation = SubElement(music_clip, 'audio-automation', {
                        'lane': 'volume'
                    })
                    if fade_in:
                        SubElement(audio_automation, 'keyframe', {
                            'time': '0/1s',
                            'value': fade_silence_value
                        })
                        SubElement(audio_automation, 'keyframe', {
                            'time': _fmt_time(fade_in),
                            'value': fade_full_value
                        })
                    else:
                        SubElement(audio_automation, 'keyframe', {
                            'time': '0/1s',
                            'value': fade_full_value
                        })

                    if fade_out and fade_out_start is not None:
                        SubElement(audio_automation, 'keyframe', {
                            'time': _fmt_time(fade_out_start),
                            'value': fade_full_value
                        })
                        SubElement(audio_automation, 'keyframe', {
                            'time': _fmt_time(clip_duration),
                            'value': fade_silence_value
                        })
                    else:
                        SubElement(audio_automation, 'keyframe', {
                            'time': _fmt_time(clip_duration),
                            'value': fade_full_value
                        })
                transition_duration = fade_out_duration or fade_duration
                if transition_duration:
                    if transition_duration > clip_duration:
                        transition_duration = clip_duration
                    transition_start = clip_start + clip_duration - transition_duration
                    if transition_start < clip_start:
                        transition_start = clip_start
                    transition_end = clip_start + clip_duration
                    if transition_end > transition_start:
                        SubElement(spine, 'transition', {
                            'start': _fmt_time(transition_start),
                            'end': _fmt_time(transition_end),
                            'type': 'audioCrossfade'
                        })
                music_pos += clip_duration
                music_index += 1
    
    # Add final gap
    gap = SubElement(spine, 'gap', {
        'name': 'Gap',
        'start': '3600/1s',
        'offset': f'{timeline_pos_sec.numerator}/{timeline_pos_sec.denominator}s',
        'duration': '1001/2000s'
    })
    
    # Write XML file with proper formatting
    tree = ElementTree(fcpxml)
    
    # Custom indent function for proper formatting
    def indent_xml(elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level+1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent_xml(fcpxml)
    
    # Write with DOCTYPE
    with open(output_file, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(b'<!DOCTYPE fcpxml>\n')
        tree.write(f, encoding='utf-8', xml_declaration=False)
    
    total_input = sum(info['scene']['duration'] for info in clip_infos)
    total_output = sum(info['scene']['duration'] / info['scene']['speed'] for info in clip_infos)
    compression = (1 - total_output / total_input) * 100
    
    print()
    print('='*70)
    print('📊 Timeline Export Complete')
    print('='*70)
    print(f'  Format:          FCP XML 1.13 (DaVinci Resolve)')
    print(f'  Original:        {total_input/60:.1f} min')
    print(f'  Timeline:        {total_output/60:.1f} min')
    print(f'  Compression:     {compression:.0f}%')
    print(f'  Clips:           {len(clip_infos)}')
    print('='*70)
    print()
    print('⚠️  IMPORTANT - Import Steps for DaVinci Resolve:')
    print()
    print('   1. Open DaVinci Resolve')
    intro_exists = bool(intro_used_path and Path(intro_used_path).expanduser().resolve().exists())
    outro_exists = bool(outro_used_path and Path(outro_used_path).expanduser().resolve().exists())

    if use_rendered_all:
        print('   2. Import rendered clips to Media Pool FIRST')
        print('      (File → Import Media → Select all files in ai_clips/*/ including .mov intro/outro)')
    elif use_rendered_any:
        print('   2. Import source videos AND rendered clips to Media Pool FIRST')
        print('      (File → Import Media → Select videos and ai_clips/*/*.mkv)')
    else:
        print('   2. Import source videos to Media Pool FIRST')
        print('      (File → Import Media → Select all input videos)')
    if intro_exists or outro_exists:
        extra = []
        if intro_exists:
            extra.append(str(Path(intro_used_path).expanduser().resolve()))
        if outro_exists:
            extra.append(str(Path(outro_used_path).expanduser().resolve()))
        print('      Also import intro/outro clips:')
        for path in extra:
            print(f'      - {path}')
    print('   3. Then import timeline:')
    print('      File → Import → Timeline → Import AAF/EDL/XML')
    print(f'      Select: {output_file}')
    print()
    print('   ⚠️  Media MUST be in Media Pool before importing XML!')
    print()
    if use_rendered_any:
        print('✅ Timeline uses pre-rendered clips to avoid timeMap issues.')
    else:
        print('✅ No re-encoding needed! Timeline uses original video with speed changes.')
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export DaVinci Resolve FCPXML timeline')
    parser.add_argument('--analysis', default='scene_analysis_advanced3.json', help='Analysis JSON file or directory')
    parser.add_argument('--video-dir', default='.', help='Directory containing source videos')
    parser.add_argument('--output', default='timeline_davinci_resolve.fcpxml', help='Output FCPXML file')
    parser.add_argument('--clips-dir', default='ai_clips', help='Base directory for rendered clips')
    parser.add_argument('--use-rendered', dest='use_rendered', action='store_true', default=True,
                        help='Use pre-rendered clips when available (default)')
    parser.add_argument('--use-original', dest='use_rendered', action='store_false',
                        help='Use original videos instead of rendered clips')
    parser.add_argument('--legacy-asset-format', action='store_true', help='Use legacy asset-clip format instead of Resolve ref-clip format')
    parser.add_argument('--exclude-boring', action='store_true', help='Exclude boring scenes from the timeline')
    parser.add_argument('--dedupe', action='store_true', help='Remove near-duplicate scenes across videos')
    parser.add_argument('--hash-threshold', type=int, default=6, help='Hamming distance threshold for dedupe')
    parser.add_argument('--config', default='project_config.json', help='Project config JSON file')
    args = parser.parse_args()
    
    analysis_path = args.analysis
    if not os.path.exists(analysis_path):
        print(f'❌ Analysis path not found: {analysis_path}')
        sys.exit(1)
    
    create_fcpxml_timeline(
        analysis_path,
        args.video_dir,
        args.output,
        clip_base_dir=args.clips_dir,
        dedupe=args.dedupe,
        hash_threshold=args.hash_threshold,
        use_rendered=args.use_rendered,
        resolve_format=not args.legacy_asset_format,
        exclude_boring=args.exclude_boring,
        config=load_project_config(args.config)
    )
