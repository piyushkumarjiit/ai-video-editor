# AI Video Editing Pipeline

An intelligent video editing automation system that uses computer vision and large language models to analyze, classify, and automatically edit long-form videos into engaging, compressed timelines ready for DaVinci Resolve.

## Overview

This pipeline transforms lengthy raw footage (30-60+ minutes) into polished, watchable videos by automatically detecting scene quality, adjusting playback speeds, extracting highlight moments, generating professional timelines with music/transitions/watermarks, rendering in DaVinci Resolve, and uploading to YouTube.

**Key Features:**
- AI-powered scene classification (boring, low, moderate, interesting)
- Automated speed ramping (1x-6x) based on content quality
- Showcase moment extraction for teaser sections
- Intelligent duplicate scene detection across multiple videos
- DaVinci Resolve FCPXML timeline generation
- Optional LUT application in Resolve Media Pool
- YouTube rendering (H.265, 4K, bitrate control)
- YouTube upload with OAuth 2.0, playlist support, and thumbnails
- Multi-track audio with background music and teaser soundtracks
- Configurable watermarks with opacity and positioning
- GPU-accelerated video processing (NVENC)

## Pipeline Architecture

```
INPUT: Raw Videos (MOV/MP4)
         |
         v
+-------------------------+
| STAGE 1: ANALYSIS       |
| (analyze_advanced5.py)  |
+-------------------------+
| Pass 1: Frame Analysis  |
|  - Sample every 2s      |
|  - ResNet-50 features   |
|  - Qwen2.5-VL captions  |
|  - CLIP embeddings      |
|  - Perceptual hashes    |
+-------------------------+
         |
         v
+-------------------------+
| Scene Detection         |
|  - Visual transitions   |
|  - Logical segmentation |
+-------------------------+
         |
         v
+-------------------------+
| Pass 2: Classification  |
|  - LLM quality rating   |
|  - Classify scenes      |
|  - Showcase moments     |
|  - Assign speeds        |
+-------------------------+
         |
         v
OUTPUT: scene_analysis_*.json
         |
         v
+-------------------------+
| STAGE 2: EXTRACTION     |
| (extract_scenes.py)     |
+-------------------------+
| Scene Clip Extraction   |
|  - FFmpeg extraction    |
|  - Speed adjustments    |
|  - GPU HEVC encode      |
|  - Skip boring scenes   |
+-------------------------+
         |
         v
+-------------------------+
| Showcase Extraction     |
|  - 5-sec clips at 1x    |
|  - Center on highlights |
+-------------------------+
         |
         v
OUTPUT: ai_clips/
         |
         v
+-------------------------+
| STAGE 3: EXPORT         |
| (export_resolve.py)     |
+-------------------------+
| Teaser Assembly         |
|  - Top showcase clips   |
|  - Interesting scenes   |
|  - Max 45 seconds       |
+-------------------------+
         |
         v
+-------------------------+
| Timeline Build          |
|  1. Teaser clips        |
|  2. Intro video         |
|  3. Main content        |
|  4. Outro video         |
|                         |
|  - Filter boring        |
|  - Deduplicate scenes   |
|  - Cross-dissolves      |
|  - Rotation transforms  |
+-------------------------+
         |
         v
+-------------------------+
| Audio + Watermark       |
|  Lane 1: Teaser music   |
|  Lane 2: Background     |
|  Video: Muted (-96dB)   |
|  Watermark: 70% opacity |
+-------------------------+
         |
         v
OUTPUT: timeline.fcpxml
     |
     v
+-------------------------+
| DaVinci Resolve Import  |
+-------------------------+
     |
     v
+-------------------------+
| Optional LUT Apply      |
| (apply_lut_resolve.py)  |
+-------------------------+
     |
     v
+-------------------------+
| Render (render_youtube) |
+-------------------------+
     |
     v
OUTPUT: MP4 (4K, H.265)
     |
     v
+-------------------------+
| Upload (upload_youtube) |
+-------------------------+
     |
     v
   YouTube Video
```

## Scene Classification System

The AI analyzes video content and assigns classifications that determine playback speed:

| Classification | Speed | Use Case | Description |
|---------------|-------|----------|-------------|
| **Interesting** | 1.0x | Key moments | High-action, critical content, showcase-worthy |
| **Moderate** | 2.0x | Standard content | Average interest, clear context needed |
| **Low** | 4.0x | Background activity | Minor details, setup, transitions |
| **Boring** | 6.0x | Filler content | Repetitive, minimal value (optional skip) |
| **Skip** | N/A | Excluded | Unusable footage (not exported) |

### Compression Example

```
Original Video:    45.3 minutes
├─ Interesting:     1 scene  @1x  →  0.6 min
├─ Moderate:       35 scenes @2x  →  5.2 min
├─ Low:            31 scenes @4x  →  2.4 min
└─ Boring:         12 scenes @6x  →  0.5 min (excluded)
                                    ─────────
Final Timeline:    14.7 minutes     (64% compression)
```

## AI Models and Technologies

### Computer Vision Models

```
+-------------------------------------+
| Qwen2.5-VL-7B (Vision-Language)     |
| - Frame captions                    |
| - Quality rating (1-10)             |
| - Scene classification              |
| - Cache: ~/.cache/huggingface/      |
| - Size: ~4.7GB (Q4_K_M GGUF)        |
+-------------------------------------+
              |
              v
+-------------------------------------+
| CLIP ViT-B/32 (Contrastive)         |
| - Semantic embeddings               |
| - Text-image matching               |
| - Cache: ~/.cache/clip/             |
+-------------------------------------+
              |
              v
+-------------------------------------+
| ResNet-50 (Feature Extraction)      |
| - Visual features (2048-dim)        |
| - Perceptual similarity             |
| - Cache: ~/.cache/torch/hub/        |
+-------------------------------------+
```

### Analysis Workflow

1. **Frame Sampling**: Extract frames at 2-second intervals
2. **Caption Generation**: Qwen2.5-VL describes visual content
3. **Feature Extraction**: ResNet-50 extracts 2048-dim features
4. **Semantic Encoding**: CLIP generates embeddings
5. **Hash Computation**: Perceptual hashing for scene detection
6. **Scene Segmentation**: Group frames into logical scenes
7. **LLM Classification**: Rate and classify each scene
8. **Speed Assignment**: Map classification to playback speed

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for analysis and encoding)
- FFmpeg with NVENC support
- DaVinci Resolve (for final editing)

### Required Environment (Verified)

- **OS**: Fedora 43 (Workstation)
- **GPU**: NVIDIA GPU with NVENC support
- **RAM**: 32 GB+ recommended (16 GB minimum)
- **Storage**: SSD recommended (20 GB+ free for cache and outputs)
- **DaVinci Resolve**: 20.x (automation verified on 20.0.1)

**Downloads:**
- DaVinci Resolve: https://www.blackmagicdesign.com/support/family/davinci-resolve-and-fusion
- Filmic LUT Pack (iPhone): https://www.filmicpro.com/products/luts/
- Filmic LUT Pack direct download: https://www.filmicpro.com/downloads/Filmic_Pro_deLOG_LUT_Pack_May_2022.zip

### Setup

```bash
# Clone repository
cd /home/mazsola/video

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AI models (automatic on first run)
# Models will be cached to ~/.cache/huggingface/ and ~/.cache/clip/

### Python Requirements

All Python dependencies are pinned in [requirements.txt](requirements.txt).
```

### Directory Structure


```
/home/mazsola/video/
├── analyze_advanced5.py          # Stage 1: AI video analysis
├── extract_scenes.py              # Stage 2: Scene extraction
├── export_resolve.py              # Stage 3: Timeline export
├── run_pipeline.py                # Master orchestrator
├── apply_lut_resolve.py            # Optional LUT application in Resolve
├── render_youtube.py               # Render timeline to MP4 (Resolve API)
├── upload_youtube.py               # Upload to YouTube + thumbnail
├── project_config.json            # Configuration file
├── assets/
│   ├── Start-Intro-V3.mov        # Intro video (10-bit)
│   ├── Finish-Intro-V3.mov       # Outro video (10-bit)
│   ├── qr-code.jpg                # Watermark image
│   ├── music-background/          # Background music (WAV)
│   └── music-teaser/              # Teaser music (WAV)
├── ai_clips/                      # Extracted scene clips
│   ├── {video_stem}/
│   │   ├── *_scene_*.mov
│   │   └── *_showcase_*.mov
└── timeline_davinci_resolve.fcpxml # Final timeline
```

## Usage

### Quick Start

```bash
# Run complete pipeline
python run_pipeline.py

# This executes all three stages:
# 1. Analyze videos
# 2. Extract scenes
# 3. Export timeline
```

### Render and Upload

```bash
# Apply LUTs (optional)
python apply_lut_resolve.py --config project_config.json

# Render from DaVinci Resolve
python render_youtube.py --output /home/mazsola/Videos/output.mp4

# Upload to YouTube (uses project_config.json defaults)
python upload_youtube.py --video /home/mazsola/Videos/output.mp4
```

### Individual Stages

```bash
# Stage 1: Analyze video (generates scene_analysis_*.json)
python analyze_advanced5.py --video INPUT.MOV

# Stage 2: Extract clips (creates ai_clips/ directory)
python extract_scenes.py --analysis-dir . --output-dir ai_clips

# Stage 3: Export timeline (generates timeline_davinci_resolve.fcpxml)
python export_resolve.py --config project_config.json \
                         --analysis . \
                         --video-dir . \
                         --clips-dir ai_clips \
                         --output timeline_davinci_resolve.fcpxml
```

### Command-Line Options

#### analyze_advanced5.py

```bash
--video PATH              # Input video file
--sample-interval SECS    # Frame sampling rate (default: 2)
--llm-batch-size N        # LLM processing batch size (default: 10)
--gpu                     # Enable GPU acceleration
```

#### extract_scenes.py

```bash
--config PATH             # Project config file
--analysis-dir PATH       # Directory with scene_analysis_*.json
--video-dir PATH          # Source video directory
--output-dir PATH         # Output directory for clips
--exclude-boring          # Skip boring scenes during extraction
```

#### export_resolve.py

```bash
--config PATH             # Project config file
--analysis PATH           # Analysis JSON or directory
--video-dir PATH          # Source video directory
--clips-dir PATH          # Extracted clips directory
--output PATH             # Output FCPXML file
--use-rendered            # Use pre-rendered clips (default)
--use-original            # Use original videos with speed changes
--exclude-boring          # Exclude boring scenes from timeline
--dedupe                  # Remove duplicate scenes across videos
--hash-threshold N        # Hamming distance for deduplication (default: 6)
```

## Configuration

### project_config.json

```json
{
  "paths": {
    "input_dir": "./",
    "output_dir": "./",
    "clips_dir": "./ai_clips",
    "timeline": "./timeline_davinci_resolve.fcpxml"
  },
  "analysis": {
    "sample_interval": 2,
    "target_output_ratio": 0.15,
    "max_speed_multiplier": 8.0,
    "captioning": {
      "enabled": true,
      "model": "Qwen/Qwen2.5-VL-3B-Instruct",
      "device": "cuda"
    }
  },
  "pipeline": {
    "dedupe": false,
    "hash_threshold": 6,
    "use_rendered": true,
    "exclude_boring": true
  },
  "timeline": {
    "intro_clip": "./assets/Start-Intro-V3.mkv",
    "outro_clip": "./assets/Finish-Intro-V3.mkv",
    "teaser_enabled": true,
    "teaser_max_duration": 45.0,
    "exclude_boring": true,
    "rotation_zoom": 1.78,
    "transition_duration": 1.0,
    "watermark": {
      "path": "./qr-code.jpg",
      "position": {"x": 3059.0, "y": -890.0},
      "transparency": 0.3
    },
    "background_music": {
      "folder": "./assets/music-background",
      "audio_lane": 2,
      "fade_duration": 3.0
    },
    "snippet_audio_volume_db": -96
  },
  "audio": {
    "teaser_music": {
      "folder": "./assets/music-teaser",
      "audio_lane": 1,
      "fade_duration": 1.0
    }
  },
  "youtube": {
    "channel_url": "https://www.youtube.com/@modernhackers",
    "upload_title": "Scale Model Car Build",
    "default_description": "...",
    "category_id": "26",
    "default_privacy": "unlisted",
    "made_for_kids": false,
    "altered_content": false,
    "default_playlist_id": "PLxxxxxxxxxxxxxxxxx"
  }
}
```

### Configuration Options

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `pipeline` | `exclude_boring` | Skip boring scenes globally | `true` |
| `pipeline` | `use_rendered` | Use pre-rendered clips | `true` |
| `pipeline` | `dedupe` | Remove duplicate scenes | `false` |
| `timeline` | `teaser_enabled` | Include teaser section | `true` |
| `timeline` | `teaser_max_duration` | Teaser length (seconds) | `45.0` |
| `timeline` | `rotation_zoom` | Zoom factor for rotated clips | `1.78` |
| `timeline` | `transition_duration` | Cross-dissolve duration | `1.0` |
| `watermark` | `transparency` | Watermark transparency (0-1) | `0.3` |
| `background_music` | `fade_duration` | Music fade in/out (seconds) | `3.0` |

## Output Format

### Timeline Structure

```
Timeline: timeline_davinci_resolve.fcpxml (FCPXML 1.13)

Video Tracks:
├─ V1 (Lane 0): Main video clips
│   ├─ Teaser clips (showcase moments)
│   ├─ Intro video
│   ├─ Scene clips (classified and speed-adjusted)
│   └─ Outro video
└─ V2 (Lane 1): Watermark overlay

Audio Tracks:
├─ A1 (Lane 1): Teaser music (one random track)
├─ A2 (Lane 2): Background music (shuffled, crossfaded)
└─ Video audio: -96dB (muted)

Effects:
├─ Cross-dissolve transitions (1 second overlap)
├─ Rotation transforms (270° for portrait videos)
├─ Zoom adjustments (1.78x for rotated clips)
└─ Audio fades (1s teaser, 3s background)
```

### Import to DaVinci Resolve

1. **Import Media First**:
   ```
   File → Import Media
   - Select all files in ai_clips/*/ directories
   - Include Start-Intro-V3.mkv and Finish-Intro-V3.mkv
   - Add music files from assets/music-*
   - Add watermark image (qr-code.jpg)
   ```

2. **Import Timeline**:
   ```
   File → Import → Timeline → Import AAF/EDL/XML
   - Select timeline_davinci_resolve.fcpxml
   - Verify all media is linked (no red clips)
   ```

3. **Verify Settings**:
   - Timeline resolution: 3840x2160 (4K)
   - Frame rate: 24 fps
   - Audio channels: Stereo (48kHz)
   - Color space: Rec. 709

## Performance Optimization

### GPU Acceleration

The pipeline uses GPU acceleration at multiple stages:

- **Analysis**: CUDA for model inference (Qwen, CLIP, ResNet)
- **Extraction**: NVENC for hardware video encoding
- **Speed**: 3-5x faster than CPU-only processing

### Disk Space Requirements

```
Input Videos:      ~10GB (45 min @ 1080p)
Analysis Data:     ~50MB (JSON + embeddings)
Extracted Clips:   ~3GB (pre-rendered with speed)
AI Model Cache:    ~6GB (one-time download)
                   ─────
Total:             ~19GB per project
```

### Processing Time Estimates

| Stage | Duration | GPU | CPU-Only |
|-------|----------|-----|----------|
| Analysis (45 min video) | Pass 1 | 5 min | 20 min |
| Analysis (45 min video) | Pass 2 | 3 min | 8 min |
| Extraction (79 scenes) | GPU Encode | 8 min | 25 min |
| Timeline Export | XML Gen | 5 sec | 5 sec |
| **Total** | | **16 min** | **53 min** |

## Troubleshooting

### Common Issues

**Issue**: Missing AI models on first run
```bash
# Solution: Models download automatically
# Check cache: ls -lh ~/.cache/huggingface/hub/
```

**Issue**: NVENC encoding fails
```bash
# Solution: Falls back to CPU (libx265)
# Check GPU: nvidia-smi
# Verify NVENC: ffmpeg -encoders | grep nvenc
```

**Issue**: DaVinci Resolve shows red clips
```bash
# Solution: Import media before timeline
# Verify paths in FCPXML match actual file locations
```

**Issue**: Watermark opacity incorrect
```bash
# Solution: Set transparency in config (0.0-1.0)
# 0.3 transparency = 70% opaque
```

**Issue**: YouTube upload fails or shows 0% in Studio
```bash
# Solution: Use resumable upload (default) and keep the terminal open
# Large files take time to process in Studio after upload completes
```

**Issue**: Thumbnail rejected or stretched
```bash
# Solution: Use upload_youtube.py thumbnail support (auto-resize to 1280x720)
# Provide --thumbnail or place images in assets/photos/
```

**Issue**: Timeline too long/short
```bash
# Solution: Adjust exclude_boring setting
# Enable: 59% compression (excludes boring)
# Disable: 64% compression (includes all)
```

## Advanced Features

### Teaser Section

Automatically creates a 30-50 second teaser from:
- Top-rated showcase moments (rating 9-10)
- Interesting scene clips (rating 8+)

Sorted by quality score and limited to `teaser_max_duration`.

### Duplicate Detection

Cross-video deduplication using perceptual hashing:
```bash
python export_resolve.py --dedupe --hash-threshold 6
```

Hamming distance threshold:
- 0-5: Identical/near-identical scenes
- 6-10: Similar scenes (default)
- 11-15: Visually related
- 16+: Different scenes

### Multi-Video Projects

Process multiple videos in one timeline:
```bash
# Analyze all videos
for video in *.MOV; do
    python analyze_advanced5.py --video "$video"
done

# Extract all scenes
python extract_scenes.py --analysis-dir .

# Export combined timeline
python export_resolve.py --analysis . --dedupe
```

## Technical Details

### Video Encoding Settings

**Extraction** (HEVC NVENC):
```
Codec:     HEVC (H.265)
Encoder:   hevc_nvenc
Preset:    p4 (balanced)
Quality:   CQ 23
Container: Matroska (MKV)
Audio:     PCM 16-bit 48kHz stereo
```

**Speed Adjustment**:
```
Video:     setpts=PTS/{speed},fps=24
Audio:     atempo chain (max 2.0 per stage)
```

### FCPXML Format

DaVinci Resolve-compatible FCPXML 1.13 with:
- Asset references (file:// URIs)
- Ref-clip format for original videos
- Asset-clip format for rendered clips
- TimeMap elements for speed changes
- Adjust-transform for rotation/zoom
- Adjust-blend for opacity
- Audio automation for fades

## License

Copyright 2026. All rights reserved.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: February 8, 2026  
**Platform**: Linux (CUDA required for GPU acceleration)
