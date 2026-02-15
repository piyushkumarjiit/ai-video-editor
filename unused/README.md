# Scale Model Video Analysis Pipeline

## Project Goal
Automatically process scale modeling videos (5-25min iPhone 10-bit recordings) to identify and extract only meaningful scenes, filtering out repetitive/boring segments.

## Workflow

### Phase 1: Scene Detection & Splitting
**Script:** `split3.py`

- Reads scene timestamps from PySceneDetect CSV output
- Splits video into individual scene clips using ffmpeg
- Preserves 10-bit HEVC quality with stream copy
- Filters out micro-scenes (<3 seconds)
- **Output:** Individual scene clips in `clips/` directory

### Phase 2: Content Analysis & Speed Adjustment
**Script:** `analyze_scenes.py`

**Analysis Method:**
1. Extracts frames at 1 fps from each scene clip
2. Converts frames to HSV color space
3. Calculates histogram differences between consecutive frames
4. Scores visual change rate (higher = more activity/progress)

**Classification:**
- **Interesting (1x speed):** High visual change (score ≥ 15)
  - Examples: New parts appearing, fresh paint, polishing results
- **Moderate (2x speed):** Medium change (score 5-15)
  - Examples: Ongoing assembly, moderate activity
- **Boring (4x speed):** Low/no change (score < 5)
  - Examples: Repetitive polishing, static shots

**Outputs:**
- `scene_analysis.json` - Full analysis report with scores
- `clips_speed/` - Speed-adjusted clips (optional)
- Time saved statistics

### Phase 3: Cutting (Future)
Once algorithm is validated, boring scenes can be cut completely instead of sped up.

## Key Features

### What Makes Scenes "Interesting"
- ✅ New components appearing
- ✅ Visible progress: bare → primed → painted → polished
- ✅ Assembly actions (parts joining)
- ✅ Before/after differences
- ✅ New shine, texture changes

### What Gets Sped Up/Cut
- ❌ Static repetitive actions (same polishing for minutes)
- ❌ No visible progress
- ❌ Duplicate angles of same state

## Video Specs
- **Source:** iPhone 16 (HEVC Main 10 profile)
- **Format:** 3840x2160, 10-bit yuv420p10le, HDR (HLG)
- **Duration:** 5-25 minutes typical
- **Bitrate:** ~20 Mbps

## Dependencies
- Python 3.x
- ffmpeg (with HEVC support)
- opencv-python
- numpy

## Usage

```bash
# Step 1: Split video into scenes
python split3.py

# Step 2: Analyze and optionally speed-adjust
python analyze_scenes.py
```

## Future Improvements
- Machine learning for better scene classification
- Scene similarity detection (remove duplicates)
- Motion analysis for assembly sequences
- Custom thresholds per video type (painting vs polishing vs assembly)
