#!/bin/bash
# archive_env.sh - Snapshots current AI environments

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="env_backups/$TIMESTAMP"
mkdir -p "$ARCHIVE_DIR"

echo "📸 Archiving environment states to $ARCHIVE_DIR..."

# Snapshot Env 1 (Main)
~/.virtualenvs/ai-video-env/bin/python -m pip freeze > "$ARCHIVE_DIR/requirements_env1_main.txt"

# Snapshot Env 2 (Denoise)
~/.virtualenvs/ai-video-v2/bin/python -m pip freeze > "$ARCHIVE_DIR/requirements_env2_denoise.txt"

# Snapshot System Info (CUDA/Drivers)
nvidia-smi > "$ARCHIVE_DIR/nvidia_state.txt"
python3 cuda_active_check.py > "$ARCHIVE_DIR/gpu_diag_results.txt"

echo "✅ Snapshot complete. To restore later, use: pip install -r $ARCHIVE_DIR/requirements_env1_main.txt"