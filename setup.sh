#!/bin/bash

# 1. Configuration - Change this if you want a different env name
ENV_PATH="$HOME/.virtualenvs/ai-video-env"
PROJECT_ROOT=$(pwd)

echo "🚀 Starting environment setup for $PROJECT_ROOT"

# 2. Install System Dependencies (Build tools for llama-cpp, etc.)
echo "📦 Checking system dependencies (may ask for sudo password)..."
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev

# 3. Create Virtual Environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    echo "🐍 Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
else
    echo "✅ Virtual environment already exists."
fi

# 4. Activation & Installation
echo "🔌 Activating environment and installing requirements..."
source "$ENV_PATH/bin/activate"

# Upgrade pip first to avoid "broken pip" errors
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✨ All Python libraries installed successfully!"
else
    echo "⚠️ Warning: requirements.txt not found. Skipping library install."
fi

echo "🎉 Setup complete! Run 'source $ENV_PATH/bin/activate' to start working."