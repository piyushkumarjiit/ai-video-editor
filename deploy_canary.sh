#!/bin/bash

# --- CONFIGURATION ---
ENV_FILE=".env"
IMAGE_NAME="nvcr.io/nim/nvidia/canary-1b:latest"
CONTAINER_NAME="speech-nim"

# --- 1. VALIDATION ---
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found."
    exit 1
fi

# Load variables from .env
# We use grep to avoid comments and leading/trailing whitespace
NGC_API_KEY=$(grep -v '^#' "$ENV_FILE" | grep 'NGC_API_KEY' | cut -d '=' -f2 | xargs)

if [ -z "$NGC_API_KEY" ]; then
    echo "❌ Error: NGC_API_KEY is not set in $ENV_FILE."
    exit 1
fi

echo "✅ Found NGC_API_KEY in $ENV_FILE."

# --- 2. DOCKER LOGIN ---
echo "🔑 Attempting to login to NVIDIA Container Registry (nvcr.io)..."
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

if [ $? -ne 0 ]; then
    echo "❌ Error: Docker login failed. Please check your NGC_API_KEY."
    exit 1
fi

echo "✅ Login Succeeded."

# --- 3. CLEANUP OLD CONTAINER ---
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🧹 Removing existing container: $CONTAINER_NAME..."
    docker rm -f $CONTAINER_NAME > /dev/null
fi

# --- 4. EXECUTION ---
echo "🚀 Starting $CONTAINER_NAME..."

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  --gpus all \
  --env-file "$ENV_FILE" \
  --shm-size=8GB \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 9000:9000 \
  -p 50051:50051 \
  "$IMAGE_NAME"

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "✅ Success! Canary NIM is starting up."
    echo "📍 Check progress: docker logs -f $CONTAINER_NAME"
    echo "📍 Health check: curl -X 'GET' 'http://localhost:9000/v1/health/ready'"
    echo "--------------------------------------------------------"
else
    echo "❌ Error: Failed to start the Docker container."
    exit 1
fi