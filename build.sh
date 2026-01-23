#!/bin/bash
set -e

echo "Building Low-Latency Jailbreak Detection Service..."
echo ""

# Detect if sudo is needed for docker
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
    if sudo docker ps >/dev/null 2>&1; then
        echo "Docker requires sudo in this environment"
        DOCKER_CMD="sudo docker"
    else
        echo "ERROR: Cannot access Docker (tried with and without sudo)"
        exit 1
    fi
fi

# Load .env file
if [ -f .env ]; then
    echo "Loading HF_TOKEN from .env file"
    export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
else
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your HF_TOKEN"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not found in .env file!"
    echo "Please add HF_TOKEN=your_token to .env"
    exit 1
fi

echo "HF_TOKEN found"
echo ""
echo "Building Docker image (model will be pre-downloaded)..."
$DOCKER_CMD build --build-arg HF_TOKEN="$HF_TOKEN" -t lowlatency-jailbreak .

echo ""
echo "Build complete!"
echo ""
echo "Next steps:"
echo "docker run --gpus all --env-file .env -p 8001:8001 -d --name jb-server lowlatency-jailbreak"
