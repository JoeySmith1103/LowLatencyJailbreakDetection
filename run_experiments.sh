#!/bin/bash
set -e

# Detect if sudo is needed for docker
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
    if sudo docker ps >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    else
        echo "ERROR: Cannot access Docker"
        exit 1
    fi
fi

echo "=========================================="
echo "Running Experiments"
echo "=========================================="
echo ""

# Check if container is running
if ! $DOCKER_CMD ps | grep -q jb-server; then
    echo "ERROR: Container 'jb-server' is not running!"
    echo "Please start the service first:"
    echo "  $DOCKER_CMD run --gpus all --env-file .env -p 8001:8001 -d --name jb-server lowlatency-jailbreak"
    exit 1
fi

echo "Container is running. Starting experiments..."
echo ""

# Experiment 1: Ablation Study
echo "=========================================="
echo "Experiment 1: Ablation Study"
echo "=========================================="
$DOCKER_CMD exec jb-server python3 /app/run_ablation_study.py \
  --data /app/LLMSafetyAPIService_data.json \
  --api-url http://localhost:8001 \
  --runs 3

echo ""
echo "=========================================="
echo "Experiment 2: Threshold Analysis"
echo "=========================================="
$DOCKER_CMD exec jb-server python3 /app/analyze_threshold.py \
  --data /app/LLMSafetyAPIService_data.json \
  --api-url http://localhost:8001 \
  --thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
  --case-study

echo ""
echo "=========================================="
echo "Copying results to ./images/"
echo "=========================================="
mkdir -p ./images
$DOCKER_CMD cp jb-server:/app/ablation_latency.png ./images/
$DOCKER_CMD cp jb-server:/app/ablation_tradeoff.png ./images/
$DOCKER_CMD cp jb-server:/app/analysis_latency_vs_threshold.png ./images/
$DOCKER_CMD cp jb-server:/app/analysis_precision_recall.png ./images/

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to ./images/"
echo "Check the console output above for detailed metrics."
