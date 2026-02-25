#!/bin/bash

# Usage: ./run_apptainer.sh [RAW|HEURISTIC|DRL|SEM_EDGE]
MODE=${1:-DRL} # Default to DRL if not specified
export BASELINE=$MODE
export EXPERIMENT_STEPS=${EXPERIMENT_STEPS:-5000}

echo "Starting Simulation in BASELINE MODE: $BASELINE"

echo "Starting Edge Services..."
apptainer run --bind ./edge_encoder:/app,./models:/app/models,./sender/utils:/app/utils edge_encoder.sif &
P1=$!
apptainer run --bind ./edge_decoder:/app,./models:/app/models,./sender/utils:/app/utils edge_decoder.sif &
P2=$!

sleep 2

echo "Starting Channel..."
# Channel requires an isolated network namespace (--net) so `tc` doesn't affect the host
apptainer run --net --bind ./channel:/app channel.sif &
P3=$!

sleep 2

echo "Starting Receiver..."
apptainer run --bind ./receiver:/app,./runs:/app/runs,./models:/app/models,./sender/utils:/app/utils receiver.sif &
P4=$!

sleep 2

echo "Starting Sender..."
apptainer run --bind ./sender:/app,./runs:/app/runs,./models:/app/models,./sender/utils:/app/utils \
  --env EXPERIMENT_STEPS=$EXPERIMENT_STEPS \
  --env BASELINE=$BASELINE \
  sender.sif &
P5=$!

echo "Simulation running. Press Ctrl+C to stop."

cleanup() {
    echo "Stopping simulation..."
    kill $P1 $P2 $P3 $P4 $P5 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

wait $P5
cleanup
