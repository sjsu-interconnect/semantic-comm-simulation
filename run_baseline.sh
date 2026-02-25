#!/bin/bash

# Usage: ./run_baseline.sh [RAW|HEURISTIC|DRL|SEM_EDGE]

MODE=${1:-DRL} # Default to DRL if not specified

echo "Starting Simulation in BASELINE MODE: $MODE"

# Stop existing containers
docker-compose down

# Set the BASELINE environment variable and run
# We use 'sender' service specifically, but compose handles dependencies
BASELINE=$MODE docker-compose up --build -d

echo "Simulation started. Logs:"
docker-compose logs -f sender receiver
