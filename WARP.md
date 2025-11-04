# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a semantic communication emulation system with Deep Reinforcement Learning (DRL). The system simulates a communication pipeline where:
- **Sender** extracts semantic features from images using ResNet18 and sends them over a network
- **Channel** acts as a noisy communication medium, adding Gaussian noise to transmitted data
- **Receiver** reconstructs semantic meaning from noisy vectors and sends reward feedback to the sender

The architecture implements a **two-way feedback loop** where the receiver calculates performance metrics (latency, semantic loss) and sends rewards back to the sender for reinforcement learning.

## Commands

### Running the System
```bash
# Build and start all services
docker-compose up --build

# Start services in detached mode
docker-compose up -d

# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f sender
docker-compose logs -f channel
docker-compose logs -f receiver

# Stop all services
docker-compose down
```

### Development
```bash
# Rebuild a specific service after code changes
docker-compose up --build sender
docker-compose up --build channel
docker-compose up --build receiver

# Enter a running container for debugging
docker exec -it <container_name> /bin/bash
```

## Architecture

### Three-Service Pipeline
1. **Sender** (`sender/app.py`)
   - Extracts 512-dimensional feature vectors from images using ResNet18 (pretrained)
   - Sends structured messages: `timestamp | label | vector`
   - Listens for reward feedback on port 65500 (separate thread)
   - Uses message framing with 4-byte header (`struct.pack('!I', len(message))`)
   
2. **Channel** (`channel/app.py`)
   - Listens on port 65431 for sender connections
   - Adds Gaussian noise (NOISE_LEVEL = 0.1) to semantic vectors
   - Forwards noisy data to receiver
   - Uses `recv_all()` helper to ensure complete message reception
   
3. **Receiver** (`receiver/app.py`)
   - Maintains a knowledge base of ideal feature vectors for known classes (cat, car, dog)
   - Receives noisy vectors and decodes semantic meaning via cosine similarity
   - Calculates performance metrics:
     - **Latency**: `reception_time - send_timestamp`
     - **Semantic Loss**: MSE between ground truth and noisy vector
     - **Reward**: `-1 * (latency_penalty + α * semantic_loss)` where α = 0.5
   - Sends reward feedback to sender's port 65500

### Message Protocol
All inter-service communication uses TCP sockets with a framing protocol:
- **Header**: 4-byte unsigned integer (`struct.pack('!I', length)`) indicating message size
- **Payload**: `timestamp_bytes | label_bytes | vector_bytes` delimited by `|`
- Timestamp: float64 (8 bytes)
- Label: UTF-8 encoded string
- Vector: float32 array (512 elements = 2048 bytes)

### Feedback Loop
The system implements Phase 1 of the DRL roadmap (see `project_plan.md`):
- Sender sends ground truth (label) with each message
- Receiver calculates reward based on latency deadline (τ = 1.0s) and semantic preservation
- Reward is sent back to sender via separate TCP connection
- Currently logs rewards; future phases will integrate actual DRL agent (gymnasium + stable-baselines3)

## Development Notes

### Adding New Image Classes
1. Add JPEG to `images/` directory
2. Add filename to `IMAGE_FILENAMES` in `sender/app.py`
3. Add class name to `CLASSES` in `receiver/app.py`
4. Rebuild containers

### Modifying Noise Characteristics
Edit `NOISE_LEVEL` in `channel/app.py` (currently 0.1 for Gaussian noise standard deviation)

### Future DRL Integration (Phases 2-4)
The `project_plan.md` outlines:
- Phase 2: Integrate gymnasium and stable-baselines3 into sender
- Phase 3: Replace main loop with RL training loop
- Phase 4: Make channel conditions dynamic (variable bandwidth, noise)

### Image Requirements
- Format: JPEG
- Preprocessing: ResNet18 default transforms (224x224 RGB)
- Sender uses images from `/app/images` (volume-mounted)
- Receiver builds knowledge base from `/app/images` (copied during build)
