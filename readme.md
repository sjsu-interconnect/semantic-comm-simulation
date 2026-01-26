# Adaptive Semantic Communication with Deep Reinforcement Learning (Simulated)

This project is a high-fidelity Docker-based emulation of a dynamic semantic communication system. It explores how **Deep Reinforcement Learning (DRL)** can optimize data transmission by intelligently switching between **Local Compression** (TinyVAE), **Edge Offloading** (Stable Diffusion), and **Raw Transmission**.

The core is a **DQN Agent** (Sender) that learns to balance **Visual Quality** against **Latency** and **Bandwidth** in a fluctuating network environment.

---

## 🏛️ System Architecture

The simulation mimics a modern IoT-to-Edge pipeline with three distinct tiers of data processing:

```mermaid
graph TD
    subgraph Sender_Node [IoT Device / Sender]
        style Sender_Node fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
        Sender_Agent[DQN Agent]
        TinyVAE["TinyVAE Encoder<br/>(Local)"]
        
        Sender_Agent -- Selects Action --> Action_Decision{"Action?"}
        Action_Decision -- "SEM_LOCAL (0)" --> TinyVAE
        Action_Decision -- "SEM_EDGE (2)" --> Edge_Client[HTTP Client]
        Action_Decision -- "RAW (1)" --> Raw_Packager[Raw Packager]
        
        TinyVAE -- Latent Vector --> Msg_Packager[Message Packager]
        Edge_Client -- "Image (HTTP)" --> Edge_Service_Enc
        Raw_Packager -- JPEG Bytes --> Msg_Packager
    end

    subgraph Edge_Node [Edge Server]
        style Edge_Node fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
        Edge_Service_Enc["Edge Encoder<br/>(Stable Diffusion VAE)"]
        Edge_Service_Dec["Edge Decoder<br/>(Stable Diffusion VAE)"]
        
        Edge_Service_Enc -- "Latent Vector" --> Edge_Client
    end

    subgraph Channel_Sim [Dynamic Channel]
        style Channel_Sim fill:#fff3e0,stroke:#e65100,stroke-width:2px
        Channel_Logic[Dynamic Channel Logic]
        
        Msg_Packager == "Message" ==> Channel_Logic
        Channel_Logic -- "Adds Noise & Latency" --> Receiver_In
    end

    subgraph Receiver_Node [Receiver / Cloud]
        style Receiver_Node fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
        Receiver_In[Receiver Input]
        TinyDecoder[TinyVAE Decoder]
        
        Receiver_In -- "Local Vector" --> TinyDecoder
        Receiver_In -- "Edge Vector" --> Edge_Service_Dec
        Receiver_In -- "Raw" --> Feature_Ext[Image Processor]
        
        TinyDecoder -- "Reconstructed Img" --> Reward_Calc[Reward Calculator]
        Edge_Service_Dec -- "Reconstructed Img" --> Reward_Calc
        Feature_Ext -- "Ground Truth" --> Reward_Calc
    end

    %% Feedback Flow
    Reward_Calc -- "Reward + Net State" --> Sender_Agent
```

---

## ⚖️ The Trade-off (Agent Actions)

The agent chooses one of three actions for every image frame ($256 \times 256$):

| Action | Component | Description | Trade-offs |
| :--- | :--- | :--- | :--- |
| **0: SEM_LOCAL** | **TinyVAE (TAESD)** | Compresses image on the device (Sender). | ✅ **Fast Encoding** (Simulated Mobile CPU)<br>⚠️ **Medium Quality** (Slight compression artifacts)<br>✅ **Low Bandwidth** |
| **1: RAW** | **None** | Sends the original image. | ✅ **Perfect Quality**<br>❌ **Massive Bandwidth** (High latency if network is slow)<br>❌ **Vulnerable to Noise** |
| **2: SEM_EDGE** | **SD VAE** | Uploads image to Edge for high-quality compression. | ✅ **Excellent Quality** (Stable Diffusion VAE)<br>✅ **Fast Compute** (Simulated GPU)<br>❌ **Upload Latency** (Using bandwidth to upload) |

### Simulation Physics (Latencies)
*   **Local Compute**: `0.5s` (Simulating a slow mobile processor).
*   **Edge Compute**: `0.01s` (Simulating a powerful GPU cluster).
*   **Network**: Dynamic. Sending to Edge takes time proportional to Bandwidth.

**The "Winning" Strategy:**
*   **High Bandwidth**: Use **Edge**. (Fast Upload + Instant Compute + Great Quality).
*   **Low Bandwidth**: Use **Local**. (Upload takes too long; Local compute is slow but quicker than bad network).
*   **Very High Bandwidth + No Noise**: Use **Raw**.

---

## 🚀 How to Run

### Prerequisites
*   Docker & Docker Compose
*   (Optional) NVIDIA GPU for Edge containers (defaults to CPU, which is fine for simulation).

### Start the Simulation
From the root directory:

```bash
docker-compose up --build
```

**Note**: The first run will be slow because the **Sender** and **Edge** containers need to download the pretrained models (`madebyollin/taesd` and `CompVis/stable-diffusion-v1-4`) from Hugging Face.

### Watch it Learn 🧠
You will see logs from all services.
1.  **Exploration (Step 0-100)**: The agent acts randomly to fill its replay buffer.
2.  **Training (Step 101+)**: The agent starts learning.
    *   Watch the `Analysis` logs or check `runs/` for TensorBoard/Plots.
    *   Ideally, the agent converges to **Local** or **Edge** depending on the simulated channel conditions.

---

## 🛠️ Components

### 1. `Sender` (The Agent)
Hosted in `sender/`. Runs the PyTorch DRL agent (`stable-baselines3` DQN). It observes CPU load, memory, and channel feedback to pick an action.

### 2. `Edge Services`
Hosted in `edge_encoder/` and `edge_decoder/`. These are FastAPI microservices that host the heavy **Stable Diffusion VAE**.

### 3. `Channel`
Hosted in `channel/`. A simulated network pipe that injects Gaussian noise and delays packets based on their size and the current "Simulated Bandwidth" (which fluctuates randomly).

### 4. `Receiver`
Hosted in `receiver/`. Receives messages, decodes them (using TinyVAE or calling the Edge Decoder), compares the result to the Ground Truth (MSE Loss), and calculates the **Reward**.