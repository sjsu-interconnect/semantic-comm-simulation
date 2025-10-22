# Project Plan: Adaptive Semantic Communication with DRL

This document outlines the development phases and requirements for upgrading the existing semantic communication emulation with a Deep Reinforcement Learning (DRL) agent. The agent will live in the `Sender` and learn to select the optimal encoding strategy based on dynamic network and resource states.

---

## Phase 1: Establish the Two-Way Feedback Loop

**Objective:** To enable the `Receiver` to calculate performance (latency, semantic loss) and send a `reward` signal back to the `Sender`. This closes the loop, turning the one-way emulation into a measurable environment.

### Requirements

#### `Receiver` (receiver.py)
* **Performance Calculation:**
    * Must receive a `send_timestamp` and "ground truth" semantic vector (or an ID) from the sender.
    * Must calculate `total_latency = time.now() - send_timestamp`.
    * Must calculate `semantic_loss` by comparing the reconstructed vector with the ground truth vector (e.g., using MSE).
    * Must compute a final `reward` score based on latency and semantic loss.
* **Feedback Socket:**
    * Must open a *new* socket connection to the `Sender` (e.g., at `sender:65500`).
    * Must send the calculated `reward` and any `next_state` information (e.g., observed network conditions) back to the `Sender`.

#### `Sender` (sender.py)
* **Data Transmission:**
    * Must send its `send_timestamp` and the "ground truth" vector with every message.
* **Reward Socket:**
    * Must open a *second* socket to *listen* for the feedback message from the `Receiver` (e.g., on port `65500`).
    * The socket must be non-blocking or handled in a separate thread to avoid halting the main loop.

#### `docker-compose.yml`
* **Sender Service:**
    * Add an `expose` tag to the `sender` service to make the new reward port discoverable:
        ```yaml
        expose:
          - "65500"
        ```
    * Ensure the `receiver` `depends_on` the `sender`.

---

## Phase 2: Integrate the DRL Agent and Training Loop

**Objective:** To replace the `Sender`'s simple injector loop with a full DRL training loop. The agent will learn to select between two different encoding strategies ("Semantic" vs. "Raw") based on the environment state.

### Requirements

#### `Sender` (sender/Dockerfile)
* **New Libraries:**
    * Add `gymnasium` and `stable-baselines3` to `requirements.txt`.
    * Add `psutil` to `requirements.txt` for monitoring local resource usage.

#### `Sender` (sender.py)
* **DRL Imports:** Import `gymnasium`, `stable_baselines3`, and `psutil`.
* **Define Spaces:**
    * **Action Space:** `spaces.Discrete(2)`
        * `Action 0`: Semantic Transfer (send 512-vector).
        * `Action 1`: Raw Data Transfer (send `.jpeg` file).
    * **State Space:** `spaces.Box` (continuous) to represent:
        * `cpu_load`
        * `memory_available`
        * `data_size_to_send`
* **State Function:** Implement `get_current_state()` that uses `psutil` to get CPU/memory and simulates a new task (`data_size`).
* **Agent Initialization:** Instantiate a DRL agent (e.g., `DQN` or `PPO`) with the defined spaces.
* **Main Training Loop:** Replace the `while True:` loop with the DRL training sequence:
    1.  `state = get_current_state()`
    2.  `action, _ = model.predict(state)`
    3.  Create the message based on the `action`:
        * If `action == 0`: `message = b"SEM" | timestamp_bytes | label_bytes | vector_bytes`
        * If `action == 1`: `message = b"RAW" | timestamp_bytes | label_bytes | raw_image_file_bytes`
    4.  Send the `message` to the channel (with the 4-byte length header).
    5.  Wait and receive the `reward` from the feedback socket.
    6.  `next_state = get_current_state()`
    7.  Store the experience (`state`, `action`, `reward`, `next_state`) in the agent's replay buffer.
    8.  Call `model.train()` to perform a learning step.

#### `Channel` (channel.py)
* **Message-Aware Logic:** After receiving the full `data` payload:
    * Split the message: `type_bytes, rest_of_data = data.split(b'|', 1)`
    * **If `type_bytes == b"SEM":`**
        * Split `rest_of_data` to get the `vector_bytes`.
        * Apply noise: `noisy_vector_bytes = add_noise(vector_bytes)`.
        * Re-assemble the message to forward.
    * **If `type_bytes == b"RAW":`**
        * Do not apply noise.
        * Forward the original `data` message directly.

#### `Receiver` (receiver.py)
* **Message-Aware Logic:** After receiving the full `data` payload:
    * Split the message: `type_bytes, rest_of_data = data.split(b'|', 1)`
    * Extract `timestamp_bytes` and `original_label_bytes`.
    * **If `type_bytes == b"SEM":`**
        * Get the `noisy_vector_bytes`.
        * `reconstructed_vector = np.frombuffer(noisy_vector_bytes, ...)`
    * **If `type_bytes == b"RAW":`**
        * Get the `raw_image_file_bytes`.
        * Load into memory: `img = Image.open(io.BytesIO(raw_image_file_bytes))`
        * `reconstructed_vector = get_image_feature_vector(img)`
    * **Common Calculation:**
        * The *rest of the script* (calculating latency, semantic loss, reward, and sending feedback) will run identically, as it only needs the `reconstructed_vector`.

---

## Phase 3: Create the Main DRL Training Loop

**Objective:** To replace the `Sender`'s current `while True:` loop with a DRL training loop. This is where the agent actively makes decisions, interacts with the environment, and learns from the rewards.

### Requirements

#### `Sender` (sender.py)
* **Replace Main Loop:** The `while` loop must be converted into a `for step in range(TOTAL_TRAINING_STEPS):` loop.
* **The RL Loop Logic:** Each iteration of the loop must follow this exact sequence:
    1.  **Get State:** Get the current `state` from `get_current_state()`.
    2.  **Select Action:** Use `action, _ = model.predict(state)` to let the agent choose an encoding model.
    3.  **Perform Action:**
        * Execute the chosen encoding function based on the `action`.
        * Send the message (with timestamp/ground truth) to the `Channel`.
    4.  **Receive Feedback:**
        * Wait for the `reward` message from the `Receiver` on the reward socket.
        * Get the `next_state` from `get_current_state()`.
    5.  **Learn:**
        * Store this "experience" (`state`, `action`, `reward`, `next_state`) in the agent's replay buffer.
        * Call `model.train()` (or the equivalent) to update the agent's policy.
    6.  **Update State:** Set `state = next_state` for the next loop.

---

## Phase 4: Create a Dynamic Environment

**Objective:** To make the environment conditions change over time. The agent cannot learn a useful policy if the state is static (e.g., noise is always 0.1). This phase makes the problem realistic.

### Requirements

#### `Channel` (channel.py)
* **Variable State:**
    * Convert fixed variables like `NOISE_LEVEL` into dynamic class properties (e.g., `self.current_noise`).
    * Add a new property for `self.current_bandwidth` (e.g., in Mbps).
* **Simulate Transmission Delay:**
    * When data is received, calculate a `delay = data_size / self.current_bandwidth`.
    * Add `time.sleep(delay)` to simulate the transmission time.
* **Dynamic Update Function:**
    * Create a function `update_channel_state()` that runs in a separate thread (or periodically in the main loop).
    * This function should randomly change `self.current_noise` and `self.current_bandwidth` over time (e.g., using a random walk) to simulate fluctuating network conditions.

#### `Sender` (sender.py)
* **State Expansion:**
    * The state *must* now include the network conditions.
    * The `reward` message from the `Receiver` (Phase 1) must be updated to include the network state it experienced (e.g., `{'reward': r, 'noise': n, 'bandwidth': b}`).
    * The `State Space` definition must be expanded to include these new variables. The agent needs this information to learn the connection between network state and the best action.