import socket
import time
import random
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import threading
import struct
import io

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
import psutil
import queue

# --- CONFIGURATION ---
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
IMAGE_DIR = '/app/images'
IMAGE_FILENAMES = ['cat.jpeg', 'car.jpeg', 'dog.jpeg']

# --- FEEDBACK CONFIG ---
FEEDBACK_HOST = '0.0.0.0'
FEEDBACK_PORT = 65500

# --- NEW IN PHASE 3: DRL Agent Config ---
ACTION_SEMANTIC = 0
ACTION_RAW = 1

# --- SIMULATED TIMES (Seconds) ---
SIM_ENC_SEMANTIC = 0.100
SIM_ENC_RAW = 0.005

# State: [cpu, mem, data_size, last_noise, last_bandwidth]
STATE_SPACE_LOW = np.array([0.0, 0.0, 50.0, 0.0, 1.0], dtype=np.float32)
STATE_SPACE_HIGH = np.array([100.0, 100.0, 2048.0, 0.5, 20.0], dtype=np.float32)

TOTAL_TRAINING_STEPS = 100
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
LEARNING_STARTS = 100 
TRAIN_FREQUENCY = 4

# Feedback queue now expects a tuple: (reward, noise, bandwidth)
feedback_queue = queue.Queue()

# --- MODEL SETUP ---
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
preprocess = weights.transforms()

# --- Dummy env to hold spaces ---
class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        # --- NEW IN PHASE 3: 5-dimensional state space ---
        self.observation_space = spaces.Box(
            low=STATE_SPACE_LOW, 
            high=STATE_SPACE_HIGH, 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
    def step(self, action): pass
    def reset(self, seed=None, options=None): pass


def get_image_feature_vector(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
        vector = features.squeeze().numpy()
    return vector

# --- REWARD LISTENER ---
def reward_listener_thread():
    """
    Listens for feedback (reward, noise, bw) and puts it in the queue.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((FEEDBACK_HOST, FEEDBACK_PORT))
        s.listen()
        print(f"Sender feedback listener started on port {FEEDBACK_PORT}...")
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    # --- NEW IN PHASE 3: Receive 3 floats (12 bytes) ---
                    data = conn.recv(12) 
                    if not data or len(data) != 12:
                        continue
                    
                    # Unpack the 3-float array
                    feedback_data = np.frombuffer(data, dtype=np.float32)
                    reward = feedback_data[0]
                    noise = feedback_data[1]
                    bandwidth = feedback_data[2]
                    
                    # Put the full tuple into the queue
                    feedback_queue.put((reward, noise, bandwidth))
                    
            except Exception as e:
                print(f"Error in feedback listener: {e}")

# --- DRL Helper Functions ---
def get_local_state():
    """
    Gets the sender's local resource state and a simulated task.
    """
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    data_size = np.random.randint(STATE_SPACE_LOW[2], STATE_SPACE_HIGH[2])
    
    return np.array([cpu, mem, data_size], dtype=np.float32)

def send_message(sock, message_payload):
    msg_len_header = struct.pack('!I', len(message_payload))
    sock.sendall(msg_len_header)
    sock.sendall(message_payload)

# --- MAIN SENDER DRL LOOP ---
print("Sender is starting...")

listener_thread = threading.Thread(target=reward_listener_thread, daemon=True)
listener_thread.start()

# --- Initialize DRL Agent ---
dummy_env = DummyEnv()

model = DQN(
    "MlpPolicy",
    dummy_env,
    learning_rate=1e-4,
    buffer_size=REPLAY_BUFFER_SIZE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    train_freq=TRAIN_FREQUENCY,
    gradient_steps=-1,
    target_update_interval=1000,
    verbose=1
)

model.replay_buffer = ReplayBuffer(
    model.buffer_size,
    dummy_env.observation_space,
    dummy_env.action_space,
    model.device,
    n_envs=1,
    optimize_memory_usage=False
)

time.sleep(10) # Give other services time to start up

# --- Main Training Loop ---
print("--- Starting DRL Training Loop ---")

# --- NEW IN PHASE 3: Initialize state components ---
local_state = get_local_state()
# Initial (dummy) network state: no noise, 10 Mbps bandwidth
network_state = np.array([0.0, 10.0], dtype=np.float32) 
state = np.concatenate((local_state, network_state))


try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Sender trying to connect to channel...")
        s.connect((CHANNEL_HOST, CHANNEL_PORT))
        print("Sender connected to channel. Starting DRL loop.")

        for step in range(1, TOTAL_TRAINING_STEPS + 1):
            
            # 1. AGENT: Select Action based on current state
            action, _ = model.predict(state, deterministic=False)
            
            # 2. SENDER: Perform Action
            # removed send_timestamp here
            
            image_name = random.choice(IMAGE_FILENAMES)
            image_path = os.path.join(IMAGE_DIR, image_name)
            label = image_name.split('.')[0]
            
            # Moved timestamp prep to after encoding
            label_bytes = label.encode('utf-8')

            if action == ACTION_SEMANTIC:
                vector = get_image_feature_vector(image_path)
                payload = vector.tobytes()
                type_bytes = b"SEM"
                log_msg_type = "SEMANTIC"
                
            else: # ACTION_RAW
                with open(image_path, 'rb') as f:
                    payload = f.read()
                type_bytes = b"RAW"
                log_msg_type = "RAW"

            # --- Capture "Ready to Send" Timestamp ---
            # This effectively excludes the encoding time from the network latency calculation at the receiver,
            # allowing us to add the fixed SIM_ENC constant instead.
            send_timestamp = time.time()
            timestamp_bytes = np.array([send_timestamp], dtype=np.float64).tobytes()

            message = type_bytes + b'|' + timestamp_bytes + b'|' + label_bytes + b'|' + payload
            
            print(f"Step {step}: State={np.round(state, 2)}")
            print(f"  -> Action: {log_msg_type}")
            send_message(s, message)
            
            # 3. ENVIRONMENT: (Receiver/Channel process the action)
            
            # 4. SENDER: Receive Feedback
            try:
                # --- NEW IN PHASE 3: Get full feedback package ---
                reward, noise, bandwidth = feedback_queue.get(timeout=20.0) # Increased timeout
            except queue.Empty:
                print("Feedback timeout! Skipping step.")
                continue

            # 5. AGENT: Learn
            # --- NEW IN PHASE 3: Construct next_state ---
            next_local_state = get_local_state()
            next_network_state = np.array([noise, bandwidth], dtype=np.float32)
            next_state = np.concatenate((next_local_state, next_network_state))
            
            # Add experience to replay buffer
            model.replay_buffer.add(state, next_state, action, reward, done=False, infos=[{}])
            
            # Update state for next loop
            state = next_state
            
            # Train the agent
            if step > LEARNING_STARTS:
                model.train(gradient_steps=1)
            
            if step % 100 == 0:
                print(f"--- Step {step}, Last Reward: {reward:.3f} ---")
                model.save("drl_agent_checkpoint")

except Exception as e:
    print(f"An error occurred in the main loop: {e}. Exiting.")
finally:
    print("Saving final model.")
    model.save("drl_agent_final")