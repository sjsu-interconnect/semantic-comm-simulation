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
import pickle
import logging
from typing import Dict, Any, Tuple

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [Sender] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- CONFIGURATION CONSTANTS ---
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
IMAGE_DIR = '/app/images'
IMAGE_FILENAMES = ['cat.jpeg', 'car.jpeg', 'dog.jpeg']

# --- FEEDBACK CONFIG ---
FEEDBACK_HOST = '0.0.0.0'
FEEDBACK_PORT = 65500

# --- DRL Agent Config ---
ACTION_SEMANTIC = 0
ACTION_RAW = 1
STATE_SPACE_LOW = np.array([0.0, 0.0, 50.0, 0.0, 1.0], dtype=np.float32)
STATE_SPACE_HIGH = np.array([100.0, 100.0, 2048.0, 0.5, 20.0], dtype=np.float32)

TOTAL_TRAINING_STEPS = 50000
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
LEARNING_STARTS = 100
TRAIN_FREQUENCY = 4
TARGET_UPDATE_INTERVAL = 1000
LEARNING_RATE = 1e-4

# --- Dummy env to hold spaces (required by SB3) ---
class DummyEnv(gym.Env):
    """A minimal gym.Env to hold the observation and action spaces."""
    def __init__(self):
        super(DummyEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=STATE_SPACE_LOW,
            high=STATE_SPACE_HIGH,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2) # 0=SEMANTIC, 1=RAW
    def step(self, action): pass
    def reset(self, seed=None, options=None): pass


class SenderAgent:
    """
    The main DRL Agent class.
    
    This class initializes the DRL model, feature extractors, and network
    connections, and then runs the main training loop.
    """
    def __init__(self):
        # --- Store configuration ---
        self.channel_host = CHANNEL_HOST
        self.channel_port = CHANNEL_PORT
        self.image_dir = IMAGE_DIR
        self.image_filenames = IMAGE_FILENAMES
        self.feedback_host = FEEDBACK_HOST
        self.feedback_port = FEEDBACK_PORT
        
        self.action_semantic = ACTION_SEMANTIC
        self.action_raw = ACTION_RAW
        
        self.state_low = STATE_SPACE_LOW
        self.state_high = STATE_SPACE_HIGH
        
        self.total_steps = TOTAL_TRAINING_STEPS
        self.batch_size = BATCH_SIZE
        self.learning_starts = LEARNING_STARTS
        
        # --- Thread-safe queue for feedback ---
        self.feedback_queue = queue.Queue()

        # --- Initialize Feature Extractor ---
        logging.info("Initializing ResNet-18 feature extractor...")
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor.eval()
        self.preprocess = weights.transforms()

        # --- Initialize DRL Agent ---
        logging.info("Initializing DRL Agent (DQN)...")
        dummy_env = DummyEnv()
        self.model = DQN(
            "MlpPolicy",
            dummy_env,
            learning_rate=LEARNING_RATE,
            buffer_size=REPLAY_BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            train_freq=TRAIN_FREQUENCY,
            gradient_steps=-1, # We train manually
            target_update_interval=TARGET_UPDATE_INTERVAL,
            verbose=0 # Set to 1 for more SB3 output
        )
        # Initialize the replay buffer
        self.model.replay_buffer = ReplayBuffer(
            self.model.buffer_size,
            dummy_env.observation_space,
            dummy_env.action_space,
            self.model.device,
            n_envs=1,
            optimize_memory_usage=False
        )

    def _get_image_feature_vector(self, image_path: str) -> np.ndarray:
        """Loads an image from path and extracts its feature vector."""
        img = Image.open(image_path).convert('RGB')
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            features = self.feature_extractor(batch_t)
            vector = features.squeeze().numpy()
        return vector

    def _reward_listener_thread(self):
        """
        Listens for feedback (reward, noise, bw) and puts it in the queue.
        Runs in a separate, daemonic thread.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.feedback_host, self.feedback_port))
            s.listen()
            logging.info(f"Feedback listener started on port {self.feedback_port}...")
            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(12)  # 3 floats * 4 bytes/float
                        if not data or len(data) != 12:
                            logging.warning(f"[Feedback] Incomplete data (got {len(data)} bytes).")
                            continue
                        
                        feedback_data = np.frombuffer(data, dtype=np.float32)
                        reward, noise, bandwidth = feedback_data
                        
                        # Put the full tuple into the queue
                        self.feedback_queue.put((reward, noise, bandwidth))
                        
                except Exception as e:
                    logging.error(f"Error in feedback listener: {e}")

    def start_feedback_listener(self):
        """Starts the feedback listener thread."""
        listener_thread = threading.Thread(target=self._reward_listener_thread, daemon=True)
        listener_thread.start()

    def _get_local_state(self) -> np.ndarray:
        """Gets the sender's local resource state and a simulated task size."""
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        data_size = np.random.uniform(self.state_low[2], self.state_high[2])
        return np.array([cpu, mem, data_size], dtype=np.float32)

    def _send_message(self, sock: socket.socket, message_payload: bytes):
        """Frames and sends a message (header + payload)."""
        msg_len_header = struct.pack('!I', len(message_payload))
        sock.sendall(msg_len_header)
        sock.sendall(message_payload)

    def _create_message_payload(self, action: int) -> Tuple[Dict[str, Any], str]:
        """
        Creates the message dictionary based on the agent's action.
        Returns the message dict and a string for logging.
        """
        send_timestamp = time.time()
        image_name = random.choice(self.image_filenames)
        image_path = os.path.join(self.image_dir, image_name)
        label = image_name.split('.')[0]
        
        message_dict = {
            'ts': send_timestamp,
            'label': label
        }

        if action == self.action_semantic:
            message_dict['type'] = 'SEM'
            message_dict['payload'] = self._get_image_feature_vector(image_path)
            log_msg_type = "SEMANTIC"
            
        else:  # ACTION_RAW
            message_dict['type'] = 'RAW'
            with open(image_path, 'rb') as f:
                message_dict['payload'] = f.read()
            log_msg_type = "RAW"
            
        return message_dict, log_msg_type

    def run(self):
        """
        Connects to the channel and runs the main DRL training loop.
        """
        logging.info("Waiting 10s for other services to start...")
        time.sleep(10)
        
        # Initialize the 5D state
        local_state = self._get_local_state()
        network_state = np.array([0.0, 10.0], dtype=np.float32)  # Initial guess
        state = np.concatenate((local_state, network_state))

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Connecting to channel at {self.channel_host}:{self.channel_port}...")
                s.connect((self.channel_host, self.channel_port))
                logging.info("Connected to channel. Starting DRL loop.")

                for step in range(1, self.total_steps + 1):
                    
                    # 1. AGENT: Select Action
                    action, _ = self.model.predict(state, deterministic=False)
                    
                    # 2. SENDER: Create and send message
                    message_dict, log_msg_type = self._create_message_payload(action)
                    message_payload_bytes = pickle.dumps(message_dict)
                    
                    logging.info(f"Step {step}: State={np.round(state, 2)}")
                    logging.info(f"  -> Action: {log_msg_type}")
                    self._send_message(s, message_payload_bytes)
                    
                    # 3. SENDER: Receive Feedback
                    try:
                        reward, noise, bandwidth = self.feedback_queue.get(timeout=20.0)
                    except queue.Empty:
                        logging.warning("Feedback timeout! Skipping step.")
                        continue

                    # 4. AGENT: Construct next_state and learn
                    next_local_state = self._get_local_state()
                    next_network_state = np.array([noise, bandwidth], dtype=np.float32)
                    next_state = np.concatenate((next_local_state, next_network_state))
                    
                    # Add experience to replay buffer
                    self.model.replay_buffer.add(state, next_state, action, reward, done=False, infos=[{}])
                    
                    # Update state for next loop
                    state = next_state
                    
                    # Train the agent
                    if step > self.learning_starts:
                        self.model.train(gradient_steps=1)
                    
                    if step % 100 == 0:
                        logging.info(f"--- Step {step}, Last Reward: {reward:.3f} ---")
                        self.model.save("drl_agent_checkpoint")

        except socket.error as e:
            logging.error(f"Socket error in main loop: {e}. Exiting.")
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}. Exiting.")
        finally:
            logging.info("Training finished. Saving final model.")
            self.model.save("drl_agent_final")

if __name__ == "__main__":
    agent = SenderAgent()
    agent.start_feedback_listener()
    agent.run()