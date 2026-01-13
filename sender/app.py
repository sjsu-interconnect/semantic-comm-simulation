import socket
import time
import random
import os
import torch
import torchvision
import torchvision.transforms as transforms
from utils.models import PretrainedMobileNetEncoder # Import custom Encoder
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
import requests
import json
from typing import Dict, Any, Tuple, List
from torch.utils.data import Dataset
import csv

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [Sender] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- CONFIGURATION CONSTANTS ---
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
# IMAGE_DIR = '/app/images' # Removed in favor of CIFAR-10
# IMAGE_FILENAMES = ['cat.jpeg', 'car.jpeg', 'dog.jpeg'] # Removed

# --- FEEDBACK CONFIG ---
FEEDBACK_HOST = '0.0.0.0'
FEEDBACK_PORT = 65500

# --- SIMULATED TIMES (Seconds) ---
SIM_ENC_SEMANTIC = 0.100
SIM_ENC_RAW = 0.005

# --- DRL Agent Config ---
# --- DRL Agent Config ---
ACTION_SEMANTIC_LOCAL = 0
ACTION_RAW = 1
ACTION_SEMANTIC_EDGE = 2

STATE_SPACE_LOW = np.array([0.0, 0.0, 50.0, 0.0, 1.0], dtype=np.float32)
STATE_SPACE_HIGH = np.array([100.0, 100.0, 2048.0, 0.5, 20.0], dtype=np.float32)


# Default to 100,000 if not set
TOTAL_TRAINING_STEPS = int(os.environ.get('EXPERIMENT_STEPS', 100000))
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
LEARNING_STARTS = 100
TRAIN_FREQUENCY = 1
TARGET_UPDATE_INTERVAL = 1000
LEARNING_RATE = 1e-3

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
        self.action_space = spaces.Discrete(3) # 0=SEMANTIC_LOCAL, 1=RAW, 2=SEMANTIC_EDGE
    def step(self, action):
        # Return valid (obs, reward, terminated, truncated, info)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, False, False, {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Set seed
        # Return (obs, info)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}


class CustomImageDataset(Dataset):
    """
    A custom dataset that loads images from a flat directory.
    Assumes filenames are like 'cat_01.jpg' where 'cat' is the label.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()

        # Scan directory
        if not os.path.exists(root_dir):
            logging.warning(f"Custom dataset directory {root_dir} does not exist.")
            return

        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, filename))
                # Derive label from filename (e.g., "cat_01.jpg" -> "cat")
                # Split by '_' or '.' and take the first part
                label = filename.split('_')[0].split('.')[0]
                self.labels.append(label)
                self.classes.add(label)
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_str = self.labels[idx]
        label_idx = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def load_dataset(dataset_type: str) -> Tuple[Any, List[str]]:
    """
    Factory function to load the requested dataset.
    Returns (dataset, classes_list).
    """
    logging.info(f"Loading dataset: {dataset_type}")
    
    if dataset_type == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        return dataset, dataset.classes
        
    elif dataset_type == 'STL10':
        # STL-10 images are 96x96
        dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=None)
        return dataset, dataset.classes
        
    elif dataset_type == 'CUSTOM':
        dataset = CustomImageDataset(root_dir='/app/images', transform=None)
        return dataset, dataset.classes
        
    else:
        logging.warning(f"Unknown dataset type '{dataset_type}'. Fallback to CIFAR10.")
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        return dataset, dataset.classes


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
        self.feedback_host = FEEDBACK_HOST
        self.feedback_port = FEEDBACK_PORT
        
        self.action_semantic_local = ACTION_SEMANTIC_LOCAL
        self.action_raw = ACTION_RAW
        self.action_semantic_edge = ACTION_SEMANTIC_EDGE
        
        self.state_low = STATE_SPACE_LOW
        self.state_high = STATE_SPACE_HIGH
        
        self.total_steps = TOTAL_TRAINING_STEPS
        self.batch_size = BATCH_SIZE
        self.learning_starts = LEARNING_STARTS
        
        # --- Exploration Config (Epsilon-Greedy) ---
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # --- Thread-safe queue for feedback ---
        self.feedback_queue = queue.Queue()
        
        # --- Aggregate Stats ---
        self.semantic_local_count = 0
        self.raw_count = 0
        self.semantic_edge_count = 0


        # --- Initialize Feature Extractor (Encoder) ---
        logging.info("Initializing Custom Encoder (MobileNetV3/Local)...")
        # Use MobileNetV3 for local processing
        self.feature_extractor = PretrainedMobileNetEncoder(encoded_space_dim=512)
        
        # Load pre-trained weights if available
        ae_weights_path = "/app/models/mobilenet_encoder.pth"
        if os.path.exists(ae_weights_path):
            logging.info(f"Loading MobileNet weights from {ae_weights_path}...")
            try:
                state_dict = torch.load(ae_weights_path, map_location='cpu')
                # No prefix issues this time, we save state_dict directly from model
                self.feature_extractor.load_state_dict(state_dict)
                logging.info("Encoder weights loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load encoder weights: {e}")
        else:
            logging.warning("No pre-trained weights found. Using random/imagenet weights.")

        # IMPORTANT: MobileNet expects 224x224
        self.feature_extractor.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), # Resize for MobileNet
            transforms.ToTensor(),
        ])

        # --- Initialize Dataset ---
        dataset_type = 'CIFAR10'
        self.dataset, self.classes = load_dataset(dataset_type)
        
        if len(self.dataset) == 0:
            logging.error("Dataset is empty! Please check configuration.")

        # --- Initialize DRL Agent ---
        logging.info("Initializing DRL Agent (DQN)...")
        self.env = DummyEnv()
        dummy_env = self.env
        
        # Check for existing model to load
        load_model_path = os.environ.get('LOAD_MODEL')
        if load_model_path and os.path.exists(f"/app/models/{load_model_path}.zip"):
            logging.info(f"Loading model from {load_model_path}...")
            self.model = DQN.load(f"/app/models/{load_model_path}", env=dummy_env)
            # Re-set buffer if needed, though load usually handles it. 
            # We need to ensure replay buffer is initialized if we want to continue training.
            if self.model.replay_buffer is None:
                 self.model.replay_buffer = ReplayBuffer(
                    self.model.buffer_size,
                    dummy_env.observation_space,
                    dummy_env.action_space,
                    self.model.device,
                    n_envs=1,
                    optimize_memory_usage=False
                )
        else:
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
                verbose=1,
                tensorboard_log="/app/runs/sender_logs"
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
        
        # FIX: Manually configure logger to avoid '_logger' or 'logger' attribute errors during manual train() calls
        from stable_baselines3.common.logger import configure
        new_logger = configure("/app/runs/sender_logs", ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)

    def _get_image_feature_vector(self, img: Image.Image) -> np.ndarray:
        """Extracts feature vector from a PIL Image using the Encoder."""
        # Ensure image is RGB
        img = img.convert('RGB')
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            # Encoder returns flattened vector
            vector = self.feature_extractor(batch_t)
            vector = vector.squeeze().numpy()
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

    def _get_edge_feature_vector(self, img: Image.Image) -> np.ndarray:
        """Extracts feature vector using the Edge Encoder service."""
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Send POST request
            response = requests.post(
                "http://edge-encoder:8000/encode",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")}
            )
            
            if response.status_code == 200:
                data = response.json()
                vector = np.array(data['vector'], dtype=np.float32)
                return vector
            else:
                logging.error(f"Edge Encoder failed: {response.text}")
                # Fallback to local encoder or random?
                # For now, let's fallback to local to avoid crash, but log error
                return self._get_image_feature_vector(img)
                
        except Exception as e:
            logging.error(f"Error calling Edge Encoder: {e}")
            return self._get_image_feature_vector(img)

    def _create_message_payload(self, action: int) -> Tuple[Dict[str, Any], str]:
        """
        Creates the message dictionary based on the agent's action.
        Returns the message dict and a string for logging.
        """
        # send_timestamp removed from here to simulate "ready-to-send" time

        
        # Pick a random image from the dataset
        if len(self.dataset) > 0:
            idx = random.randint(0, len(self.dataset) - 1)
            img, label_idx = self.dataset[idx] # img is PIL Image
            label = self.classes[label_idx]
        else:
            # Fallback if dataset is empty
            img = Image.new('RGB', (32, 32), color='red')
            label = "error"
        
        # Calculate Ground Truth Vector (for semantic loss calculation if needed)
        # BUT for Reconstruction, we need the GT IMAGE.
        # We send the GT Image as bytes in 'gt_image' field.
        
        # Convert PIL image to bytes for GT
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        gt_image_bytes = img_byte_arr.getvalue()

        # --- Capture "Ready to Send" Timestamp ---
        # We capture this *after* all encoding/processing to simulate the start of network transmission.
        # This allows us to simply add the SIM_ENC constant at the receiver for the total latency.
        send_timestamp = time.time()

        message_dict = {
            'ts': send_timestamp,
            'label': label,
            'gt_image': gt_image_bytes # Send GT Image for reconstruction loss
        }

        if action == self.action_semantic_local:
            self.semantic_local_count += 1
            message_dict['type'] = 'SEM_LOCAL'
            # Send the semantic vector (compressed)
            vector = self._get_image_feature_vector(img)
            message_dict['payload'] = vector
            log_msg_type = "SEM_LOCAL"

        elif action == self.action_semantic_edge:
            self.semantic_edge_count += 1
            message_dict['type'] = 'SEM_EDGE'
            # Send the semantic vector from Edge service
            vector = self._get_edge_feature_vector(img)
            message_dict['payload'] = vector
            log_msg_type = "SEM_EDGE"
            
        else:  # ACTION_RAW
            self.raw_count += 1
            message_dict['type'] = 'RAW'
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            
            # --- SIMULATION HACK: Pad RAW payload ---
            # To simulate high-res images (e.g. 50-150KB) so that Bandwidth actually matters.
            # 150KB padding guarantees failure at 1Mbps (~1.2s delay > 1.0s deadline).
            padding_size = 150 * 1024 # 150 KB
            dummy_padding = b'\x00' * padding_size
            
            message_dict['payload'] = img_byte_arr.getvalue() + dummy_padding
            log_msg_type = f"RAW (+{padding_size//1024}KB Pad)"
            
        return message_dict, log_msg_type

    def _get_normalized_state(self, local_state, noise, bandwidth) -> np.ndarray:
        """Combines and normalizes the state vector."""
        # 1. Concatenate
        network_state = np.array([noise, bandwidth], dtype=np.float32)
        full_state = np.concatenate((local_state, network_state))
        
        # 2. Normalize (Simple Min-Max scaling using known High bounds)
        # Avoid division by zero
        high = np.maximum(self.state_high, 1e-6)
        normalized_state = full_state / high
        
        # Clip to [0, 1] just in case
        normalized_state = np.clip(normalized_state, 0.0, 1.0)
        
        return normalized_state

    def run(self):
        """
        Connects to the channel and runs the main DRL training loop.
        """
        logging.info("Waiting 10s for other services to start...")
        time.sleep(10)
        
        # Initialize the 5D state
        local_state = self._get_local_state()
        # network_state = np.array([0.0, 10.0], dtype=np.float32)  # Initial guess
        # state = np.concatenate((local_state, network_state))
        
        # Use Normalized State
        state = self._get_normalized_state(local_state, 0.05, 10.0)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Connecting to channel at {self.channel_host}:{self.channel_port}...")
                s.connect((self.channel_host, self.channel_port))
                logging.info("Connected to channel. Starting DRL loop.")

                # --- Initialize CSV Logger ---
                csv_file_path = "/app/runs/results.csv"
                logging.info(f"Initializing results logger at {csv_file_path}")
                try:
                    # Create/Overwrite file and write header
                    with open(csv_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['step', 'timestamp', 'cpu', 'mem', 'data_size', 'action', 'reward', 'noise', 'bandwidth', 'epsilon'])
                except Exception as e:
                    logging.error(f"Failed to create CSV logger: {e}")


                for step in range(1, self.total_steps + 1):
                    
                    # 1. AGENT: Select Action
                    # Explicit Epsilon-Greedy Exploration
                    if random.random() < self.epsilon:
                        action = self.env.action_space.sample()
                        # logging.info(f"Exploration: Random Action {action} selected.")
                    else:
                        action, _ = self.model.predict(state, deterministic=False)
                    
                    # 2. SENDER: Create and send message
                    message_dict, log_msg_type = self._create_message_payload(action)
                    message_payload_bytes = pickle.dumps(message_dict)
                    
                    logging.info(f"Step {step}: State={np.round(state, 2)}")
                    logging.info(f"  -> Action Int: {action}, Epsilon: {self.epsilon:.4f}")
                    logging.info(f"  -> Action Type: {log_msg_type}")
                    self._send_message(s, message_payload_bytes)
                    
                    # 3. SENDER: Receive Feedback
                    try:
                        reward, noise, bandwidth = self.feedback_queue.get(timeout=60.0)
                    except queue.Empty:
                        logging.warning("Feedback timeout! Skipping step.")
                        continue
                    
                    # --- Log to CSV ---
                    try:
                        with open(csv_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            # state = [cpu, mem, data_size, last_noise, last_bw]
                            # We want CURRENT local state used for this action
                            # We have 'local_state' variable from before loop or update?
                            # Wait, state is [cpu, mem, data, net_noise, net_bw].
                            
                            # Let's extract from 'state' which was used for decision
                            s_cpu = state[0]
                            s_mem = state[1]
                            s_data = state[2]
                            
                            writer.writerow([step, time.time(), s_cpu, s_mem, s_data, log_msg_type, reward, noise, bandwidth, self.epsilon])
                    except Exception as e:
                        logging.error(f"Error logging to CSV: {e}")

                    # 4. AGENT: Construct next_state and learn
                    next_local_state = self._get_local_state()
                    next_state = self._get_normalized_state(next_local_state, noise, bandwidth)
                    
                    # Add experience to replay buffer
                    self.model.replay_buffer.add(state, next_state, action, reward, done=False, infos=[{}])
                    
                    # Update state for next loop
                    state = next_state
                    
                    # Train the agent
                    if step > self.learning_starts:
                        try:
                            self.model.train(gradient_steps=4)
                        except Exception as e:
                            logging.error(f"Error during model.train() at step {step}: {e}")
                            logging.error(f"Error during model.train() at step {step}: {e}")
                            logging.error("Continuing without training for this step...")
                    
                    # Decay Epsilon
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    
                    if step % 100 == 0:
                        total_sent = self.semantic_local_count + self.raw_count + self.semantic_edge_count
                        sem_local_pct = (self.semantic_local_count / total_sent) * 100 if total_sent > 0 else 0
                        sem_edge_pct = (self.semantic_edge_count / total_sent) * 100 if total_sent > 0 else 0
                        raw_pct = (self.raw_count / total_sent) * 100 if total_sent > 0 else 0
                        
                        logging.info(f"--- Step {step}, Last Reward: {reward:.3f} ---")
                        logging.info(f"--- Stats: Total={total_sent}, LOCAL={self.semantic_local_count} ({sem_local_pct:.1f}%), EDGE={self.semantic_edge_count} ({sem_edge_pct:.1f}%), RAW={self.raw_count} ({raw_pct:.1f}%) ---")
                        self.model.save(f"/app/models/drl_agent_checkpoint_{step}")

        except socket.error as e:
            logging.error(f"Socket error in main loop: {e}. Exiting.")
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}. Exiting.")
        finally:
            total_sent = self.semantic_local_count + self.raw_count + self.semantic_edge_count
            sem_local_pct = (self.semantic_local_count / total_sent) * 100 if total_sent > 0 else 0
            sem_edge_pct = (self.semantic_edge_count / total_sent) * 100 if total_sent > 0 else 0
            raw_pct = (self.raw_count / total_sent) * 100 if total_sent > 0 else 0
            
            logging.info("Training finished. Saving final model.")
            logging.info(f"=== FINAL STATS === Total Steps: {total_sent}")
            logging.info(f"=== LOCAL: {self.semantic_local_count} ({sem_local_pct:.1f}%)")
            logging.info(f"=== EDGE: {self.semantic_edge_count} ({sem_edge_pct:.1f}%)")
            logging.info(f"=== RAW: {self.raw_count} ({raw_pct:.1f}%)")
            self.model.save("/app/models/drl_agent_final")

if __name__ == "__main__":
    agent = SenderAgent()
    agent.start_feedback_listener()
    agent.run()