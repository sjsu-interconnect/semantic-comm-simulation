import socket
import time
import random
import os
import torch
import torchvision
import torchvision
import torchvision.transforms as transforms
from utils.models import TinyVAEWrapper # Local is now TinyVAE
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

ACTION_RAW = 1
ACTION_SEMANTIC_EDGE = 2

# VAE Latent: 4 * 32 * 32 = 4096 float32s. 
# Data Size High = 4096 * 4 bytes / 1024 = 16 KB.
# Let's adjust state space high for data size to 20KB.
STATE_SPACE_LOW = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
STATE_SPACE_HIGH = np.array([100.0, 100.0, 20.0, 0.5, 20.0], dtype=np.float32)



# Default to 100,000 if not set
TOTAL_TRAINING_STEPS = int(os.environ.get('EXPERIMENT_STEPS', 100000))
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
LEARNING_STARTS = 100
TRAIN_FREQUENCY = 1
TARGET_UPDATE_INTERVAL = 1000
LEARNING_RATE = 1e-4 # Reduced from 1e-3 for stability

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
        self.epsilon_decay = 0.9995 # Slower decay for better exploration
        
        # --- Thread-safe queue for feedback ---
        self.feedback_queue = queue.Queue()
        
        # --- Aggregate Stats ---
        self.semantic_local_count = 0
        self.raw_count = 0
        self.semantic_edge_count = 0

        # --- Baseline Mode Config ---
        self.baseline_mode = os.environ.get('BASELINE', 'DRL').upper() # 'DRL', 'RAW', 'HEURISTIC', 'SEM_EDGE'
        if self.baseline_mode not in ['DRL', 'RAW', 'HEURISTIC', 'SEM_EDGE']:
             logging.warning(f"Unknown BASELINE mode '{self.baseline_mode}'. Defaulting to DRL.")
             self.baseline_mode = 'DRL'
        logging.info(f"Running in Mode: {self.baseline_mode}")


        # --- Initialize Feature Extractor (Encoder) ---
        logging.info("Initializing TinyVAE Encoder (Local/Fast)...")
        # Use TinyVAE for local processing (Fast, 256x256)
        self.feature_extractor = TinyVAEWrapper(device='cpu')
        
        # Preprocessing for TinyVAE (Resize to 256x256)
        self.feature_extractor.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)), 
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
                    n_envs=1,
                    optimize_memory_usage=False
                )
            # Load metadata (epsilon, steps)
            self._load_metadata(load_model_path)
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
        

        # --- Generate Unique Run ID ---
        import datetime
        self.run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = f"/app/runs/{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True)
        logging.info(f"Initialized Run Directory: {self.run_dir}")
        
        # --- Configure Loggers ---
        from stable_baselines3.common.logger import configure
        new_logger = configure(f"{self.run_dir}/sender_logs", ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)
    def _save_metadata(self, filename: str, step: int):
        """Saves agent metadata (epsilon, step) to a sidecar JSON file."""
        metadata = {
            'step': step,
            'epsilon': self.epsilon
        }
        try:
            with open(f"{filename}.json", 'w') as f:
                json.dump(metadata, f)
            logging.info(f"Saved metadata to {filename}.json")
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")

    def _load_metadata(self, model_name: str):
        """Loads agent metadata from sidecar JSON if it exists."""
        meta_path = f"/app/models/{model_name}.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore epsilon
                if 'epsilon' in metadata:
                    self.epsilon = float(metadata['epsilon'])
                    logging.info(f"Restored Epsilon: {self.epsilon:.4f}")
                
                # NOTE: We generally don't restore 'step' to overwrite self.total_steps loop,
                # because we often want to run *more* steps. But we could use it for logging offset.
            except Exception as e:
                logging.error(f"Failed to load metadata from {meta_path}: {e}")
        else:
            logging.warning(f"No metadata file found at {meta_path}. Using default/e_min parameters.")
            # If loading a model but no metadata, assume we are continuing training, so drop epsilon
            if self.epsilon == 1.0: 
                 self.epsilon = 0.5 # Conservative start if file missing but model loaded


    def _get_image_feature_vector(self, img: Image.Image) -> np.ndarray:
        """Extracts latent tensor from a PIL Image using TinyVAE."""
        # Ensure image is RGB
        img = img.convert('RGB')
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            # Encoder returns latent tensor [1, 4, 32, 32]
            latent = self.feature_extractor.encode(batch_t)
            if latent is not None:
                # TinyVAE latent usually matches SD scale, but TAESD native output might differ slightly 
                # depending on exact weights. Assuming madebyollin/taesd matches SD.
                vector = latent.squeeze(0).cpu().numpy() # [4, 32, 32]
            else:
                vector = np.zeros((4, 32, 32), dtype=np.float32)
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
                        # Expecting 5 floats (20 bytes): [reward, noise, bandwidth, latency, mse]
                        # But handle legacy 12 bytes gracefully if possible? No, we updated Receiver.
                        data = conn.recv(20) 
                        
                        if not data:
                            continue
                            
                        if len(data) == 20:
                            feedback_data = np.frombuffer(data, dtype=np.float32)
                            reward, noise, bandwidth, latency, mse = feedback_data
                            self.feedback_queue.put((reward, noise, bandwidth, latency, mse))
                        elif len(data) == 12:
                            feedback_data = np.frombuffer(data, dtype=np.float32)
                            reward, noise, bandwidth = feedback_data
                            self.feedback_queue.put((reward, noise, bandwidth, 0.0, 0.0))
                        else:
                            logging.warning(f"[Feedback] Incomplete data (got {len(data)} bytes).")
                        
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
                # Expecting list or base64? New logic returns 'vector' (latent)
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

        message_dict = {
            # 'ts': assigned at end of function
            'label': label,
            'gt_image': gt_image_bytes # Send GT Image for reconstruction loss
        }

        if action == self.action_semantic_local:
            self.semantic_local_count += 1
            message_dict['type'] = 'SEM_LOCAL'
            # Send the semantic vector (compressed)
            start_encode = time.time()
            vector = self._get_image_feature_vector(img)
            encode_time = time.time() - start_encode
            message_dict['encode_time'] = encode_time
            message_dict['payload'] = vector
            log_msg_type = "SEM_LOCAL"

        elif action == self.action_semantic_edge:
            self.semantic_edge_count += 1
            message_dict['type'] = 'SEM_EDGE'
            # Send the semantic vector from Edge service
            # We assume edge GPU is unconstrained and ultra-fast
            vector = self._get_edge_feature_vector(img)
            message_dict['encode_time'] = 0.010
            message_dict['payload'] = vector
            log_msg_type = "SEM_EDGE"
            
        else:  # ACTION_RAW
            self.raw_count += 1
            message_dict['type'] = 'RAW'
            
            start_encode = time.time()
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            
            # --- SIMULATION HACK: Pad RAW payload ---
            # To simulate high-res images (e.g. 50-150KB) so that Bandwidth actually matters.
            # 150KB padding guarantees failure at 1Mbps (~1.2s delay > 1.0s deadline).
            padding_size = 150 * 1024 # 150 KB
            dummy_padding = b'\x00' * padding_size
            
            payload_bytes = img_byte_arr.getvalue() + dummy_padding
            encode_time = time.time() - start_encode
            message_dict['encode_time'] = encode_time
            message_dict['payload'] = payload_bytes
            log_msg_type = f"RAW (+{padding_size//1024}KB Pad)"
            
        # --- Update Timestamp JUST BEFORE RETURN --- 
        # This ensures we don't count the local/edge processing time (overhead) in the network latency.
        # The receiver will add the simulated encoding time (SIM_ENC_*) explicitly.
        message_dict['ts'] = time.time()
            
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
                # Retry logic for connection
                connected = False
                for i in range(10):
                    try:
                        logging.info(f"Connecting to channel at {self.channel_host}:{self.channel_port} (Attempt {i+1}/10)...")
                        s.connect((self.channel_host, self.channel_port))
                        connected = True
                        break
                    except socket.error as e:
                        logging.warning(f"Connection failed: {e}. Retrying in 2 seconds...")
                        time.sleep(2)
                
                if not connected:
                    logging.error("Could not connect to channel after 10 attempts. Exiting.")
                    return

                logging.info("Connected to channel. Starting DRL loop.")

                # --- Initialize CSV Logger ---
                csv_file_path = f"{self.run_dir}/results.csv"
                logging.info(f"Initializing results logger at {csv_file_path}")
                try:
                    # Create/Overwrite file and write header
                    with open(csv_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['step', 'timestamp', 'cpu', 'mem', 'data_size', 'action', 'reward', 'noise', 'bandwidth', 'epsilon', 'latency', 'mse'])
                except Exception as e:
                    logging.error(f"Failed to create CSV logger: {e}")


                for step in range(1, self.total_steps + 1):
                    
                    # 1. AGENT: Select Action
                    
                    if self.baseline_mode == 'RAW':
                        # BASELINE: Always send RAW
                        action = self.action_raw
                        # logging.info("Baseline: Forcing RAW action.")
                        
                    elif self.baseline_mode == 'SEM_EDGE':
                        # BASELINE: Always send SEM_EDGE
                        action = self.action_semantic_edge
                        # logging.info("Baseline: Forcing SEM_EDGE action.")
                        
                    elif self.baseline_mode == 'HEURISTIC':
                        # BASELINE: Heuristic Rules
                        # Use last_bandwidth (simulating observance)
                        bw = self.last_bandwidth
                        
                        if bw < 2.0:
                            action = self.action_semantic_local # Survival Mode
                        elif bw < 8.0:
                            action = self.action_semantic_edge # Quality Trade-off
                        else:
                            action = self.action_raw # High Quality
                            
                        # logging.info(f"Baseline Heuristic (BW={bw:.2f}): Selected {action}")

                    else:
                        # STANDARD DRL MODE
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
                        # Expecting: [reward, noise, bandwidth, latency, mse]
                        feedback_data = self.feedback_queue.get(timeout=60.0)
                        if len(feedback_data) == 5:
                            reward, noise, bandwidth, latency, mse = feedback_data
                        elif len(feedback_data) == 3:
                             # Backward compatibility
                             reward, noise, bandwidth = feedback_data
                             latency, mse = 0.0, 0.0
                        else:
                             logging.error(f"Invalid feedback length: {len(feedback_data)}")
                             continue
                             
                    except queue.Empty:
                        logging.warning("Feedback timeout! Skipping step.")
                        continue
                        
                    # --- Log to CSV ---
                    try:
                        with open(csv_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            # Columns: step, timestamp, cpu, mem, data_size, action, reward, noise, bandwidth, epsilon, latency, mse
                            data_size_kb = len(message_payload_bytes) / 1024.0
                            writer.writerow([step, time.time(), 0.0, 0.0, data_size_kb, log_msg_type, reward, noise, bandwidth, self.epsilon, latency, mse])
                            f.flush() # Force flush
                            # print(f"DEBUG: Wrote step {step} to CSV")
                    except Exception as e:
                        logging.error(f"CSV append failed: {e}")
                        print(f"DEBUG: CSV append exception: {e}")
                    


                    # 4. AGENT: Construct next_state and learn
                    next_local_state = self._get_local_state()
                    next_state = self._get_normalized_state(next_local_state, noise, bandwidth)
                    
                    # Add experience to replay buffer
                    # Ensure action/reward/done are arrays for SB3 ReplayBuffer
                    action_arr = np.array([action])
                    reward_arr = np.array([reward], dtype=np.float32)
                    done_arr = np.array([False])
                    
                    self.model.replay_buffer.add(state, next_state, action_arr, reward_arr, done_arr, infos=[{}])
                    
                    # Update state for next loop
                    state = next_state
                    
                    # Train the agent
                    if step > self.learning_starts:
                        try:
                            # Gradient Steps = 1 for stability (was 4)
                            self.model.train(gradient_steps=1)
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
                        
                        # Save checkpoint and metadata
                        ckpt_path = f"{self.run_dir}/drl_agent_checkpoint_{step}"
                        self.model.save(ckpt_path)
                        self._save_metadata(ckpt_path, step)

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
            
            final_path = f"{self.run_dir}/drl_agent_final"
            self.model.save(final_path)
            self._save_metadata(final_path, self.total_steps)

if __name__ == "__main__":
    agent = SenderAgent()
    agent.start_feedback_listener()
    agent.run()