import socket
import torch
import torchvision.transforms as transforms
from utils.models import SimpleDecoder # Import custom Decoder
from PIL import Image
import os
import numpy as np
import time
import io
import pickle
import logging
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import requests

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [Receiver] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration Constants ---
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 65432
DEFAULT_IMAGE_DIR = '/app/images'
DEFAULT_CLASSES = ['cat', 'car', 'dog']
DEFAULT_SENDER_HOST = 'sender'
DEFAULT_SENDER_FEEDBACK_PORT = 65500
DEFAULT_LATENCY_DEADLINE_TAU = 1.0
DEFAULT_MISSED_DEADLINE_PENALTY = 10.0
DEFAULT_ALPHA_WEIGHT = 20.0
DEFAULT_LATENCY_PENALTY_FACTOR = 2.0

# --- SIMULATED TIMES (Seconds) ---
SIM_ENC_SEMANTIC_LOCAL = 0.200 # Slow Mobile CPU
SIM_ENC_SEMANTIC_EDGE = 0.020  # Fast Edge GPU
SIM_ENC_RAW = 0.005

SIM_DEC_SEMANTIC = 0.050
SIM_DEC_RAW = 0.010

EDGE_QUALITY_MULTIPLIER = 0.5 # Edge model is better, so 50% less loss
RAW_NOISE_SENSITIVITY = 10.0 # RAW is very sensitive to noise (simulating packet corruption)


class Receiver:
    """
    Acts as the receiver in the semantic communication loop.
    
    It listens for messages from the channel, decodes them, calculates
    performance (latency, semantic loss), and sends a feedback package
    (reward, net_state) back to the sender.
    """

    def __init__(self, host: str, port: int, image_dir: str, classes: list,
                 sender_host: str, feedback_port: int, deadline: float, alpha: float):
        
        # --- Configuration ---
        self.host = host
        self.port = port
        self.image_dir = image_dir
        self.classes = classes
        self.sender_host = sender_host
        self.feedback_port = feedback_port
        self.deadline_tau = deadline
        self.alpha_weight = alpha
        self.latency_penalty_factor = DEFAULT_LATENCY_PENALTY_FACTOR

        # --- Initialize Decoder (Simple/Local) ---
        logging.info("Initializing Custom Decoder (Simple/Local)...")
        self.decoder = SimpleDecoder(encoded_space_dim=512)
        
        # Load pre-trained weights if available
        ae_weights_path = "/app/models/simple_decoder.pth"
        if os.path.exists(ae_weights_path):
            logging.info(f"Loading SimpleDecoder weights from {ae_weights_path}...")
            try:
                state_dict = torch.load(ae_weights_path, map_location='cpu')
                # No prefix stripping needed for standalone decoder weights
                self.decoder.load_state_dict(state_dict)
                logging.info("Decoder weights loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load decoder weights: {e}")
        else:
            logging.warning("No pre-trained weights found. Using random initialization.")
        self.decoder.eval()
        
        # Transform for GT image (to match Decoder output)
        self.preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(), # Converts to [0, 1]
        ])
        
        # --- Knowledge Base Initialization ---
        # self.knowledge_base = self._create_knowledge_base() # Removed: We now receive GT vector
        
        # --- TensorBoard Logging ---
        logging.info("Initializing TensorBoard SummaryWriter...")
        self.writer = SummaryWriter(log_dir="/app/runs/receiver_logs")
        self.step_counter = 0

    def _decode_edge_vector(self, vector: np.ndarray) -> torch.Tensor:
        """Decodes vector using the Edge Decoder service."""
        try:
            # Send POST request
            response = requests.post(
                "http://edge-decoder:8000/decode",
                json={"vector": vector.tolist()}
            )
            
            if response.status_code == 200:
                img_bytes = response.content
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return self.preprocess(img)
            else:
                logging.error(f"Edge Decoder failed: {response.text}")
                # Fallback to local decoder
                return self._decode_vector(vector)
                
        except Exception as e:
            logging.error(f"Error calling Edge Decoder: {e}")
            return self._decode_vector(vector)

    def _decode_vector(self, vector: np.ndarray) -> torch.Tensor:
        """Decodes the semantic vector into an image tensor."""
        with torch.no_grad():
            # Vector is numpy, convert to tensor
            z = torch.from_numpy(vector).float().unsqueeze(0) # Add batch dim
            reconstructed_img = self.decoder(z)
            return reconstructed_img.squeeze(0) # Remove batch dim -> [C, H, W]

    def _get_image_feature_vector(self, image_path: str) -> np.ndarray:
        """Utility function to extract feature vector from a file path."""
        img = Image.open(image_path).convert('RGB')
        return self._get_vector_from_image(img)

    # def _create_knowledge_base(self) -> Dict[str, np.ndarray]:
    #     """
    #     Generates the ideal feature vector for each known class and stores it.
    #     This is the receiver's semantic "ground truth".
    #     """
    #     logging.info("Creating receiver's knowledge base...")
    #     knowledge_base = {}
    #     for class_name in self.classes:
    #         image_path = f"{self.image_dir}/{class_name}.jpeg"
    #         if not os.path.exists(image_path):
    #             logging.warning(f"Image file not found {image_path}. Skipping.")
    #             continue
    #         knowledge_base[class_name] = self._get_image_feature_vector(image_path)
    #         logging.info(f" - Generated vector for '{class_name}'")
    #     return knowledge_base

    def _calculate_reconstruction_loss(self, gt_image_tensor: torch.Tensor, 
                                     reconstructed_image_tensor: torch.Tensor) -> float:
        """Calculates MSE loss between two image tensors."""
        if gt_image_tensor is None or reconstructed_image_tensor is None:
            return 1.0
        
        # MSE Loss
        loss = torch.nn.functional.mse_loss(reconstructed_image_tensor, gt_image_tensor)
        return loss.item()

    def _calculate_reward(self, semantic_loss: float, latency: float) -> float:
        """Calculates the reward based on loss and latency."""
        latency_met = latency <= self.deadline_tau
        
        # Binary Penalty for missing deadline
        binary_penalty = 0.0 if latency_met else DEFAULT_MISSED_DEADLINE_PENALTY
        
        # New Reward Function:
        # Cost = BinaryDeadline + (LinearLatency * Factor) + (MSE * Alpha)
        cost = binary_penalty + (latency * self.latency_penalty_factor) + (self.alpha_weight * semantic_loss)
        return -cost

    def _send_feedback(self, reward: float, noise: float, bandwidth: float):
        """Sends the calculated reward and observed network state back to the sender."""
        try:
            logging.info(f"Sending feedback to {self.sender_host}:{self.feedback_port} -> Reward: {reward:.4f}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as feedback_sock:
                feedback_sock.connect((self.sender_host, self.feedback_port))
                feedback_payload = np.array([reward, noise, bandwidth], dtype=np.float32)
                feedback_sock.sendall(feedback_payload.tobytes())
            logging.info("Feedback sent successfully.")
        except socket.error as e:
            logging.error(f"Error sending feedback to sender ({self.sender_host}:{self.feedback_port}): {e}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity, handling potential None values."""
        if vec1 is None or vec2 is None:
            return -1.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _decode_semantic_meaning(self, reconstructed_vector: np.ndarray) -> (str, float):
        """Finds the best match for the vector from the knowledge base."""
        # For CIFAR-10, we don't have a static KB anymore. 
        # We could compare to the GT vector we received, but that's trivial (sim=1.0).
        # So we just return "Dynamic", 0.0 for now or implement a proper classifier later.
        return "Dynamic", 0.0

    def _receive_n_bytes(self, conn: socket.socket, n: int) -> Optional[bytes]:
        """Reads exactly n bytes from the connection."""
        buffer = b""
        while len(buffer) < n:
            chunk = conn.recv(n - len(buffer))
            if not chunk:
                return None
            buffer += chunk
        return buffer

    def _receive_message_payload(self, conn: socket.socket) -> Optional[bytes]:
        """Receives the full message payload."""
        logging.info("Starting to receive message payload...")
        conn.settimeout(5.0) # Set 5s timeout to prevent hanging
        buffer = b""
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    logging.info("Client closed connection (EOF).")
                    break
                buffer += chunk
                # logging.info(f"Received chunk: {len(chunk)} bytes. Total: {len(buffer)}")
            logging.info(f"Finished receiving. Total size: {len(buffer)} bytes.")
            return buffer
        except socket.timeout:
            logging.warning("Socket timed out while receiving payload!")
            return None
        except Exception as e:
            logging.error(f"Error receiving payload: {e}")
            return None

    def _process_message(self, data: bytes, reception_timestamp: float):
        """
        The core business logic to unpack, decode, and handle a received message.
        """
        logging.info(f"Processing message of size {len(data)} bytes...")
        reconstructed_vector = None
        original_label = None
        send_timestamp = None
        log_msg_type = "UNKNOWN"
        observed_noise = -1.0
        observed_bandwidth = -1.0
        
        # For latency calc
        sim_enc_time = 0.0
        sim_dec_time = 0.0
        is_edge_quality = False

        try:
            # --- 1. Unpack Network State from Payload ---
            # --- 1. Unpack Network State from Payload ---
            # FORMAT: noise (4 bytes) | bandwidth (4 bytes) | pickled_payload
            noise_bytes = data[0:4]
            bw_bytes = data[4:8]
            pickled_payload = data[8:]

            observed_noise = np.frombuffer(noise_bytes, dtype=np.float32)[0]
            observed_bandwidth = np.frombuffer(bw_bytes, dtype=np.float32)[0]

            # --- 2. Unpack the main message dictionary ---
            logging.info("Unpickling payload...")
            message_dict: Dict[str, Any] = pickle.loads(pickled_payload)
            
            send_timestamp = message_dict['ts']
            original_label = message_dict['label']
            msg_type = message_dict['type']
            payload = message_dict['payload']

            # --- SNR Calculation ---
            snr_db = -1.0 # Default if not applicable
            if msg_type in ["SEM_LOCAL", "SEM_EDGE"]:
                # Payload is the NOISY vector (from Channel)
                # We can approximate Signal Power as Mean(Vector^2)
                # Note: This includes Noise Power too, but if SNR is high, P_rx approx P_signal
                # Or strictly: P_rx = P_signal + P_noise
                # P_signal = P_rx - P_noise. 
                # P_noise = observed_noise^2
                
                # Let's use simple P_signal estimate from received vector for robustness
                # (Standard practice in receiver is measuring RSSI)
                received_vector = payload
                p_rx = np.mean(received_vector**2)
                p_noise = observed_noise**2
                
                # Avoid div by zero
                if p_noise > 1e-9:
                    p_signal_est = max(0.0, p_rx - p_noise) # Subtract estimated noise power
                    if p_signal_est > 0:
                         snr_db = 10 * np.log10(p_signal_est / p_noise)
                    else:
                         snr_db = 0.0 # Very low SNR
                else:
                    snr_db = 50.0 # High SNR cap if no noise
                    
                logging.info(f"SNR Calculation: P_rx={p_rx:.4f}, NoiseSigma={observed_noise:.4f}, SNR={snr_db:.2f} dB")


            # --- 3. Decode payload based on type ---
            logging.info(f"Decoding message type: {msg_type}")
            gt_image_bytes = message_dict.get('gt_image')
            gt_image = None
            if gt_image_bytes:
                logging.info("Processing GT image...")
                gt_img_pil = Image.open(io.BytesIO(gt_image_bytes)).convert('RGB')
                gt_image = self.preprocess(gt_img_pil) # [C, H, W] tensor

            if msg_type == "SEM_LOCAL":
                log_msg_type = "SEM_LOCAL"
                # Payload is the noisy vector
                noisy_vector = payload 
                logging.info("Running Local Decoder...")
                reconstructed_image = self._decode_vector(noisy_vector)
                
                sim_enc_time = SIM_ENC_SEMANTIC_LOCAL
                sim_dec_time = SIM_DEC_SEMANTIC

            elif msg_type == "SEM_EDGE":
                log_msg_type = "SEM_EDGE"
                # Payload is the noisy vector
                noisy_vector = payload 
                logging.info("Running Edge Decoder...")
                reconstructed_image = self._decode_edge_vector(noisy_vector)
                
                # Edge Latency = Fast Compute + Upload Lag
                # Simulate 50KB upload to Edge
                edge_upload_size_bits = 50 * 1024 * 8 
                # Avoid div by zero
                current_bw_bps = max(1.0, observed_bandwidth) * 1_000_000
                edge_upload_delay = edge_upload_size_bits / current_bw_bps
                
                sim_enc_time = SIM_ENC_SEMANTIC_EDGE + edge_upload_delay
                sim_dec_time = SIM_DEC_SEMANTIC
                
                # Apply Quality Multiplier (simulate better model)
                # We apply it to the final loss calculation later, or modify here?
                # Let's flag it.
                is_edge_quality = True

            elif msg_type == "RAW":
                log_msg_type = "RAW"
                # Payload is the raw image bytes (same as GT in this case)
                img_stream = io.BytesIO(payload)
                img = Image.open(img_stream).convert('RGB')
                logging.info("Processing RAW image...")
                reconstructed_image = self.preprocess(img)
                
                sim_enc_time = SIM_ENC_RAW
                sim_dec_time = SIM_DEC_RAW

            else:
                logging.warning(f"Error: Unknown message type '{msg_type}'. Skipping.")
                return

        except Exception as e:
            logging.error(f"Error unpacking/processing data: {e}. Buffer size: {len(data)}. Skipping packet.")
            return

        # --- 4. Performance Calculation ---
        logging.info("Calculating performance...")
        # reception_timestamp is passed in now to capture network arrival time
        
        # Network Latency
        network_latency = reception_timestamp - send_timestamp
        
        # Total Latency = Network + Sim Enc + Sim Dec
        total_latency = network_latency + sim_enc_time + sim_dec_time

        # Calculate Reconstruction Loss (MSE)
        reconstruction_loss = self._calculate_reconstruction_loss(gt_image, reconstructed_image)
        
        # Apply Logic:
        # 1. Edge Quality Benefit
        if is_edge_quality:
            reconstruction_loss *= EDGE_QUALITY_MULTIPLIER
            
        # 2. RAW Noise Penalty (Simulate Corruption)
        if msg_type == "RAW" and observed_noise > 0:
            # Simulate massive corruption proportional to noise
            raw_corruption = RAW_NOISE_SENSITIVITY * (observed_noise ** 2)
            reconstruction_loss += raw_corruption
            logging.info(f"  -> Applied RAW Noise Penalty: +{raw_corruption:.4f} MSE")

        reward = self._calculate_reward(reconstruction_loss, total_latency)

        # --- 5. Send Feedback ---
        logging.info("Sending feedback...")
        self._send_feedback(reward, observed_noise, observed_bandwidth)

        # --- 6. Logging ---
        self.step_counter += 1
        self.writer.add_scalar("Performance/Latency", total_latency, self.step_counter)
        self.writer.add_scalar("Performance/ReconstructionLoss", reconstruction_loss, self.step_counter)
        self.writer.add_scalar("Performance/Reward", reward, self.step_counter)
        self.writer.add_scalar("Network/Noise", observed_noise, self.step_counter)
        self.writer.add_scalar("Network/Bandwidth", observed_bandwidth, self.step_counter)
        if snr_db > -1.0:
             self.writer.add_scalar("Network/SNR_dB", snr_db, self.step_counter)
            
            # Log images occasionally
        if self.step_counter % 50 == 0:
            self.writer.add_image("Images/Reconstructed", reconstructed_image, self.step_counter)
            if gt_image is not None:
                self.writer.add_image("Images/GroundTruth", gt_image, self.step_counter)

        logging.info(f"Received {log_msg_type} for: {original_label}")
        logging.info(f"Latency: {total_latency:.4f}s")
        logging.info(f"Reconstruction Loss (MSE): {reconstruction_loss:.4f}")
        logging.info(f"Network State: (Noise: {observed_noise:.3f}, BW: {observed_bandwidth:.2f})")
        logging.info(f"Calculated Reward: {reward:.4f}")

    def handle_client(self, conn: socket.socket, addr):
        """Manages a single connection from the channel."""
        logging.info(f"Connected by {addr}")
        try:
            data = self._receive_message_payload(conn)
            if not data:
                logging.info(f"Received an empty message from {addr}. Skipping.")
            else:
                # Capture timestamp immediately after reception!
                reception_timestamp = time.time()
                self._process_message(data, reception_timestamp)
        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}")
        finally:
            conn.close()

    def run(self):
        """Starts the main server loop to listen for channel connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, self.port))
                s.listen()
                logging.info(f"Receiver is listening on {self.host}:{self.port}...")
                
                while True:
                    try:
                        conn, addr = s.accept()
                        # This design is simple and handles one connection at a time,
                        # which is fine for your sequential DRL loop.
                        self.handle_client(conn, addr)
                    except Exception as e:
                        logging.error(f"Error accepting connection: {e}")
        except OSError as e:
            logging.critical(f"Failed to bind socket: {e}. Exiting.")


if __name__ == "__main__":
    receiver = Receiver(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        image_dir=DEFAULT_IMAGE_DIR,
        classes=DEFAULT_CLASSES,
        sender_host=DEFAULT_SENDER_HOST,
        feedback_port=DEFAULT_SENDER_FEEDBACK_PORT,
        deadline=DEFAULT_LATENCY_DEADLINE_TAU,
        alpha=DEFAULT_ALPHA_WEIGHT
    )
    receiver.run()