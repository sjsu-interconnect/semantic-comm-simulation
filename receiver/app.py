import socket
import torch
import torchvision.transforms as transforms
from utils.models import Decoder # Import custom Decoder
from PIL import Image
import os
import numpy as np
import time
import io
import pickle
import logging
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

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
DEFAULT_ALPHA_WEIGHT = 0.5


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

        # --- Initialize Decoder ---
        logging.info("Initializing Custom Decoder...")
        self.decoder = Decoder(encoded_space_dim=512)
        
        # Load pre-trained weights if available
        ae_weights_path = "/app/models/autoencoder_cifar10.pth"
        if os.path.exists(ae_weights_path):
            logging.info(f"Loading pre-trained Autoencoder weights from {ae_weights_path}...")
            try:
                state_dict = torch.load(ae_weights_path, map_location='cpu')
                # Filter for decoder keys only (prefix 'decoder.')
                # Autoencoder has self.decoder = Decoder(). So keys are 'decoder.decoder_lin...'
                # Decoder expects 'decoder_lin...'
                
                decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
                self.decoder.load_state_dict(decoder_state_dict)
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
        latency_met = 1.0 if latency <= self.deadline_tau else 0.0
        latency_penalty = 1.0 - latency_met
        cost = latency_penalty + self.alpha_weight * semantic_loss
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

    def _process_message(self, data: bytes):
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

        try:
            # --- 1. Unpack Network State from Payload ---
            # FORMAT: noise | bandwidth | pickled_payload
            noise_bytes, rest_of_data = data.split(b'|', 1)
            bw_bytes, pickled_payload = rest_of_data.split(b'|', 1)

            observed_noise = np.frombuffer(noise_bytes, dtype=np.float32)[0]
            observed_bandwidth = np.frombuffer(bw_bytes, dtype=np.float32)[0]

            # --- 2. Unpack the main message dictionary ---
            logging.info("Unpickling payload...")
            message_dict: Dict[str, Any] = pickle.loads(pickled_payload)
            
            send_timestamp = message_dict['ts']
            original_label = message_dict['label']
            msg_type = message_dict['type']
            payload = message_dict['payload']

            # --- 3. Decode payload based on type ---
            logging.info(f"Decoding message type: {msg_type}")
            gt_image_bytes = message_dict.get('gt_image')
            gt_image = None
            if gt_image_bytes:
                logging.info("Processing GT image...")
                gt_img_pil = Image.open(io.BytesIO(gt_image_bytes)).convert('RGB')
                gt_image = self.preprocess(gt_img_pil) # [C, H, W] tensor

            if msg_type == "SEM":
                log_msg_type = "SEMANTIC"
                # Payload is the noisy vector
                noisy_vector = payload 
                logging.info("Running Decoder...")
                reconstructed_image = self._decode_vector(noisy_vector)

            elif msg_type == "RAW":
                log_msg_type = "RAW"
                # Payload is the raw image bytes (same as GT in this case)
                img_stream = io.BytesIO(payload)
                img = Image.open(img_stream).convert('RGB')
                logging.info("Processing RAW image...")
                reconstructed_image = self.preprocess(img)

            else:
                logging.warning(f"Error: Unknown message type '{msg_type}'. Skipping.")
                return

        except Exception as e:
            logging.error(f"Error unpacking/processing data: {e}. Buffer size: {len(data)}. Skipping packet.")
            return

        # --- 4. Performance Calculation ---
        logging.info("Calculating performance...")
        reception_timestamp = time.time()
        total_latency = reception_timestamp - send_timestamp

        # Calculate Reconstruction Loss (MSE)
        reconstruction_loss = self._calculate_reconstruction_loss(gt_image, reconstructed_image)

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
                self._process_message(data)
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