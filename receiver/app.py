import socket
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import numpy as np
import time
import io
import pickle
import logging
from typing import Dict, Any, Optional

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

        # --- Model Initialization ---
        logging.info("Initializing ResNet-18 feature extractor...")
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()
        self.preprocess = self.weights.transforms()
        
        # --- Knowledge Base Initialization ---
        self.knowledge_base = self._create_knowledge_base()

    def _get_vector_from_image(self, img: Image.Image) -> np.ndarray:
        """Extracts a feature vector from a PIL Image object."""
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            features = self.feature_extractor(batch_t)
            return features.squeeze().numpy()

    def _get_image_feature_vector(self, image_path: str) -> np.ndarray:
        """Utility function to extract feature vector from a file path."""
        img = Image.open(image_path).convert('RGB')
        return self._get_vector_from_image(img)

    def _create_knowledge_base(self) -> Dict[str, np.ndarray]:
        """
        Generates the ideal feature vector for each known class and stores it.
        This is the receiver's semantic "ground truth".
        """
        logging.info("Creating receiver's knowledge base...")
        knowledge_base = {}
        for class_name in self.classes:
            image_path = f"{self.image_dir}/{class_name}.jpeg"
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found {image_path}. Skipping.")
                continue
            knowledge_base[class_name] = self._get_image_feature_vector(image_path)
            logging.info(f" - Generated vector for '{class_name}'")
        return knowledge_base

    def _calculate_semantic_loss(self, ground_truth_vec: Optional[np.ndarray],
                                 reconstructed_vec: Optional[np.ndarray]) -> float:
        """Calculates Mean Squared Error (MSE) between two vectors."""
        if ground_truth_vec is None or reconstructed_vec is None:
            return 1.0  # Max penalty
        return np.mean((ground_truth_vec - reconstructed_vec)**2)

    def _calculate_reward(self, semantic_loss: float, latency: float) -> float:
        """Calculates the reward based on loss and latency."""
        latency_met = 1.0 if latency <= self.deadline_tau else 0.0
        latency_penalty = 1.0 - latency_met
        cost = latency_penalty + self.alpha_weight * semantic_loss
        return -cost

    def _send_feedback(self, reward: float, noise: float, bandwidth: float):
        """Sends the calculated reward and observed network state back to the sender."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as feedback_sock:
                feedback_sock.connect((self.sender_host, self.feedback_port))
                feedback_payload = np.array([reward, noise, bandwidth], dtype=np.float32)
                feedback_sock.sendall(feedback_payload.tobytes())
        except socket.error as e:
            logging.error(f"Error sending feedback to sender: {e}")

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
        max_similarity = -1.0
        best_match = "UNKNOWN"
        for class_name, ideal_vector in self.knowledge_base.items():
            similarity = self._cosine_similarity(reconstructed_vector, ideal_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = class_name
        return best_match, max_similarity

    def _receive_full_message(self, conn: socket.socket) -> bytes:
        """Reads all data from a single connection until it closes."""
        buffer = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buffer += chunk
        return buffer

    def _process_message(self, data: bytes):
        """
        The core business logic to unpack, decode, and handle a received message.
        """
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
            message_dict: Dict[str, Any] = pickle.loads(pickled_payload)
            
            send_timestamp = message_dict['ts']
            original_label = message_dict['label']
            msg_type = message_dict['type']
            payload = message_dict['payload']

            # --- 3. Decode payload based on type ---
            if msg_type == "SEM":
                log_msg_type = "SEMANTIC"
                reconstructed_vector = payload  # Payload is the noisy numpy vector

            elif msg_type == "RAW":
                log_msg_type = "RAW"
                img_stream = io.BytesIO(payload)  # Payload is the raw image bytes
                img = Image.open(img_stream).convert('RGB')
                reconstructed_vector = self._get_vector_from_image(img)

            else:
                logging.warning(f"Error: Unknown message type '{msg_type}'. Skipping.")
                return

        except Exception as e:
            logging.error(f"Error unpacking data: {e}. Buffer size: {len(data)}. Skipping packet.")
            return

        # --- 4. Performance Calculation ---
        reception_timestamp = time.time()
        total_latency = reception_timestamp - send_timestamp

        ground_truth_vector = self.knowledge_base.get(original_label)
        
        if (ground_truth_vector is not None and 
            reconstructed_vector is not None and
            ground_truth_vector.shape != reconstructed_vector.shape):
            
            logging.warning(f"Shape mismatch! GT: {ground_truth_vector.shape}, Rec: {reconstructed_vector.shape}")
            semantic_loss = 1.0 # Max penalty
        else:
            semantic_loss = self._calculate_semantic_loss(ground_truth_vector, reconstructed_vector)

        reward = self._calculate_reward(semantic_loss, total_latency)

        # --- 5. Send Feedback ---
        self._send_feedback(reward, observed_noise, observed_bandwidth)

        # --- 6. Logging ---
        decoded_label, similarity = self._decode_semantic_meaning(reconstructed_vector)
        
        logging.info(f"Received {log_msg_type} for: {original_label}")
        logging.info(f"-> Decoded Meaning: **{decoded_label}** (Similarity: {similarity:.4f})")
        logging.info(f"Latency: {total_latency:.4f}s")
        logging.info(f"Semantic Loss (MSE): {semantic_loss:.4f}")
        logging.info(f"Network State: (Noise: {observed_noise:.3f}, BW: {observed_bandwidth:.2f})")
        logging.info(f"Calculated Reward: {reward:.4f}")

        if original_label == decoded_label:
            logging.info("✅ SUCCESS: Meaning was recovered.")
        else:
            logging.info("❌ FAILURE: Meaning was lost.")

    def handle_client(self, conn: socket.socket, addr):
        """Manages a single connection from the channel."""
        logging.info(f"Connected by {addr}")
        try:
            data = self._receive_full_message(conn)
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