import socket
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import numpy as np
import time
import io

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
IMAGE_DIR = '/app/images'
CLASSES = ['cat', 'car', 'dog']

# --- FEEDBACK CONFIG ---
SENDER_HOST = 'sender'
SENDER_FEEDBACK_PORT = 65500
LATENCY_DEADLINE_TAU = 1.0
ALPHA_WEIGHT = 0.5

# --- MODEL SETUP ---
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
preprocess = weights.transforms()

# --- FEEDBACK HELPERS ---
def calculate_semantic_loss(ground_truth_vec, noisy_vec):
    if ground_truth_vec is None or noisy_vec is None: return 1.0
    return np.mean((ground_truth_vec - noisy_vec)**2)

def calculate_reward(semantic_loss, latency, deadline):
    latency_met = 1.0 if latency <= deadline else 0.0
    latency_penalty = 1.0 - latency_met
    cost = latency_penalty + ALPHA_WEIGHT * semantic_loss
    return -cost

# --- NEW IN PHASE 3: Send back reward AND network state ---
def send_feedback(reward, noise, bandwidth):
    """Sends the calculated reward and observed network state back to the sender."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as feedback_sock:
            feedback_sock.connect((SENDER_HOST, SENDER_FEEDBACK_PORT))
            
            # Create a 3-element float array
            feedback_payload = np.array([reward, noise, bandwidth], dtype=np.float32)
            
            feedback_sock.sendall(feedback_payload.tobytes())
    except Exception as e:
        print(f"❌ Error sending feedback to sender: {e}")

# --- Refactored vector extraction ---
def get_vector_from_image(img):
    """Extracts a vector from a PIL Image object."""
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
        return features.squeeze().numpy()

def get_image_feature_vector(image_path):
    """Utility function to extract feature vector from a file path."""
    img = Image.open(image_path).convert('RGB')
    return get_vector_from_image(img)

# --- ORIGINAL FUNCTIONS ---
def create_knowledge_base():
    print("Creating receiver's knowledge base...")
    knowledge_base = {}
    for class_name in CLASSES:
        image_path = f"{IMAGE_DIR}/{class_name}.jpeg"
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found {image_path}. Skipping.")
            continue
        knowledge_base[class_name] = get_image_feature_vector(image_path)
        print(f" - Generated vector for '{class_name}'")
    return knowledge_base

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return -1
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def decode_semantic_meaning(noisy_vector, knowledge_base):
    max_similarity = -1
    best_match = "UNKNOWN"
    for class_name, ideal_vector in knowledge_base.items():
        similarity = cosine_similarity(noisy_vector, ideal_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name
    return best_match, similarity

# --- MAIN RECEIVER LOOP ---
knowledge_base = create_knowledge_base()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("\nReceiver is listening for connections...")
    
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"\nConnected by {addr}")
            
            buffer = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            data = buffer

            if not data:
                print("Received an empty message. Skipping.")
                continue

            reconstructed_vector = None
            original_label = None
            send_timestamp = None
            log_msg_type = "UNKNOWN"
            
            # --- NEW IN PHASE 3: Network state variables ---
            observed_noise = -1.0
            observed_bandwidth = -1.0

            try:
                # --- NEW IN PHASE 3: 5-Part Message Format ---
                # noise | bandwidth | type | timestamp | label | payload
                noise_bytes, rest_of_data = data.split(b'|', 1)
                bw_bytes, rest_of_data = rest_of_data.split(b'|', 1)
                type_bytes, rest_of_data = rest_of_data.split(b'|', 1)
                timestamp_bytes, rest_of_data = rest_of_data.split(b'|', 1)
                original_label_bytes, payload = rest_of_data.split(b'|', 1)

                # Unpack network state
                observed_noise = np.frombuffer(noise_bytes, dtype=np.float32)[0]
                observed_bandwidth = np.frombuffer(bw_bytes, dtype=np.float32)[0]

                # Unpack original message
                send_timestamp = np.frombuffer(timestamp_bytes, dtype=np.float64)[0]
                original_label = original_label_bytes.decode('utf-8')

                if type_bytes == b"SEM":
                    log_msg_type = "SEMANTIC"
                    reconstructed_vector = np.frombuffer(payload, dtype=np.float32)

                elif type_bytes == b"RAW":
                    log_msg_type = "RAW"
                    img_stream = io.BytesIO(payload)
                    img = Image.open(img_stream).convert('RGB')
                    reconstructed_vector = get_vector_from_image(img)

                else:
                    print(f"Error: Unknown message type '{type_bytes}'. Skipping.")
                    continue
            
            except Exception as e:
                print(f"Error unpacking data: {e}. Buffer size: {len(data)}. Skipping packet.")
                continue

            # --- Performance Calculation (Same as Phase 1) ---
            reception_timestamp = time.time()
            total_latency = reception_timestamp - send_timestamp

            ground_truth_vector = knowledge_base.get(original_label)
            
            if (ground_truth_vector is not None and 
                reconstructed_vector is not None and
                ground_truth_vector.shape != reconstructed_vector.shape):
                
                print(f"Shape mismatch! Ground Truth: {ground_truth_vector.shape}, Reconstructed: {reconstructed_vector.shape}")
                semantic_loss = 1.0 # Max penalty
            else:
                semantic_loss = calculate_semantic_loss(ground_truth_vector, reconstructed_vector)

            reward = calculate_reward(semantic_loss, total_latency, LATENCY_DEADLINE_TAU)

            # --- NEW IN PHASE 3: Send full feedback package ---
            send_feedback(reward, observed_noise, observed_bandwidth)

            # --- Logging ---
            decoded_label, similarity = decode_semantic_meaning(reconstructed_vector, knowledge_base)
            
            print(f"Received {log_msg_type} for: {original_label}")
            print(f"-> Decoded Meaning: **{decoded_label}** (Similarity: {similarity:.4f})")
            print(f"Latency: {total_latency:.4f}s")
            print(f"Semantic Loss (MSE): {semantic_loss:.4f}")
            print(f"Network State: (Noise: {observed_noise:.3f}, BW: {observed_bandwidth:.2f})")
            print(f"Calculated Reward: {reward:.4f}")

            if original_label == decoded_label:
                print("✅ SUCCESS: Meaning was recovered.")
            else:
                print("❌ FAILURE: Meaning was lost.")