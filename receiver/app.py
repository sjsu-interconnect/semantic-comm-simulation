import socket
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import numpy as np
import time

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
IMAGE_DIR = '/app/images'
CLASSES = ['cat', 'car', 'dog']

# --- FEEDBACK CONFIG (NEW IN PHASE 1) ---
SENDER_HOST = 'sender'
SENDER_FEEDBACK_PORT = 65500
LATENCY_DEADLINE_TAU = 1.0  # 1.0 second deadline
ALPHA_WEIGHT = 0.5  # Weight for semantic loss in reward

# --- MODEL SETUP (Same as sender to ensure consistency) ---
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
preprocess = weights.transforms()

# --- FEEDBACK HELPERS (NEW IN PHASE 1) ---

def calculate_semantic_loss(ground_truth_vec, noisy_vec):
    """Calculates Mean Squared Error (MSE) between two vectors."""
    # Ensure vectors are not None
    if ground_truth_vec is None or noisy_vec is None:
        return 1.0 # Return max penalty if something is wrong
    return np.mean((ground_truth_vec - noisy_vec)**2)

def calculate_reward(semantic_loss, latency, deadline):
    """Calculates the reward based on loss and latency."""
    latency_met = 1.0 if latency <= deadline else 0.0
    latency_penalty = 1.0 - latency_met
    
    # Cost function to be minimized
    cost = latency_penalty + ALPHA_WEIGHT * semantic_loss
    
    # Reward is the negative of the cost
    return -cost

def send_feedback(reward):
    """Sends the calculated reward back to the sender."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as feedback_sock:
            # Connect to the sender's feedback port
            feedback_sock.connect((SENDER_HOST, SENDER_FEEDBACK_PORT))
            
            # Send the reward as a 32-bit float
            feedback_sock.sendall(np.array([reward], dtype=np.float32).tobytes())
            
    except Exception as e:
        print(f"❌ Error sending feedback to sender: {e}")

# --- ORIGINAL FUNCTIONS ---

def get_image_feature_vector(image_path):
    """Utility function to extract feature vector."""
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
        return features.squeeze().numpy()

def create_knowledge_base():
    """
    Generates the ideal feature vector for each known class and stores it.
    """
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
    """Calculates cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return -1
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def decode_semantic_meaning(noisy_vector, knowledge_base):
    """Finds the best match for the noisy vector from the knowledge base."""
    max_similarity = -1
    best_match = "UNKNOWN"
    for class_name, ideal_vector in knowledge_base.items():
        similarity = cosine_similarity(noisy_vector, ideal_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name
    return best_match, max_similarity

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
            
            # --- THIS IS THE FIX ---
            # Create a buffer to accumulate all incoming data
            buffer = b""
            while True:
                # Receive data in chunks
                chunk = conn.recv(4096)
                if not chunk:
                    # If chunk is empty, the sender (channel) has closed
                    # the connection. We have the full message.
                    break
                buffer += chunk # Add chunk to the buffer
            
            # Now, 'buffer' contains the entire message
            data = buffer
            # --- END OF FIX ---

            if not data:
                print("Received an empty message. Skipping.")
                continue

            # --- NEW IN PHASE 1: Data Deserialization ---
            try:
                # NEW FORMAT: timestamp_bytes | label_bytes | noisy_vector_bytes
                timestamp_bytes, rest_of_data = data.split(b'|', 1)
                original_label_bytes, noisy_vector_bytes = rest_of_data.split(b'|', 1)

                # Deserialize
                send_timestamp = np.frombuffer(timestamp_bytes, dtype=np.float64)[0]
                original_label = original_label_bytes.decode('utf-8')
                noisy_vector = np.frombuffer(noisy_vector_bytes, dtype=np.float32)

            except Exception as e:
                print(f"Error unpacking data: {e}. Buffer size: {len(data)}. Skipping packet.")
                continue

            # --- NEW IN PHASE 1: Performance Calculation ---
            
            # 1. Calculate Latency
            reception_timestamp = time.time()
            total_latency = reception_timestamp - send_timestamp

            # 2. Calculate Semantic Loss
            ground_truth_vector = knowledge_base.get(original_label)
            if ground_truth_vector is not None:
                # Check vector shape integrity just in case
                if noisy_vector.shape != ground_truth_vector.shape:
                    print(f"Shape mismatch! Expected {ground_truth_vector.shape} but got {noisy_vector.shape}. Assigning max penalty.")
                    semantic_loss = 1.0 # Max penalty
                else:
                    semantic_loss = calculate_semantic_loss(ground_truth_vector, noisy_vector)
            else:
                print(f"Error: Unknown label '{original_label}' received.")
                semantic_loss = 1.0 # Assign max penalty

            # 3. Calculate Reward
            reward = calculate_reward(semantic_loss, total_latency, LATENCY_DEADLINE_TAU)

            # 4. Send Feedback
            send_feedback(reward)

            # --- Original decoding logic (for logging) ---
            decoded_label, similarity = decode_semantic_meaning(noisy_vector, knowledge_base)
            
            print(f"Original Label: {original_label}")
            print(f"-> Decoded Meaning: **{decoded_label}** (Similarity: {similarity:.4f})")
            
            # --- NEW IN PHASE 1: Print new metrics ---
            print(f"Latency: {total_latency:.4f}s")
            print(f"Semantic Loss (MSE): {semantic_loss:.4f}")
            print(f"Calculated Reward: {reward:.4f}")

            if original_label == decoded_label:
                print("✅ SUCCESS: Meaning was recovered.")
            else:
                print("❌ FAILURE: Meaning was lost.")