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
import struct  # --- NEW: Import struct for message framing ---

# --- CONFIGURATION ---
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
IMAGE_DIR = '/app/images'
IMAGE_FILENAMES = ['cat.jpeg', 'car.jpeg', 'dog.jpeg']

# --- FEEDBACK CONFIG (NEW IN PHASE 1) ---
FEEDBACK_HOST = '0.0.0.0'
FEEDBACK_PORT = 65500

# --- MODEL SETUP ---
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

# --- IMAGE PREPROCESSING ---
preprocess = weights.transforms()

def get_image_feature_vector(image_path):
    """Loads an image, preprocesses it, and extracts its feature vector."""
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        features = feature_extractor(batch_t)
        vector = features.squeeze().numpy()
    return vector

# --- REWARD LISTENER (NEW IN PHASE 1) ---
def reward_listener_thread():
    """
    Sits in a separate thread and listens for reward feedback from the receiver.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((FEEDBACK_HOST, FEEDBACK_PORT))
        s.listen()
        print(f"Sender feedback listener started on port {FEEDBACK_PORT}...")
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    
                    reward = np.frombuffer(data, dtype=np.float32)[0]
                    print(f"\n---  Reward Received: {reward:.4f} ---")
                    
            except Exception as e:
                print(f"Error in feedback listener: {e}")

# --- MAIN SENDER LOOP ---
print("Sender is starting...")

listener_thread = threading.Thread(target=reward_listener_thread, daemon=True)
listener_thread.start()

time.sleep(10)

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print("Sender trying to connect to channel...")
            s.connect((CHANNEL_HOST, CHANNEL_PORT))
            print("Sender connected to channel. Starting to send image semantics.")
            
            while True:
                image_name = random.choice(IMAGE_FILENAMES)
                image_path = os.path.join(IMAGE_DIR, image_name)
                
                vector = get_image_feature_vector(image_path)
                
                send_timestamp = time.time()
                label = image_name.split('.')[0]
                
                timestamp_bytes = np.array([send_timestamp], dtype=np.float64).tobytes()
                label_bytes = label.encode('utf-8')
                vector_bytes = vector.tobytes()
                
                # Create the message payload
                message = timestamp_bytes + b'|' + label_bytes + b'|' + vector_bytes

                # --- NEW: Message Framing ---
                # 1. Pack the length of the message into a 4-byte header
                #    'I' means unsigned integer (4 bytes)
                msg_len_header = struct.pack('!I', len(message))
                
                # 2. Send the header
                s.sendall(msg_len_header)
                
                # 3. Send the actual message
                s.sendall(message)
                
                print(f"Sending semantics for: {label} (Size: {len(message)} bytes)")
                
                time.sleep(5)

    except Exception as e:
        print(f"An error occurred: {e}. Retrying in 5 seconds...")
        time.sleep(5)