import socket
import numpy as np
import time
import struct  # Import struct for message framing

SENDER_HOST = '0.0.0.0'
SENDER_PORT = 65431
RECEIVER_HOST = 'receiver'
RECEIVER_PORT = 65432

# --- NOISE CONFIGURATION ---
# NOISE_LEVEL = 0.1  # <-- REMOVED
# VECTOR_SIZE = 512  # <-- REMOVED
# def add_noise(vector_bytes): # <-- ENTIRE FUNCTION REMOVED
#     ...

# --- Helper function to read exactly n bytes ---
def recv_all(conn, n):
    """
    Helper function to receive exactly n bytes from a socket.
    """
    buffer = b""
    while len(buffer) < n:
        chunk = conn.recv(n - len(buffer))
        if not chunk:
            # Connection dropped before we got all the data
            return None
        buffer += chunk
    return buffer

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((SENDER_HOST, SENDER_PORT))
    s.listen()
    print("Channel is permanently listening for senders...")

    while True:
        print("\nWaiting for a new sender connection...")
        try:
            conn, addr = s.accept()
            with conn:
                print(f"Sender connected from {addr}")
                
                # --- Main loop with message framing ---
                while True:
                    # 1. Read the 4-byte message length header
                    header_bytes = recv_all(conn, 4)
                    if not header_bytes:
                        print(f"Sender {addr} disconnected (header read).")
                        break
                    
                    # Unpack the message length
                    msg_len = struct.unpack('!I', header_bytes)[0]
                    
                    # 2. Read exactly that many bytes for the full message
                    data = recv_all(conn, msg_len)
                    if not data:
                        print(f"Sender {addr} disconnected (payload read).")
                        break
                    
                    # --- The rest of your logic is now safe ---
                    try:
                        timestamp_bytes, rest_of_data = data.split(b'|', 1)
                        label_bytes, vector_bytes = rest_of_data.split(b'|', 1)
                    
                    except Exception as e:
                        print(f"Error splitting message: {e}. Skipping.")
                        continue

                    # --- THIS IS THE CHANGE ---
                    # We no longer call add_noise. We just use the original vector bytes.
                    noisy_vector_bytes = vector_bytes
                    # --- END OF CHANGE ---
                    
                    # --- Changed print statement to reflect the new behavior ---
                    print(f"Received '{label_bytes.decode()}' (Size: {msg_len}). Forwarding (NO NOISE).")
                    
                    message_to_forward = timestamp_bytes + b'|' + label_bytes + b'|' + noisy_vector_bytes

                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
                            receiver_socket.connect((RECEIVER_HOST, RECEIVER_PORT))
                            receiver_socket.sendall(message_to_forward)
                    except Exception as e:
                        print(f"Channel error forwarding to receiver: {e}")
                        
        except Exception as e:
            print(f"Error in main connection loop: {e}. Resetting...")
            time.sleep(1)