import socket
import numpy as np
import time
import struct

SENDER_HOST = '0.0.0.0'
SENDER_PORT = 65431
RECEIVER_HOST = 'receiver'
RECEIVER_PORT = 65432

# --- NOISE IS DISABLED FOR THIS TEST ---
# NOISE_LEVEL = 0.1 
# VECTOR_SIZE = 512
# def add_noise(vector_bytes):
#    ...

# --- Helper function to read exactly n bytes ---
def recv_all(conn, n):
    buffer = b""
    while len(buffer) < n:
        chunk = conn.recv(n - len(buffer))
        if not chunk:
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
                
                while True:
                    # 1. Read the 4-byte message length header
                    header_bytes = recv_all(conn, 4)
                    if not header_bytes:
                        print(f"Sender {addr} disconnected (header read).")
                        break
                    
                    msg_len = struct.unpack('!I', header_bytes)[0]
                    
                    # 2. Read the full message payload
                    data = recv_all(conn, msg_len)
                    if not data:
                        print(f"Sender {addr} disconnected (payload read).")
                        break
                    
                    # --- Message-Aware Logic ---
                    try:
                        # NEW FORMAT: TYPE | timestamp | label | payload
                        type_bytes, rest_of_data = data.split(b'|', 1)
                        
                        if type_bytes == b"SEM":
                            # --- NOISE IS DISABLED ---
                            print(f"Received SEMANTIC message (Size: {msg_len}). Forwarding (NO NOISE).")
                            # We just forward the original data
                            message_to_forward = data

                        elif type_bytes == b"RAW":
                            print(f"Received RAW message (Size: {msg_len}). Forwarding (NO NOISE).")
                            # Forward the entire original message as-is
                            message_to_forward = data

                        else:
                            print(f"Unknown message type: {type_bytes}. Skipping.")
                            continue
                            
                    except Exception as e:
                        print(f"Error splitting message: {e}. Skipping.")
                        continue

                    # 3. Forward the message to the receiver
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
                            receiver_socket.connect((RECEIVER_HOST, RECEIVER_PORT))
                            receiver_socket.sendall(message_to_forward)
                    except Exception as e:
                        print(f"Channel error forwarding to receiver: {e}")
                        
        except Exception as e:
            print(f"Error in main connection loop: {e}. Resetting...")
            time.sleep(1)