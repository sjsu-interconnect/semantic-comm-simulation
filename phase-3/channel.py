import socket
import numpy as np
import time
import struct
import threading
import random

SENDER_HOST = '0.0.0.0'
SENDER_PORT = 65431
RECEIVER_HOST = 'receiver'
RECEIVER_PORT = 65432
VECTOR_SIZE = 512

# --- NEW IN PHASE 3: DynamicChannel Class ---
class DynamicChannel:
    def __init__(self):
        # Initial network state
        self.current_noise = 0.05
        self.current_bandwidth = 10.0  # Mbps (Megabits per second)
        
        # Start the thread to update the state
        self.state_thread = threading.Thread(target=self.update_channel_state, daemon=True)
        self.state_thread.start()

    def update_channel_state(self):
        """
        Periodically updates the channel's noise and bandwidth
        using a simple random walk.
        """
        print("[Channel State] Dynamic state update thread started.")
        while True:
            # Update Noise (random walk, bounded between 0.0 and 0.5)
            noise_change = random.uniform(-0.01, 0.01)
            self.current_noise = np.clip(self.current_noise + noise_change, 0.0, 0.5)

            # Update Bandwidth (random walk, bounded between 1 Mbps and 20 Mbps)
            bw_change = random.uniform(-1.0, 1.0)
            self.current_bandwidth = np.clip(self.current_bandwidth + bw_change, 1.0, 20.0)
            
            # print(f"[Channel State] New State: Noise={self.current_noise:.3f}, BW={self.current_bandwidth:.2f} Mbps")
            time.sleep(3) # Update state every 3 seconds

    def add_noise(self, vector_bytes):
        """Deserializes, adds noise based on current_noise, and re-serializes."""
        try:
            original_vector = np.frombuffer(vector_bytes, dtype=np.float32)
            if original_vector.shape[0] != VECTOR_SIZE:
                 print(f"  [Noise] Error: Expected vector size {VECTOR_SIZE}, got {original_vector.shape[0]}")
                 return vector_bytes # Return original
            
            # Use the dynamic noise level
            noise = np.random.normal(0, self.current_noise, original_vector.shape)
            noisy_vector = original_vector + noise
            return noisy_vector.tobytes()
        except Exception as e:
            print(f"  [Noise] Error: {e}. Returning original vector.")
            return vector_bytes
            
    def recv_all(self, conn, n):
        """Helper function to receive exactly n bytes."""
        buffer = b""
        while len(buffer) < n:
            chunk = conn.recv(n - len(buffer))
            if not chunk:
                return None
            buffer += chunk
        return buffer

    def run(self):
        """Main server loop."""
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
                            header_bytes = self.recv_all(conn, 4)
                            if not header_bytes:
                                print(f"Sender {addr} disconnected (header read).")
                                break
                            
                            msg_len = struct.unpack('!I', header_bytes)[0]
                            data = self.recv_all(conn, msg_len)
                            if not data:
                                print(f"Sender {addr} disconnected (payload read).")
                                break
                            
                            # --- DYNAMIC LOGIC ---
                            # 1. Get current state and simulate delay
                            noise_level = self.current_noise
                            bandwidth_mbps = self.current_bandwidth
                            
                            # Calculate delay
                            # msg_len is in bytes. Bandwidth is in Megabits/s
                            msg_len_megabits = (msg_len * 8) / 1_000_000
                            delay_sec = msg_len_megabits / bandwidth_mbps
                            
                            print(f"Received msg (Size: {msg_len} B). BW: {bandwidth_mbps:.2f} Mbps. Delaying for {delay_sec:.4f}s.")
                            time.sleep(delay_sec) # Simulate transmission time
                            
                            # 2. Process message
                            try:
                                type_bytes, rest_of_data = data.split(b'|', 1)
                                
                                if type_bytes == b"SEM":
                                    print(f"  Applying noise: {noise_level:.3f}")
                                    ts_bytes, label_and_vec = rest_of_data.split(b'|', 1)
                                    label_bytes, vector_bytes = label_and_vec.split(b'|', 1)
                                    
                                    noisy_vector_bytes = self.add_noise(vector_bytes)
                                    
                                    original_message = type_bytes + b'|' + ts_bytes + b'|' + label_bytes + b'|' + noisy_vector_bytes

                                elif type_bytes == b"RAW":
                                    print("  Forwarding RAW (no noise).")
                                    original_message = data
                                else:
                                    print(f"Unknown message type: {type_bytes}. Skipping.")
                                    continue
                                    
                            except Exception as e:
                                print(f"Error splitting message: {e}. Skipping.")
                                continue

                            # 3. Prepend network state and forward
                            noise_bytes = np.array([noise_level], dtype=np.float32).tobytes()
                            bw_bytes = np.array([bandwidth_mbps], dtype=np.float32).tobytes()
                            
                            # NEW FORWARD FORMAT: noise | bandwidth | original_message
                            message_to_forward = noise_bytes + b'|' + bw_bytes + b'|' + original_message

                            try:
                                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rs:
                                    rs.connect((RECEIVER_HOST, RECEIVER_PORT))
                                    rs.sendall(message_to_forward)
                            except Exception as e:
                                print(f"Channel error forwarding to receiver: {e}")
                                
                except Exception as e:
                    print(f"Error in main connection loop: {e}. Resetting...")
                    time.sleep(1)

if __name__ == "__main__":
    channel = DynamicChannel()
    channel.run()