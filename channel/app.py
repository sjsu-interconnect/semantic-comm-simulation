import socket
import numpy as np
import time
import struct
import threading
import random
import pickle
import logging
from typing import Optional, Dict, Any

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [Channel] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration Constants ---
DEFAULT_SENDER_HOST = '0.0.0.0'
DEFAULT_SENDER_PORT = 65431
DEFAULT_RECEIVER_HOST = 'receiver'
DEFAULT_RECEIVER_PORT = 65432
DEFAULT_VECTOR_SIZE = 512


class DynamicChannel:
    """
    Simulates a dynamic network channel that sits between a sender and receiver.
    
    It accepts persistent connections from a sender and forwards messages
    one-by-one to a receiver, simulating network latency and noise.
    """

    def __init__(self,
                 sender_host: str = DEFAULT_SENDER_HOST,
                 sender_port: int = DEFAULT_SENDER_PORT,
                 receiver_host: str = DEFAULT_RECEIVER_HOST,
                 receiver_port: int = DEFAULT_RECEIVER_PORT,
                 vector_size: int = DEFAULT_VECTOR_SIZE):
        
        # --- Network Configuration ---
        self.sender_host = sender_host
        self.sender_port = sender_port
        self.receiver_host = receiver_host
        self.receiver_port = receiver_port
        self.vector_size = vector_size

        # --- Dynamic State ---
        self.current_noise: float = 0.05
        self.current_bandwidth: float = 10.0  # Mbps
        
        # --- Threading ---
        self.state_thread = threading.Thread(target=self.update_channel_state, daemon=True)

    def update_channel_state(self):
        """
        Periodically updates the channel's noise and bandwidth in a separate thread.
        Uses a simple random walk, bounded by np.clip.
        Also configures the Docker container's eth0 interface using `tc` to simulate bandwidth and jitter.
        """
        import subprocess

        logging.info("Dynamic state update thread started.")
        
        # Initialize tc qdisc (clear any existing, then add root tbf)
        try:
            subprocess.run(["tc", "qdisc", "del", "dev", "eth0", "root"], check=False, stderr=subprocess.DEVNULL)
            # Add a basic Token Bucket Filter just to get started. We'll replace it in the loop.
            subprocess.run(["tc", "qdisc", "add", "dev", "eth0", "root", "handle", "1:", "tbf", "rate", f"{self.current_bandwidth}mbit", "burst", "32kbit", "latency", "400ms"], check=True)
            # Add NetEm under the TBF to simulate the jitter
            subprocess.run(["tc", "qdisc", "add", "dev", "eth0", "parent", "1:1", "handle", "10:", "netem", "delay", "30ms", "20ms"], check=True)
        except Exception as e:
            logging.error(f"Failed to initialize tc: {e} (Are you missing NET_ADMIN capability?)")

        while True:
            # Random walk for Noise
            noise_change = random.uniform(-0.1, 0.1)
            self.current_noise = np.clip(self.current_noise + noise_change, 0.0, 0.5)
            
            # Random walk for Bandwidth
            bw_change = random.uniform(-5.0, 5.0)
            self.current_bandwidth = np.clip(self.current_bandwidth + bw_change, 1.0, 20.0)
            
            # Jitter variation
            base_delay_ms = int(random.uniform(10, 50))
            jitter_ms = int(random.uniform(5, base_delay_ms/2))

            try:
                # Update TBF (Bandwidth)
                subprocess.run(["tc", "qdisc", "change", "dev", "eth0", "root", "handle", "1:", "tbf", "rate", f"{self.current_bandwidth:.2f}mbit", "burst", "32kbit", "latency", "400ms"], check=True)
                # Update NetEm (Jitter)
                subprocess.run(["tc", "qdisc", "change", "dev", "eth0", "parent", "1:1", "handle", "10:", "netem", "delay", f"{base_delay_ms}ms", f"{jitter_ms}ms"], check=True)
            except Exception as e:
                logging.error(f"Failed to dynamically update tc: {e}")

            time.sleep(1) # Update state every 1 second

    def add_noise(self, vector: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian noise to a numpy vector based on the current noise level.
        """
        try:
            noise = np.random.normal(0, self.current_noise, vector.shape)
            noisy_vector = vector + noise
            return noisy_vector
        except Exception as e:
            logging.error(f"[Noise] Error: {e}. Returning original vector.")
            return vector

    def recv_all(self, conn: socket.socket, n_bytes: int) -> Optional[bytes]:
        """
        Helper function to receive exactly n_bytes from a socket.
        Returns None if the connection is closed before all bytes are received.
        """
        buffer = b""
        while len(buffer) < n_bytes:
            chunk = conn.recv(n_bytes - len(buffer))
            if not chunk:
                return None  # Connection dropped
            buffer += chunk
        return buffer

    def receive_message(self, conn: socket.socket) -> Optional[bytes]:
        """
        Receives one complete, length-framed message from the sender.
        Returns the pickled payload, or None on disconnect.
        """
        # 1. Read the 4-byte message length header
        header_bytes = self.recv_all(conn, 4)
        if not header_bytes:
            return None  # Client disconnected
        
        msg_len = struct.unpack('!I', header_bytes)[0]
        
        # 2. Read the full message payload
        payload = self.recv_all(conn, msg_len)
        if not payload:
            return None # Client disconnected
            
        return payload

    def process_and_simulate(self, pickled_payload: bytes) -> Optional[bytes]:
        """
        Applies noise to the message. Network latency (bandwidth/jitter) is now handled natively
        by Docker Linux Traffic Control (tc).
        Returns the final message to be forwarded.
        """
        msg_len = len(pickled_payload)
        noise_level = self.current_noise
        bandwidth_mbps = self.current_bandwidth
        
        logging.info(f"Received msg (Size: {msg_len} B). TC is enforcing BW: {bandwidth_mbps:.2f} Mbps natively.")
        
        # 2. Process the message payload
        try:
            message_dict: Dict[str, Any] = pickle.loads(pickled_payload)
            
            if message_dict['type'] in ["SEM", "SEM_LOCAL", "SEM_EDGE"]:
                logging.info(f"  Applying noise: {noise_level:.3f}")
                noisy_vector = self.add_noise(message_dict['payload'])
                message_dict['payload'] = noisy_vector

            elif message_dict['type'] == "RAW":
                logging.info("  Forwarding RAW (no noise).")
                # No changes needed
            
            else:
                logging.warning(f"Unknown message type: {message_dict['type']}. Skipping.")
                return None
                
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            logging.error(f"Error unpickling/processing message: {e}. Skipping.")
            return None

        # 3. Re-pickle the (possibly modified) message
        message_payload_bytes = pickle.dumps(message_dict)

        # 4. Prepend network state for the receiver
        noise_bytes = np.array([noise_level], dtype=np.float32).tobytes()
        bw_bytes = np.array([bandwidth_mbps], dtype=np.float32).tobytes()
        
        # NEW FORMAT: noise (4b) | bandwidth (4b) | pickled_payload
        message_to_forward = noise_bytes + bw_bytes + message_payload_bytes
        
        return message_to_forward

    def forward_message(self, message_to_forward: bytes):
        """
        Opens a new, one-time connection to the receiver and sends the message.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rs:
                rs.connect((self.receiver_host, self.receiver_port))
                rs.sendall(message_to_forward)
        except socket.error as e:
            logging.error(f"Error forwarding to receiver: {e}")

    def handle_client(self, conn: socket.socket, addr):
        """
        Manages the entire lifecycle of a single connected sender.
        """
        logging.info(f"Sender connected from {addr}")
        try:
            while True:
                # 1. Receive one full message from the sender
                pickled_payload = self.receive_message(conn)
                if not pickled_payload:
                    logging.info(f"Sender {addr} disconnected.")
                    break
                
                # 2. Process, simulate delay/noise, and get final message
                message_to_forward = self.process_and_simulate(pickled_payload)
                if not message_to_forward:
                    continue # Skip this message
                
                # 3. Forward the message to the receiver
                self.forward_message(message_to_forward)
                
        except socket.error as e:
            logging.warning(f"Socket error with sender {addr}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error handling client {addr}: {e}")
        finally:
            logging.info(f"Closing connection from {addr}.")
            conn.close()

    def run(self):
        """
        Starts the main server loop to listen for sender connections.
        """
        # Start the state update thread
        self.state_thread.start()
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.sender_host, self.sender_port))
                s.listen()
                logging.info(f"Channel is permanently listening on {self.sender_host}:{self.sender_port}...")

                while True:
                    logging.info("Waiting for a new sender connection...")
                    try:
                        conn, addr = s.accept()
                        # Pass the new connection to a handler.
                        # We don't use a new thread here because the DRL loop
                        # is sequential (step-by-step), but this is possible.
                        self.handle_client(conn, addr)
                        
                    except Exception as e:
                        logging.error(f"Error accepting connection: {e}. Resetting...")
                        time.sleep(1)
        except OSError as e:
            logging.critical(f"Failed to bind socket: {e}")

if __name__ == "__main__":
    # Create the channel with the default configuration
    channel = DynamicChannel(
        sender_host=DEFAULT_SENDER_HOST,
        sender_port=DEFAULT_SENDER_PORT,
        receiver_host=DEFAULT_RECEIVER_HOST,
        receiver_port=DEFAULT_RECEIVER_PORT,
        vector_size=DEFAULT_VECTOR_SIZE
    )
    # Start the server
    channel.run()