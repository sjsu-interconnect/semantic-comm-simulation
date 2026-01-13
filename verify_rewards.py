
import numpy as np
import io
import pickle
from PIL import Image
import time

# --- Mock Constants ---
SIM_ENC_SEMANTIC_LOCAL = 0.200
SIM_ENC_SEMANTIC_EDGE = 0.020
SIM_ENC_RAW = 0.005
SIM_DEC_SEMANTIC = 0.050
SIM_DEC_RAW = 0.010

EDGE_QUALITY_MULTIPLIER = 0.5

# NEW WEIGHTS (AGGRESSIVE)
REWARD_ALPHA = 20.0 
LATENCY_PENALTY_FACTOR = 2.0
DEADLINE_TAU = 1.0
MISSED_DEADLINE_PENALTY = 10.0
RAW_NOISE_SENSITIVITY = 10.0

# --- Helper Functions ---
def get_cifar_size():
    # 32x32 image approx 645 bytes
    # BUT we added 150KB padding!
    return 645 + (150 * 1024)

def get_vector_size():
    # 512 floats approx 2199 bytes
    return 2199

def calculate_reward(latency, mse):
    latency_met = latency <= DEADLINE_TAU
    binary_penalty = 0.0 if latency_met else MISSED_DEADLINE_PENALTY
    
    # Cost = Binary + Linear + MSE
    cost = binary_penalty + (latency * LATENCY_PENALTY_FACTOR) + (REWARD_ALPHA * mse)
    return -cost

def simulate_step(bandwidth_mbps, noise_level, action_name):
    """
    Returns (latency, mse, reward)
    """
    
    # 1. Payload Size
    if action_name == "RAW":
        payload_size_bytes = get_cifar_size()
    else: # SEM_LOCAL or SEM_EDGE
        payload_size_bytes = get_vector_size()
        
    # Add overhead (approx)
    msg_size_bytes = payload_size_bytes + 100 # Headers
    
    # 2. Network Latency
    msg_size_bits = msg_size_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    net_delay = msg_size_bits / bandwidth_bps
    
    # 3. Processing Latency & Specific Logic
    mse = 0.0
    
    if action_name == "RAW":
        proc_delay = SIM_ENC_RAW + SIM_DEC_RAW
        mse = 0.0 # Lossless base
        if noise_level > 0:
            mse += RAW_NOISE_SENSITIVITY * (noise_level ** 2)
        
    elif action_name == "SEM_LOCAL":
        proc_delay = SIM_ENC_SEMANTIC_LOCAL + SIM_DEC_SEMANTIC
        mse = 0.05 + (noise_level ** 2)
        
    elif action_name == "SEM_EDGE":
        # Edge Upload Delay (50KB)
        edge_upload_bits = 50 * 1024 * 8
        bw_bps = max(1.0, bandwidth_mbps) * 1_000_000
        upload_delay = edge_upload_bits / bw_bps
        
        proc_delay = SIM_ENC_SEMANTIC_EDGE + upload_delay + SIM_DEC_SEMANTIC
        
        # Edge Quality Benefit
        base_mse = 0.05 + (noise_level ** 2)
        mse = base_mse * EDGE_QUALITY_MULTIPLIER
        
    # Calculate total latency outside the blocks
    total_latency = net_delay + proc_delay
    
    reward = calculate_reward(total_latency, mse)
    
    return payload_size_bytes, total_latency, mse, reward

# --- Main Verification Loop ---
print(f"{'Condition':<30} | {'Action':<10} | {'Size(B)':<7} | {'Lat(s)':<8} | {'MSE':<6} | {'Reward':<8}")
print("-" * 90)

scenarios = [
    ("High BW (20MB), Low Noise", 20.0, 0.0),
    ("Med BW (4MB), Low Noise", 4.0, 0.0),
    ("Low BW (1MB), Low Noise", 1.0, 0.0),
    ("High BW (20MB), High Noise", 20.0, 0.5),
]

for name, bw, noise in scenarios:
    for action in ["RAW", "SEM_LOCAL", "SEM_EDGE"]:
        size, lat, mse, reward = simulate_step(bw, noise, action)
        print(f"{name:<30} | {action:<10} | {size:<7} | {lat:.4f}   | {mse:.4f} | {reward:.4f}")
    print("-" * 90)
