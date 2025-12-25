import requests
import time
import random
import numpy as np
import itertools
import argparse

CONTROLLER_WRITE_URL = "http://localhost:8080/api/v1/data"

# Reads go directly to the Regional Nodes (simulating Geo-DNS/Edge Access)
# These ports match your docker-compose.yml external ports
REGION_NODES = {
    "us-east": "http://localhost:8081",
    "eu-west": "http://localhost:8082",
    "ap-south": "http://localhost:8083",
    "sa-east":  "http://localhost:8084", # New
    "jp-east":  "http://localhost:8085"  # New
}

KEYS = [f"user_profile_{i}" for i in range(20)]
REGIONS = list(REGION_NODES.keys())

PHASE_DURATION_SECONDS = 10

# Helper to generate skewed profiles automatically
def generate_skewed_profile(hot_key_indices, hot_region_indices, read_ratio=0.9):
    num_keys = len(KEYS)
    num_regions = len(REGIONS)
    
    if len(hot_key_indices) == num_keys:
        key_probs = np.full(num_keys, 1.0 / num_keys)
    else:
        # Distribute 20% noise to cold keys
        key_probs = np.full(num_keys, 0.2 / (num_keys - len(hot_key_indices)))
        # Distribute 80% traffic to hot keys
        for idx in hot_key_indices:
            key_probs[idx] = 0.8 / len(hot_key_indices)
    
    # If all regions are hot, distribute evenly.
    if len(hot_region_indices) == num_regions:
        region_probs = np.full(num_regions, 1.0 / num_regions)
    else:
        # Distribute 20% noise to cold regions
        region_probs = np.full(num_regions, 0.2 / (num_regions - len(hot_region_indices)))
        # Distribute 80% traffic to hot regions
        for idx in hot_region_indices:
            region_probs[idx] = 0.8 / len(hot_region_indices)

    return {
        "key_distribution": key_probs / key_probs.sum(),
        "region_distribution": region_probs / region_probs.sum(),
        "read_write_ratio": read_ratio
    }

# Define dynamic profiles (Randomized Logic)
ALL_PROFILES = [
    # 1. Keys 0-3 hot in US
    generate_skewed_profile(range(0, 4), [0]), 
    # 2. Keys 4-7 hot in EU
    generate_skewed_profile(range(4, 8), [1]),
    # 3. Keys 8-11 hot in AP
    generate_skewed_profile(range(8, 12), [2]),
    # 4. Keys 12-15 hot in SA (New region index 3)
    generate_skewed_profile(range(12, 16), [3]),
    # 5. Keys 16-19 hot in JP (New region index 4)
    generate_skewed_profile(range(16, 20), [4]),
    # 6. Global Chaos (All regions, random keys)
    generate_skewed_profile([0, 5, 10, 15], [0, 1, 2, 3, 4])
]

def send_write_request(key, value):
    """Sends write to Controller to broadcast/seed."""
    try:
        payload = {"key": key, "value": value}
        response = requests.post(CONTROLLER_WRITE_URL, json=payload, timeout=2)
        print(f"WRITE -> Key: {key}, Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"WRITE -> Key: {key}, FAILED: {e}")

def send_read_request(key, region):
    """Sends read directly to the specific Regional Node."""
    node_url = REGION_NODES.get(region)
    if not node_url: return

    try:
        # Hitting the node directly: GET /data/{key}
        # This triggers the 'handleGet' logic in Java, incrementing the local read count
        # even if the data is missing (simulated miss).
        response = requests.get(f"{node_url}/data/{key}", timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            # If 'value' is null, it was a miss (remote fetch simulation)
            hit_status = "HIT" if data.get('value') else "MISS"
            latency = data.get('latencyMs')
            print(f"READ ({region}) -> Key: {key}, {hit_status}, Latency: {latency}ms")
        else:
            print(f"READ ({region}) -> Key: {key}, Status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"READ ({region}) -> Key: {key}, FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], 
                        help="train=Cyclic patterns, test=Random patterns")
    args = parser.parse_args()

    print("--- Starting Dynamic Workload Generator (Direct Access Mode) ---")

    # Initial data seeding
    print("Seeding initial data...")
    for key in KEYS:
        send_write_request(key, f"initial_value_for_{key}")

    start_time = time.time()
    
    if args.mode == "train":
        # Cyclic
        profile_iterator = itertools.cycle(ALL_PROFILES)
        current_profile = next(profile_iterator)
    else:
        # Test: Random
        current_profile = random.choice(ALL_PROFILES)

    print(f"\n--- Switched to new workload profile ---")

    while True:
        # Check if it's time to switch profiles
        if time.time() - start_time > PHASE_DURATION_SECONDS:
            start_time = time.time()

            if args.mode == "train":
                current_profile = next(profile_iterator)
                print(f"\n--- Switched to NEXT Cyclic Profile ---")
            else:
                current_profile = random.choice(ALL_PROFILES)
                print(f"\n--- Switched to RANDOM Test Profile ---")

        # Choose parameters based on profile
        key_to_access = np.random.choice(KEYS, p=current_profile["key_distribution"])
        user_region = np.random.choice(REGIONS, p=current_profile["region_distribution"])
        is_read_action = random.random() < current_profile["read_write_ratio"]

        print(f"User from '{user_region}' is accessing data...")
        
        if is_read_action:
            # READ: Go to local node
            send_read_request(key_to_access, user_region)
        else:
            # WRITE: Go to controller
            new_value = f"val_{random.randint(1000, 9999)}"
            send_write_request(key_to_access, new_value)

        # Rate limiting
        time.sleep(0.5)