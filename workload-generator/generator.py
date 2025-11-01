import requests
import time
import random
import numpy as np
import itertools

CONTROLLER_URL = "http://localhost:8080/api/v1/data"
KEYS = [f"user_profile_{i}" for i in range(5)]
REGIONS = ["us-east", "eu-west", "ap-south"]
PHASE_DURATION_SECONDS = 5

# Profile 1: Key 0 is viral in the US
profile_1 = {
    "key_distribution": [0.80, 0.05, 0.05, 0.05, 0.05],
    "region_distribution": [0.7, 0.2, 0.1],
    "read_write_ratio": 0.95
}

# Profile 2: Key 3 becomes popular in Europe
profile_2 = {
    "key_distribution": [0.05, 0.05, 0.1, 0.75, 0.05],
    "region_distribution": [0.1, 0.8, 0.1],
    "read_write_ratio": 0.90
}

# Profile 3: Key 1 is now the focus, primarily in Asia-Pacific
profile_3 = {
    "key_distribution": [0.05, 0.80, 0.05, 0.05, 0.05],
    "region_distribution": [0.1, 0.2, 0.7],
    "read_write_ratio": 0.95
}

# Profile 4: Key 4 is read-heavy, globally distributed
profile_4 = {
    "key_distribution": [0.05, 0.05, 0.05, 0.05, 0.80],
    "region_distribution": [0.3, 0.4, 0.3],
    "read_write_ratio": 0.98
}

# Profile 5: Key 2 has a spike in writes from the US (e.g., a software update)
profile_5 = {
    "key_distribution": [0.05, 0.05, 0.80, 0.05, 0.05],
    "region_distribution": [0.7, 0.1, 0.2],
    "read_write_ratio": 0.60
}

# Profile 6: A "balanced" or "chaotic" state with no single dominant key
profile_6 = {
    "key_distribution": [0.25, 0.15, 0.20, 0.20, 0.20],
    "region_distribution": [0.33, 0.34, 0.33],
    "read_write_ratio": 0.90
}

# cycle through the profiles
WORKLOAD_PROFILES = itertools.cycle([
    profile_1, profile_2, profile_3, profile_4, profile_5, profile_6
])



def send_write_request(key, value):
    try:
        payload = {"key": key, "value": value}
        response = requests.post(CONTROLLER_URL, json=payload, timeout=2)
        print(f"WRITE -> Key: {key}, Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"WRITE -> Key: {key}, FAILED: {e}")

def send_read_request(key):
    try:
        response = requests.get(f"{CONTROLLER_URL}/{key}", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"READ  -> Key: {key}, Latency: {data.get('retrievalLatencyMs')}ms, Node: {data.get('servedByNode')}")
        else:
            print(f"READ  -> Key: {key}, Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"READ  -> Key: {key}, FAILED: {e}")


if __name__ == "__main__":
    print("--- Starting Dynamic Workload Generator ---")

    # Initial data seeding
    print("Seeding initial data...")
    for key in KEYS:
        send_write_request(key, f"initial_value_for_{key}")

    start_time = time.time()
    current_profile = next(WORKLOAD_PROFILES)
    print(f"\n--- Switched to new workload profile ---")

    while True:
        if time.time() - start_time > PHASE_DURATION_SECONDS:
            start_time = time.time()
            current_profile = next(WORKLOAD_PROFILES)
            print(f"\n--- Switched to new workload profile ---")

        key_to_access = np.random.choice(KEYS, p=current_profile["key_distribution"])
        user_region = np.random.choice(REGIONS, p=current_profile["region_distribution"])
        is_read_action = random.random() < current_profile["read_write_ratio"]

        print(f"User from '{user_region}' is accessing data...")
        if is_read_action:
            send_read_request(key_to_access)
        else:
            new_value = f"updated_value_{random.randint(1000, 9999)}"
            send_write_request(key_to_access, new_value)

        time.sleep(1)