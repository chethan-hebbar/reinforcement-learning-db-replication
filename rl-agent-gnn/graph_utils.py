import torch
import numpy as np

def parse_system_state_to_graph(state_json):
    """
    Converts JSON to Graph Tensors.
    INCLUDES NORMALIZATION to prevent NaN in training.
    """
    if not state_json:
        # Return valid empty structures to prevent model crashes
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            []
        )

    # Identify Unique Keys and Servers
    key_names = sorted(list(set(k for node in state_json for k in node.get('keyMetrics', {}).keys())))
    server_ids = [node['nodeId'] for node in state_json]

    num_keys = len(key_names)
    num_servers = len(server_ids)

    # Build Node Features
    x_keys = np.zeros((num_keys, 3), dtype=np.float32)
    x_servers = np.zeros((num_servers, 2), dtype=np.float32)

    key_stats = {k: {'reads': 0, 'writes': 0} for k in key_names}

    for s_idx, node_data in enumerate(state_json):
        # Server Feat: [Cost, Capacity]
        # Normalize Cost (approx range 0-2) -> OK
        # Normalize Capacity (0-1) -> OK
        x_servers[s_idx] = [node_data.get('storageCost', 1.0), 0.5]
        
        for k, metrics in node_data.get('keyMetrics', {}).items():
            key_stats[k]['reads'] += metrics.get('readCount', 0)
            key_stats[k]['writes'] += metrics.get('writeCount', 0)

    for i, k in enumerate(key_names):
        # Key Feat: [Global_Reads, Global_Writes, Size]
        reads = np.log1p(key_stats[k]['reads'])
        writes = np.log1p(key_stats[k]['writes'])
        x_keys[i] = [reads, writes, 1.0]

    # Build Edges
    src_indices = []
    dst_indices = []
    edge_features = []

    for s_idx, node_data in enumerate(state_json):
        for k, metrics in node_data.get('keyMetrics', {}).items():
            if k in key_names:
                k_idx = key_names.index(k)
                
                src_indices.append(k_idx)
                dst_indices.append(s_idx)
                
                # Edge Feat: [Local_Reads, Is_Present]
                local_reads = np.log1p(metrics.get('readCount', 0))
                edge_features.append([local_reads, 1.0])

    edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
    edge_attr = np.array(edge_features, dtype=np.float32)

    if len(edge_features) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 2), dtype=np.float32)

    return x_keys, x_servers, edge_index, edge_attr, key_names