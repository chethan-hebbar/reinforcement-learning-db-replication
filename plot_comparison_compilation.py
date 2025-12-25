import json
import matplotlib.pyplot as plt
import os

# --- Configuration: File Names ---
FILE_STATIC = 'evaluation_results_static.json'
FILE_MLP = 'evaluation_results_mlp.json'
FILE_GNN = 'evaluation_results_gnn.json'

def load_data(filepath):
    """Safe loading of JSON data."""
    if not os.path.exists(filepath):
        print(f"WARNING: File '{filepath}' not found. Skipping.")
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR reading '{filepath}': {e}")
        return None

def extract_metrics(data):
    """Helper to get X (minutes) and Y (metric) lists."""
    if not data: return [], [], []
    
    # Convert seconds to minutes for X-axis
    time_mins = [item['time'] / 60 for item in data]
    latency = [item['avg_latency'] for item in data]
    cost = [item['total_cost'] for item in data]
    return time_mins, latency, cost

def plot_comparison():
    # 1. Load Data
    print("Loading data...")
    static_data = load_data(FILE_STATIC)
    mlp_data = load_data(FILE_MLP)
    gnn_data = load_data(FILE_GNN)

    # 2. Extract Vectors
    t_static, lat_static, cost_static = extract_metrics(static_data)
    t_mlp, lat_mlp, cost_mlp = extract_metrics(mlp_data)
    t_gnn, lat_gnn, cost_gnn = extract_metrics(gnn_data)

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # ==========================================
    # PLOT 1: SYSTEM COST COMPARISON
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # Static (Baseline) - Red Dashed
    if static_data:
        ax1.plot(t_static, cost_static, label='Static Policy (Baseline)', 
                 color='red', linestyle='--', linewidth=2, alpha=0.7)

    # MLP (Vector Based) - Blue
    if mlp_data:
        ax1.plot(t_mlp, cost_mlp, label='MLP Agent (Fixed Vector)', 
                 color='blue', linewidth=2, alpha=0.8)

    # GNN (Graph Based) - Green (The "Hero" color)
    if gnn_data:
        ax1.plot(t_gnn, cost_gnn, label='GNN Agent (Graph Topology)', 
                 color='green', linewidth=2.5)

    ax1.set_xlabel('Time (minutes)', fontsize=13)
    ax1.set_ylabel('Total Storage Cost ($)', fontsize=13)
    ax1.set_title('Cost Efficiency: GNN vs. MLP vs. Static', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    save_path_cost = 'final_comparison_cost.png'
    plt.savefig(save_path_cost, dpi=300)
    print(f"Saved cost plot to: {save_path_cost}")
    plt.close()

    # ==========================================
    # PLOT 2: LATENCY COMPARISON
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # Static (Baseline)
    if static_data:
        ax2.plot(t_static, lat_static, label='Static Policy (Baseline)', 
                 color='red', linestyle='--', linewidth=2, alpha=0.7)

    # MLP Agent
    if mlp_data:
        ax2.plot(t_mlp, lat_mlp, label='MLP Agent (Fixed Vector)', 
                 color='blue', linewidth=2, alpha=0.8)

    # GNN Agent
    if gnn_data:
        ax2.plot(t_gnn, lat_gnn, label='GNN Agent (Graph Topology)', 
                 color='green', linewidth=2.5)

    ax2.set_xlabel('Time (minutes)', fontsize=13)
    ax2.set_ylabel('Average Read Latency (ms)', fontsize=13)
    ax2.set_title('Service Level (Latency): GNN vs. MLP vs. Static', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='best', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Optional: Set Y-limit if you want to zoom in on low latency
    # ax2.set_ylim(0, 160) 

    save_path_lat = 'final_comparison_latency.png'
    plt.savefig(save_path_lat, dpi=300)
    print(f"Saved latency plot to: {save_path_lat}")
    plt.close()

if __name__ == "__main__":
    plot_comparison()