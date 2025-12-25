import json
import matplotlib.pyplot as plt
import os

def plot_gnn_comparison():
    with open('evaluation_results_gnn.json', 'r') as f:
        gnn_data = json.load(f)

    gnn_time = [item['time'] / 60 for item in gnn_data]
    gnn_latency = [item['avg_latency'] for item in gnn_data]
    gnn_cost = [item['total_cost'] for item in gnn_data]

    static_exists = os.path.exists('evaluation_results_static.json')
    if static_exists:
        with open('evaluation_results_static.json', 'r') as f:
            static_data = json.load(f)
        static_time = [item['time'] / 60 for item in static_data]
        static_latency = [item['avg_latency'] for item in static_data]
        static_cost = [item['total_cost'] for item in static_data]
    else:
        print("Warning: 'evaluation_results_static.json' not found. Plotting GNN only.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    if static_exists:
        ax1.plot(static_time, static_cost, label='Static Policy (Baseline)', color='red', linestyle='--')
    
    ax1.plot(gnn_time, gnn_cost, label='GNN Agent (AI)', color='green', linewidth=2)
    
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Total Storage Cost ($)', fontsize=12)
    ax1.set_title('GNN Agent: System Cost', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True)
    plt.savefig('gnn_cost_comparison.png')
    plt.close()

    fig, ax2 = plt.subplots(figsize=(12, 6))

    if static_exists:
        ax2.plot(static_time, static_latency, label='Static Policy (Baseline)', color='red', linestyle='--')

    ax2.plot(gnn_time, gnn_latency, label='GNN Agent (AI)', color='green', linewidth=2)

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Avg Read Latency (ms)', fontsize=12)
    ax2.set_title('GNN Agent: Read Latency', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True)
    plt.savefig('gnn_latency_comparison.png')
    plt.close()

    print("Plots saved: gnn_cost_comparison.png, gnn_latency_comparison.png")

if __name__ == "__main__":
    plot_gnn_comparison()