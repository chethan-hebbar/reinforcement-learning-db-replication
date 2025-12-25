import json
import matplotlib.pyplot as plt

def plot_comparison(static_file, rl_file):
    """
    Loads evaluation results from two JSON files and generates
    comparative plots for latency and cost.
    """
    with open(static_file, 'r') as f:
        static_data = json.load(f)
    
    with open(rl_file, 'r') as f:
        rl_data = json.load(f)

    static_time = [item['time'] / 60 for item in static_data]
    static_latency = [item['avg_latency'] for item in static_data]
    static_cost = [item['total_cost'] for item in static_data]

    rl_time = [item['time'] / 60 for item in rl_data]
    rl_latency = [item['avg_latency'] for item in rl_data]
    rl_cost = [item['total_cost'] for item in rl_data]

    # Average Read Latency Comparison ---
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(static_time, static_latency, label='Static Policy (Baseline)', color='red', linestyle='--')
    ax1.plot(rl_time, rl_latency, label='RL Agent Policy (AI)', color='blue', linewidth=2)
    
    ax1.set_xlabel('Time (minutes)', fontsize=14)
    ax1.set_ylabel('Average Read Latency (ms)', fontsize=14)
    ax1.set_title('AI Agent vs. Static Policy: Read Latency Comparison', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True)
    
    latency_plot_filename = 'latency_comparison.png'
    plt.savefig(latency_plot_filename)
    print(f"Latency comparison plot saved as '{latency_plot_filename}'")
    plt.close()

    # Total System Cost Comparison ---

    fig, ax2 = plt.subplots(figsize=(12, 7))

    ax2.plot(static_time, static_cost, label='Static Policy (Baseline)', color='red', linestyle='--')
    ax2.plot(rl_time, rl_cost, label='RL Agent Policy (AI)', color='blue', linewidth=2)

    ax2.set_xlabel('Time (minutes)', fontsize=14)
    ax2.set_ylabel('Total Storage Cost ($)', fontsize=14)
    ax2.set_title('AI Agent vs. Static Policy: System Cost Comparison', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True)

    cost_plot_filename = 'cost_comparison.png'
    plt.savefig(cost_plot_filename)
    print(f"Cost comparison plot saved as '{cost_plot_filename}'")
    plt.close()


if __name__ == "__main__":
    static_results_file = "evaluation_results_static.json"
    rl_results_file = "evaluation_results_rl.json"
    print("Generating comparison plots...")
    plot_comparison(static_results_file, rl_results_file)
    print("Done.")