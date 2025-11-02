# Adaptive Data Replication using Reinforcement Learning

This project implements a simulated geo-distributed database where a Reinforcement Learning (RL) agent learns to make intelligent, real-time decisions about data replication. The AI agent acts as an autonomous "Replication Manager," learning a dynamic policy to replicate and evict data based on changing global read/write patterns, with the goal of minimizing latency and storage costs simultaneously.

## What is it?

In a global distributed database, naive replication strategies are inefficient. Replicating all data everywhere guarantees low read latency but incurs maximum storage costs. Replicating nothing is cheap but results in high latency for users far from the data's source. This project explores an alternative: training an RL agent to learn a dynamic replication policy.

The system models a complex trade-off problem, and the RL agent successfully learns to navigate it, outperforming static, heuristic-based approaches.

### How?
*   **AI:** Reinforcement Learning (PPO), Custom RL Environment Design (Gymnasium), Dynamic Optimization, Neural Networks (PyTorch).
*   **Systems:** Geo-Distributed Systems, Microservices, Data Replication, Containerization (Docker, Docker Compose).
*   **Languages & Frameworks:** Java, Spring Boot, Python, Stable-Baselines3.

---

## System Architecture

The project is a multi-container Docker application composed of four main services:

![System Architecture Diagram](RL-DB-Replication.drawio.png)
*(A sample diagram showing the components and their interactions)*

1.  **Replication Controller (`replicationcontroller`):** The central brain. A Java/Spring Boot service that acts as a router for all client traffic and exposes an API for the RL agent to manage the cluster.
2.  **Database Nodes (`replication`):** Java/Spring Boot services representing database instances in different geographic regions (e.g., `us-east`, `eu-west`). They store data, track read/write metrics, and simulate latency and cost.
3.  **Workload Generator (`workload-generator`):** A Python script that simulates a dynamic, global user base sending a continuous and shifting stream of read/write requests to the controller.
4.  **RL Agent (`rl-agent`):** The intelligence. A Python application using Stable-Baselines3 that observes the system state via the controller, decides on an action (replicate/evict), and executes it.

---


## Results: The Power of an AI-Driven Policy

The core of this project is proving that a trained RL agent can manage a distributed system more effectively than a static policy. We evaluated our agent against a continous replication baseline under a highly volatile and dynamic workload that shifted its focus between different data keys and geographic regions every 10 seconds.

### Scenario 1: The "High-Performance" Agent
First, we trained an agent with the primary goal of minimizing read latency at all costs (`latency_weight=0.8`), while treating storage cost as a secondary objective.

#### System Cost: High-Performance Agent
![High-Performance Cost Comparison](/rl-agent/cost_comparison_hp.png)

#### Read Latency: High-Performance Agent
![High-Performance Latency Comparison](/rl-agent/latency_comparison_hp.png)

**Analysis:**
*   **Perfect Performance:** The AI agent (blue line) achieved a perfect 10ms average read latency, perfectly matching the "gold standard" performance of the expensive static baseline (red line).
*   **Intelligent Cost Savings:** Despite the chaotic workload, the agent intelligently identified and evicted "cold" data, consistently operating at a lower system cost than the baseline. It learned to pay the maximum price only when absolutely necessary to guarantee performance.

---

### Scenario 2: The "Cost-Conscious" Agent
Next, we retrained the agent with a new motivation: prioritize minimizing storage cost above all else (`cost_weight=0.7`). The results were impressive.

#### System Cost: Cost-Conscious Agent
![Cost-Conscious Cost Comparison](rl-agent/cost_comparison.png)

#### Read Latency: Cost-Conscious Agent
![Cost-Conscious Latency Comparison](rl-agent/latency_comparison.png)

**Analysis:**
*   **Optimal Performance, Drastically Lower Cost:** This agent discovered a more sophisticated policy. It managed to deliver a perfect 10ms average read latency, but it did so from a much lower and more stable cost baseline.
*   **Predictive & Surgical Actions:** The agent learned the workload's cyclical nature. The cost plot shows it performing precise, "just-in-time" replications to handle brief periods of high demand, immediately followed by evictions to return to its optimized, low-cost state.

**Conclusion:** We successfully demonstrate that a Reinforcement Learning agent can not only manage a complex distributed system but can be tuned to meet specific business objectives, discovering non-obvious, highly optimized policies that outperform static strategies.

## How to Run the Simulation

Follow these steps to reproduce the results.

### Prerequisites
*   Docker & Docker Compose
*   Java 25 & Maven
*   Python 3.11

### Step 1: Launch the Cluster
Navigate to the project root directory (where `docker-compose.yml` is located) and run:
```bash
docker compose up --build
```

### Step 2: Start the Workload
In a second terminal, navigate to the `workload-generator` directory, set up the environment, and run the script. This will seed the database and generate continuous traffic.
```bash
cd workload-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generator.py
```

### Step 3: Train the RL Agent
In a third terminal, navigate to the `rl-agent` directory, set up its environment, and run the training script. This will train the agent and save its policy to `ppo_replication_policy.zip`.
```bash
cd rl-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

### Step 4: Evaluate the Agent
After training, run the evaluation script. First, run the baseline, then the RL agent. **Remember to restart the Docker cluster between runs for a fair comparison.**

**Run the static baseline:**
```bash
# (Restart docker compose and workload generator)
python evaluate.py --mode static
```

**Run the trained RL agent:**
```bash
# (Restart docker compose and workload generator)
python evaluate.py --mode rl --model_path pppo_replication_policy.zip
```

### Step 5: Generate the Plots
Run the plotting script to generate the final comparison images from the evaluation data.
```bash
python plot_results.py
```

---