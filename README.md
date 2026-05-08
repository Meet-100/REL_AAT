# Adaptive Water Distribution Optimization using Reinforcement Learning

## Project Overview
This project implements a **Reinforcement Learning (RL)** based smart water distribution system. It optimizes the allocation of limited water resources across three zones (Zone A, Zone B, Zone C) from a central tank, aiming to minimize water shortage and wastage.

### SDG Connection: SDG 6 – Clean Water and Sanitation
By optimizing water allocation, this project contributes to **Sustainable Development Goal 6** by ensuring efficient water management and reducing wastage in resource-constrained environments.

---

## 🛠️ Project Architecture

### 1. Simulator Environment (`sim/water_env.py`)
- **3 Zones**: Each generates random water demand at every timestep.
- **Central Tank**: Has a fixed capacity and receives a periodic refill.
- **State Space**: Discretized (Tank Level, Demand A, Demand B, Demand C).
- **Action Space**:
  - `Equal Distribution`: Divide water equally.
  - `Prioritize Zone A/B/C`: Give priority to a specific zone.

### 2. RL Methodology (`agents/qlearning_agent.py`)
- **Algorithm**: Q-Learning (Tabular).
- **Strategy**: Epsilon-greedy exploration with decay.
- **Reward Function**: `reward = -(Total Shortage + Total Wastage)`.

### 3. MLOps Workflow
- **Reproducibility**: Controlled via `configs/qlearning.yaml`.
- **Experiment Tracking**: All runs are logged in `results/experiments.csv`.
- **Versioning**: Trained models are saved in `models/` (e.g., `policy_v1.pkl`).

---

## 🚀 How to Run

### 1. Installation
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training
Train the Q-learning agent:
```bash
python train.py
```
This will:
- Train the agent for 1000 episodes (configurable).
- Save the trained policy to `models/policy_v1.pkl`.
- Log results to `results/training_logs.csv`.

### 3. Evaluation
Compare the RL agent against the baseline:
```bash
python evaluate.py
```
This will:
- Run test episodes for both RL and Fixed-Distribution baseline.
- Generate comparison plots in `results/plots/`.
- Generate a summary report in `results/evaluation_report.md`.

---

## 📊 Monitoring Plan (Real-world Deployment)
In a real-world scenario, the following metrics should be monitored continuously:
- **Water Shortages**: Frequency and severity of unmet demand.
- **Wastage**: Tank overflows or over-allocation.
- **Sensor Health**: Monitoring tank level and flow sensors for failures.
- **Leakage Detection**: Discrepancies between tank outflow and zone inflow.
- **Fairness**: Ensuring no single zone is consistently deprived of water.

---

## 📁 Folder Structure
```text
water-rl-project/
├── agents/             # RL Agent logic (Q-Learning)
├── sim/                # Environment simulator
├── configs/            # YAML configurations
├── models/             # Saved policy files (.pkl)
├── results/            # Logs, CSVs, and plots
│   └── plots/          # Visualizations
├── experiments/        # Manual experiment notes (if any)
├── utils/              # Helper utilities
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---

## 🏷️ Versioning & Tags
For academic evaluation, use Git tags to mark milestones:
- `exp-qlearning-1`: Initial baseline model.
- `exp-qlearning-2`: Optimized hyperparameters.

## ⚠️ Limitations & Future Work
- **Discrete State Space**: The current model uses discretization. Future versions could use Deep Q-Networks (DQN) for continuous states.
- **Static Pricing**: Future versions could incorporate dynamic water pricing based on demand.
- **External Factors**: Rainfall and weather forecasts could be added as state inputs.
