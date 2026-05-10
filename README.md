# Adaptive Water Distribution Optimization using Reinforcement Learning

> **A Reinforcement Learning-powered smart water distribution system designed to optimize resource allocation and minimize wastage, contributing to SDG 6.**

## 📋 1. Project Overview
This project addresses the challenge of distributing limited water resources from a central reservoir to multiple urban zones with fluctuating demands. Using **Reinforcement Learning**, the system learns an adaptive policy that prioritizes zones in need while maintaining overall network efficiency and minimizing overflow wastage.

## 🌍 2. SDG Mapping
- **SDG 6: Clean Water and Sanitation** - Specifically targeting Target 6.4: "Substantially increase water-use efficiency across all sectors and ensure sustainable withdrawals and supply of freshwater to address water scarcity."

## ❓ 3. Problem Statement
Traditional water distribution systems often rely on fixed, time-based, or equal-share allocation rules. These methods fail to adapt to real-time demand spikes or droughts in specific zones, leading to:
1.  **Water Shortage**: Critical unmet demand in high-usage areas.
2.  **Resource Wastage**: Excessive supply or tank overflow in low-usage areas.
3.  **Inefficiency**: Suboptimal utilization of central reservoir capacity.

## 🧠 4. RL Algorithm Used
We implemented **Q-Learning**, a model-free, off-policy Temporal Difference (TD) control algorithm.

## ⚖️ 5. Why Q-Learning?
Q-Learning was chosen because:
- **Discretized State Space**: The problem's state variables (tank level, demands) can be effectively binned, making a tabular approach computationally efficient.
- **Explainability**: The Q-table provides a transparent map of state-action values, which is essential for academic evaluation and safety audits in utility management.
- **Convergence**: It guarantees an optimal policy for finite MDPs given sufficient exploration.

## 📡 6. State Space
The situational state is represented as a tuple: `(tank_level, demand_A, demand_B, demand_C)`.

| Component | Range | Discretization (Bins) | Description |
|-----------|-------|----------------------|-------------|
| Tank Level| 0-100 | 5 | Empty (0), Low (1), Medium (2), High (3), Full (4) |
| Demand A  | 5-25  | 4 | Low (0) to Peak (3) request from Zone A |
| Demand B  | 5-25  | 4 | Low (0) to Peak (3) request from Zone B |
| Demand C  | 5-25  | 4 | Low (0) to Peak (3) request from Zone C |

## 🕹️ 7. Action Space
The agent can choose from 4 high-level allocation strategies:

| Action | Strategy | Implementation Logic |
|--------|----------|----------------------|
| **0** | Equal Distribution | Available water is split equally among all 3 zones. |
| **1** | Prioritize Zone A | Zone A receives full demand if possible; remainder split between B/C. |
| **2** | Prioritize Zone B | Zone B receives full demand if possible; remainder split between A/C. |
| **3** | Prioritize Zone C | Zone C receives full demand if possible; remainder split between A/B. |

## 🏆 8. Reward Function
The agent is trained to maximize a cumulative reward designed to penalize system failures:
**Reward = -(Total Shortage + Total Wastage)**
- **Shortage**: Sum of unmet demands across all zones (Priority #1).
- **Wastage**: Units lost to tank overflow during the refill phase (Priority #2).

## 🔦 9. Exploration Strategy
We use an **Epsilon-Greedy** strategy:
- **Epsilon ($\epsilon$)**: Starts at 1.0 (100% exploration).
- **Decay**: Multiplied by 0.995 each episode.
- **Minimum**: 0.01 (1% permanent exploration to ensure adaptability to demand shifts).

## 📁 10. Folder Structure
```text
water-rl-project/
├── agents/             # RL Agent implementation (QLearningAgent)
├── sim/                # Custom water distribution environment
├── configs/            # YAML files for reproducible hyperparams
├── logs/               # CSV and JSON experiment tracking
├── policies/           # Versioned models and metadata
├── results/            # Performance reports and plots
│   └── plots/          # Reward and Shortage visualizations
├── experiments/        # Historical experiment records
├── utils/              # Common utility functions
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── train.py            # Training pipeline
└── evaluate.py         # Comparative evaluation script
```

## 🛠️ 11. Installation Instructions
1. Clone the repository.
2. Ensure Python 3.8+ is installed.
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

## 📦 12. Requirements Installation
```bash
pip install -r requirements.txt
```

## 🚄 13. How to Train
Execute the training pipeline with the default configuration:
```bash
python train.py --config configs/qlearning.yaml
```

## 📊 14. How to Evaluate
Run the evaluation script to compare the RL agent against a fixed-rule baseline:
```bash
python evaluate.py --config configs/qlearning.yaml
```

## 🔄 15. Reproducibility Instructions
To reproduce the exact results shown in this report:
1. Ensure `agent.seed` is set to `42` in `configs/qlearning.yaml`.
2. Run the training script. The environment uses a local Random Number Generator (RNG) initialized with this seed to ensure demand patterns are identical across runs.

## 📈 16. Experiment Tracking Explanation
The project uses automated logging for MLOps compliance:
- **`logs/experiments.json`**: Stores every training run's ID, timestamp, hyperparameters (alpha, gamma), and final performance metrics.
- **`logs/training_logs.csv`**: Per-episode breakdown of rewards and shortages for plotting convergence.

## 💾 17. Policy Versioning Explanation
We maintain strict model versioning in the `policies/` directory:
- **`policy_v1.pkl`**: Standard production-ready model.
- **`policy_v2_explored.pkl`**: Model version trained with higher exploration or different seeds.
- **`policy_metadata.json`**: A machine-readable file tracking which algorithm, parameters, and rewards are associated with the current policies.

## 🖥️ 18. Monitoring Plan (Real-World Deployment)
In a production deployment, the system should be monitored via a dashboard tracking:
- **Congestion Frequency**: How often multiple zones simultaneously exceed 80% demand.
- **Resource Wastage**: Cumulative tank overflow units per day.
- **Fairness Index**: Variance in unmet demand across different socio-economic zones.
- **Safety Violations**: Frequency of tank levels dropping below a "Critical Low" threshold (e.g., 5%).
- **Emergency Handling**: Detection of sudden demand spikes (e.g., firefighting needs) that require manual override.

## 📝 19. Sample Outputs
### Training Log Sample:
```text
Episode 100/1000 | Avg Reward: -1455.96 | Epsilon: 0.606
Episode 500/1000 | Avg Reward: -1452.15 | Epsilon: 0.082
Training complete. Run ID: 20260510_184357
```

## 🖼️ 20. Plots Section
The following visualizations are generated automatically in `results/plots/`:
1.  **Reward Convergence**: Shows the agent learning and stabilizing over time.
2.  **Shortage Comparison**: Bar chart comparing the RL agent's efficiency against the Baseline system.

---
**Project developed for AAT academic evaluation.**
