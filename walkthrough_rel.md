# Walkthrough - Adaptive Water Distribution RL

The project "Adaptive Water Distribution Optimization using Reinforcement Learning" is now complete and fully functional.

## Changes Made
- **Simulator**: Developed a custom environment (`sim/water_env.py`) representing 3 zones and a central tank.
- **Agent**: Implemented a Tabular Q-Learning agent (`agents/qlearning_agent.py`) with epsilon-greedy exploration.
- **Pipelines**: Created modular scripts for training (`train.py`) and evaluation (`evaluate.py`).
- **MLOps**: Added YAML configuration, CSV experiment tracking, and model versioning.
- **Reporting**: Automated plot generation and Markdown report creation.

## Results & Validation

### 1. Training Progress
The agent was trained for 1000 episodes. The training logs show the epsilon decaying and the agent exploring different allocation strategies.

![Reward Progress](file:///c:/Users/acer/OneDrive/Desktop/REL_AAT/results/plots/training_reward.png)

### 2. Comparative Analysis
The RL agent was compared against a fixed baseline (Equal Distribution). 

![Comparison Metrics](file:///c:/Users/acer/OneDrive/Desktop/REL_AAT/results/plots/comparison_metrics.png)

### 3. Generated Report
A detailed evaluation report was generated at [evaluation_report.md](file:///c:/Users/acer/OneDrive/Desktop/REL_AAT/results/evaluation_report.md).

## How to Explore
- **Models**: Check `models/policy_v1.pkl` for the trained Q-table.
- **Logs**: Review `results/training_logs.csv` and `results/experiments.csv`.
- **Config**: Modify `configs/qlearning.yaml` to experiment with different parameters.
