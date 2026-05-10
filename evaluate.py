import yaml
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent

def run_evaluation(env, agent, episodes, max_steps, mode="RL"):
    results = []
    for ep in range(episodes):
        state = env.reset()
        total_shortage = 0
        total_wastage = 0
        utilization = []
        
        for step in range(max_steps):
            if mode == "RL":
                action = agent.choose_action(state)
            else: # Baseline: Fixed Equal Distribution
                action = 0
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            total_shortage += info['shortage']
            total_wastage += info['wastage']
            utilization.append(info['tank_utilization'])
            
        results.append({
            "mode": mode,
            "shortage": total_shortage,
            "wastage": total_wastage,
            "avg_utilization": np.mean(utilization)
        })
    return pd.DataFrame(results)

def evaluate(config_path):
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = WaterDistributionEnv(
        tank_capacity=config['environment']['tank_capacity'],
        refill_amount=config['environment']['refill_amount'],
        demand_range=config['environment']['demand_range']
    )

    # 2. Load RL Agent
    agent = QLearningAgent(n_actions=4)
    # Support loading from the versioned policies folder
    policy_path = config['training']['save_path']
    if os.path.exists(policy_path):
        agent.load(policy_path)
        print(f"Loaded agent from {policy_path}")
    else:
        print(f"Warning: Trained model not found at {policy_path}. Running with untrained agent.")

    test_episodes = config['evaluation']['test_episodes']
    max_steps = config['environment']['max_steps']

    # 3. Run Evaluations
    print(f"Evaluating RL Agent using config: {config_path}...")
    rl_results = run_evaluation(env, agent, test_episodes, max_steps, mode="RL")
    
    print("Evaluating Baseline (Equal Distribution)...")
    baseline_results = run_evaluation(env, None, test_episodes, max_steps, mode="Baseline")

    # 4. Generate Metrics Table
    comparison = pd.concat([rl_results, baseline_results])
    summary = comparison.groupby("mode").agg({
        "shortage": ["mean", "std"],
        "wastage": ["mean", "std"],
        "avg_utilization": "mean"
    }).round(2)
    
    print("\n--- Comparison Summary ---")
    print(summary)

    # 5. Generate Plots
    plots_dir = config['evaluation']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Shortage & Wastage Comparison
    plt.figure(figsize=(10, 6))
    summary_mean = comparison.groupby("mode")[["shortage", "wastage"]].mean()
    summary_mean.plot(kind="bar", color=['#ff6b6b', '#4ecdc4'])
    plt.title("RL Agent vs Baseline: Avg Shortage & Wastage")
    plt.ylabel("Water Units")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, "comparison_metrics.png"))
    plt.close()

    # Plot 2: Reward Progress (from logs)
    if os.path.exists(config['training']['log_path']):
        df_logs = pd.read_csv(config['training']['log_path'])
        plt.figure(figsize=(10, 6))
        plt.plot(df_logs['episode'], df_logs['reward'].rolling(window=50).mean(), color='#45b7d1')
        plt.title("RL Training Progress (Reward Convergence)")
        plt.xlabel("Episode")
        plt.ylabel("Rolling Avg Reward")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "training_reward.png"))
        plt.close()

    # 6. Generate Markdown Report
    report_path = config['evaluation']['report_path']
    with open(report_path, "w") as f:
        f.write("# Evaluation Report: Adaptive Water Distribution Optimization\n\n")
        f.write("## Overview\n")
        f.write("This report compares the Reinforcement Learning (Q-Learning) agent against a fixed baseline (Equal Distribution).\n\n")
        f.write("## Performance Metrics\n")
        f.write(summary.to_markdown() + "\n\n")
        f.write("## Conclusion\n")
        rl_avg_shortage = summary.loc["RL", ("shortage", "mean")]
        base_avg_shortage = summary.loc["Baseline", ("shortage", "mean")]
        
        if rl_avg_shortage < base_avg_shortage:
            f.write(f"The RL Agent outperformed the baseline by reducing water shortage from {base_avg_shortage} to {rl_avg_shortage}.\n")
        else:
            f.write("The RL Agent is still converging. Consider increasing training episodes or tuning hyperparameters.\n")
    
    print(f"\nEvaluation complete. Plots saved to {plots_dir}")
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Water Distribution RL Agent")
    parser.add_argument("--config", type=str, default="configs/qlearning.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    evaluate(args.config)
