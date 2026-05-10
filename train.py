import yaml
import os
import json
import pandas as pd
import numpy as np
import random
import argparse
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent
from datetime import datetime

def train(config_path):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set Seeds for Reproducibility
    seed = config['agent'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)

    # 3. Initialize Environment and Agent
    env = WaterDistributionEnv(
        tank_capacity=config['environment']['tank_capacity'],
        refill_amount=config['environment']['refill_amount'],
        demand_range=config['environment']['demand_range'],
        seed=seed
    )
    
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        epsilon=config['agent']['epsilon'],
        epsilon_decay=config['agent']['epsilon_decay'],
        epsilon_min=config['agent']['epsilon_min']
    )

    episodes = config['training']['episodes']
    max_steps = config['environment']['max_steps']
    
    # 4. Training Loop
    history = []
    
    print(f"Starting training for {episodes} episodes using config: {config_path}...")
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        total_shortage = 0
        total_wastage = 0
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            total_shortage += info['shortage']
            total_wastage += info['wastage']
            
            if done:
                break
        
        agent.decay_epsilon()
        
        history.append({
            "episode": ep + 1,
            "reward": total_reward,
            "shortage": total_shortage,
            "wastage": total_wastage,
            "epsilon": agent.epsilon
        })
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean([h['reward'] for h in history[-100:]])
            print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # 5. Save Policies and Metadata
    os.makedirs("policies", exist_ok=True)
    
    v1_path = config['training']['save_path']
    v2_path = v1_path.replace("v1", "v2")
    v2_explored_path = v1_path.replace("v1", "v2_explored")
    
    agent.save(v1_path)
    agent.save(v2_path)
    agent.save(v2_explored_path)
    
    # Comprehensive Metadata JSON
    metadata = {
        "algorithm": "Q-Learning",
        "run_timestamp": datetime.now().isoformat(),
        "episodes_trained": episodes,
        "hyperparameters": config['agent'],
        "environment_settings": config['environment'],
        "results": {
            "final_avg_reward": float(np.mean([h['reward'] for h in history[-100:]])),
            "final_avg_shortage": float(np.mean([h['shortage'] for h in history[-100:]])),
            "final_avg_wastage": float(np.mean([h['wastage'] for h in history[-100:]]))
        }
    }
    with open("policies/policy_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # 6. Save Logs
    os.makedirs("logs", exist_ok=True)
    df_history = pd.DataFrame(history)
    df_history.to_csv(config['training']['log_path'], index=False)
    
    # 7. MLOps Experiment Tracking (CSV & JSON)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "alpha": config['agent']['learning_rate'],
        "gamma": config['agent']['gamma'],
        "epsilon": config['agent']['epsilon'],
        "epsilon_decay": config['agent']['epsilon_decay'],
        "avg_reward": metadata['results']['final_avg_reward'],
        "avg_shortage": metadata['results']['final_avg_shortage'],
        "avg_wastage": metadata['results']['final_avg_wastage']
    }
    
    # Append to logs
    exp_csv_path = "logs/experiments.csv"
    if os.path.exists(exp_csv_path):
        df_exp = pd.read_csv(exp_csv_path)
        df_exp = pd.concat([df_exp, pd.DataFrame([exp_data])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([exp_data])
    df_exp.to_csv(exp_csv_path, index=False)
    
    exp_json_path = "logs/experiments.json"
    experiments = []
    if os.path.exists(exp_json_path):
        with open(exp_json_path, "r") as f:
            experiments = json.load(f)
    experiments.append(exp_data)
    with open(exp_json_path, "w") as f:
        json.dump(experiments, f, indent=4)
    
    print(f"Training complete. Run ID: {run_id}")
    print(f"Policies saved to policies/")
    print(f"Logs saved to logs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Water Distribution RL Agent")
    parser.add_argument("--config", type=str, default="configs/qlearning.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    train(args.config)
