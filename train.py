import yaml
import os
import pandas as pd
import numpy as np
from sim.water_env import WaterDistributionEnv
from agents.qlearning_agent import QLearningAgent
from datetime import datetime

def train():
    # 1. Load Configuration
    with open("configs/qlearning.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Environment and Agent
    env = WaterDistributionEnv(
        tank_capacity=config['environment']['tank_capacity'],
        refill_amount=config['environment']['refill_amount'],
        demand_range=config['environment']['demand_range']
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
    
    # 3. Training Loop
    history = []
    
    print(f"Starting training for {episodes} episodes...")
    
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

    # 4. Save Model and Logs
    os.makedirs("models", exist_ok=True)
    agent.save(config['training']['save_path'])
    
    # Also save a second version as requested in the prompt (policy_v2)
    v2_path = config['training']['save_path'].replace("v1", "v2")
    agent.save(v2_path)
    
    df_history = pd.DataFrame(history)
    df_history.to_csv(config['training']['log_path'], index=False)
    
    # MLOps Experiment Tracking
    exp_tracking_path = "results/experiments.csv"
    exp_data = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "lr": config['agent']['learning_rate'],
        "epsilon": config['agent']['epsilon'],
        "gamma": config['agent']['gamma'],
        "avg_reward": df_history['reward'].tail(100).mean(),
        "total_shortage": df_history['shortage'].sum(),
        "total_wastage": df_history['wastage'].sum()
    }
    
    if os.path.exists(exp_tracking_path):
        df_exp = pd.read_csv(exp_tracking_path)
        df_exp = pd.concat([df_exp, pd.DataFrame([exp_data])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([exp_data])
    
    df_exp.to_csv(exp_tracking_path, index=False)
    
    print(f"Training complete. Model saved to {config['training']['save_path']}")
    print(f"Logs saved to {config['training']['log_path']}")

if __name__ == "__main__":
    train()
