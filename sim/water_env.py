import numpy as np

class WaterDistributionEnv:
    """
    Simulator for Water Distribution Optimization.
    Zones: Zone A, Zone B, Zone C
    Goal: Allocate water from a central tank to meet zone demands while minimizing shortage and wastage.
    """
    def __init__(self, tank_capacity=100, refill_amount=30, demand_range=(5, 25)):
        self.tank_capacity = tank_capacity
        self.refill_amount = refill_amount
        self.demand_range = demand_range
        
        # Initial State
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.tank_level = self.tank_capacity
        self.demands = self.generate_demands()
        self.current_step = 0
        return self.get_state()

    def generate_demands(self):
        """Randomly generate water demand for 3 zones."""
        return np.random.randint(self.demand_range[0], self.demand_range[1] + 1, size=3)

    def get_state(self):
        """Return the current state (discretized for Q-learning)."""
        # Discretize tank level (0-100) into 5 bins: 0, 1, 2, 3, 4
        tank_bin = min(int(self.tank_level / 20), 4)
        
        # Discretize demands (5-25) into 4 bins: 0, 1, 2, 3
        demand_bins = [min(int((d - 5) / 5), 3) for d in self.demands]
        
        return (tank_bin, *demand_bins)

    def step(self, action):
        """
        Perform action and return next_state, reward, done.
        Actions:
        0: Equal distribution (Divide tank level equally)
        1: Prioritize Zone A (Full demand for A, then B, then C)
        2: Prioritize Zone B (Full demand for B, then A, then C)
        3: Prioritize Zone C (Full demand for C, then A, then B)
        """
        # 1. Calculate Supply based on action
        supply = np.zeros(3)
        remaining_water = self.tank_level
        
        if action == 0: # Equal
            allocated = remaining_water / 3
            for i in range(3):
                supply[i] = min(allocated, self.demands[i])
        
        elif action in [1, 2, 3]: # Priority
            priority_idx = action - 1
            other_indices = [i for i in range(3) if i != priority_idx]
            
            # Serve priority zone first
            supply[priority_idx] = min(remaining_water, self.demands[priority_idx])
            remaining_water -= supply[priority_idx]
            
            # Serve others with remainder
            for i in other_indices:
                supply[i] = min(remaining_water / 2, self.demands[i])
                remaining_water -= supply[i]

        # 2. Calculate Metrics
        shortage = np.sum(self.demands - supply)
        # Wastage can be from over-supplying (not possible here due to min()) 
        # or tank overflow when refilling
        wastage = 0 
        
        # 3. Update Tank (Refill and Usage)
        self.tank_level -= np.sum(supply)
        self.tank_level += self.refill_amount
        
        if self.tank_level > self.tank_capacity:
            wastage = self.tank_level - self.tank_capacity
            self.tank_level = self.tank_capacity
        
        self.tank_level = max(0, self.tank_level)

        # 4. Calculate Reward
        # Reward = -(shortage + wastage)
        reward = -(shortage + wastage)

        # 5. Transition to next state
        self.demands = self.generate_demands()
        self.current_step += 1
        
        next_state = self.get_state()
        done = False # Environment can be continuous or fixed length
        
        info = {
            "shortage": shortage,
            "wastage": wastage,
            "tank_utilization": self.tank_level / self.tank_capacity,
            "demands": self.demands,
            "supply": supply
        }

        return next_state, reward, done, info
