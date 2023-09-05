
import random

# Define the environment
class PlantEnvironment:
    def __init__(self):
        self.plant_growth = 50  # Initial growth level
        self.water = 0
        self.sunlight = 0
        self.fertilizer = 0

    def step(self, action):
        if action == "water":
            self.water += 1
            self.plant_growth += 10 if self.sunlight > 0 else -10
        elif action == "sunlight":
            self.sunlight += 1
            self.plant_growth += 10
        elif action == "fertilizer":
            self.fertilizer += 1
            if self.water > 0 and self.sunlight > 0:
                self.plant_growth += 20
            else:
                self.plant_growth -= 10
        return self.plant_growth

# Define a simple CRRL agent
class CRRLAgent:
    def __init__(self):
        self.causal_model = {
            "water": {"effect": 0},
            "sunlight": {"effect": 0},
            "fertilizer": {"effect": 0}
        }
        
    def update_causal_model(self, action, reward_delta):
        self.causal_model[action]["effect"] += reward_delta
    
    def select_action(self):
        return max(self.causal_model, key=lambda k: self.causal_model[k]["effect"])

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {action: 0 for action in actions}
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table, key=self.q_table.get)

    def update_q_value(self, action, reward, next_best_reward):
        old_value = self.q_table[action]
        self.q_table[action] = old_value + self.alpha * (reward + self.gamma * next_best_reward - old_value)

# Define the Enhanced CRRL agent
class EnhancedCRRLAgent:
    def __init__(self, actions):
        self.actions = actions
        self.causal_model = {action: {"effect": 0, "count": 0} for action in actions}
        self.epsilon = 0.2  # Exploration rate
        self.memory = []  # Store recent actions and rewards to consider longer-term effects
        self.time_horizon = 5  # Number of steps to consider for longer-term effects

    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.causal_model, key=lambda k: self.causal_model[k]["effect"])

    def update_causal_model(self, action, reward_delta):
        self.causal_model[action]["count"] += 1
        long_term_effect = sum([r[1] for r in self.memory[-self.time_horizon:]]) / self.time_horizon
        self.causal_model[action]["effect"] = (self.causal_model[action]["effect"] * (self.causal_model[action]["count"] - 1) + long_term_effect) / self.causal_model[action]["count"]

# Define the CCRL agent
class CCRLAgent:
    def __init__(self, actions):
        self.actions = actions
        # Causal model to understand immediate effects
        self.causal_model = {action: {"effect": 0, "count": 0} for action in actions}
        # Consequential model to predict future outcomes
        self.consequential_model = {action: 0 for action in actions}
        self.epsilon = 0.2  # Exploration rate
        self.memory = []  # Store recent actions and rewards
        self.time_horizon = 5  # Number of steps to consider for longer-term effects

    def select_action(self):
        # Blend causal and consequential insights for decision-making
        combined_insight = {action: self.causal_model[action]["effect"] + self.consequential_model[action] for action in self.actions}
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(combined_insight, key=combined_insight.get)

    def update_models(self, action, reward_delta):
        # Update causal model
        self.causal_model[action]["count"] += 1
        long_term_effect = sum([r[1] for r in self.memory[-self.time_horizon:]]) / self.time_horizon
        self.causal_model[action]["effect"] = (self.causal_model[action]["effect"] * (self.causal_model[action]["count"] - 1) + long_term_effect) / self.causal_model[action]["count"]
        # Update consequential model
        future_reward_prediction = sum([r[1] for r in self.memory]) / len(self.memory) if self.memory else 0
        self.consequential_model[action] = (self.consequential_model[action] + future_reward_prediction) / 2

# Helper functions for simulations are also included...
