import random
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, x_bins=100, vel_bins=100, action_bins=10):
        self.x_bins = x_bins
        self.vel_bins = vel_bins
        self.action_bins = action_bins
        
        # Espacios discretizados
        self.x_space = np.linspace(-1.2, 0.6, x_bins)
        self.vel_space = np.linspace(-0.07, 0.07, vel_bins)
        self.actions = list(np.linspace(-1, 1, action_bins))
        
        # Tabla Q
        self.Q = np.zeros((x_bins + 1, vel_bins + 1, action_bins))
    
    def discretize_state(self, obs):
        x, vel = obs
        x_bin = np.digitize(x, self.x_space)
        vel_bin = np.digitize(vel, self.vel_space)
        return (x_bin, vel_bin)
    
    def get_action_index(self, action):
        return np.argmin(np.abs(np.array(self.actions) - action))
    
    def next_action(self, state, epsilon=0.1, training=True):
        if training and np.random.random() < epsilon:
            # Exploración: acción aleatoria
            action = random.choice(self.actions)
        else:
            # Explotación: acción óptima
            action_idx = np.argmax(self.Q[state])
            action = self.actions[action_idx]
        
        return action

    