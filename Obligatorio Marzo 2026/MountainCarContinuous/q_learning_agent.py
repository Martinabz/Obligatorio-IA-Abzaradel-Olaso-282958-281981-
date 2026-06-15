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
        
        self.training_history = {
            'rewards': [],
            'steps': []
        }
    
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

    def train_agent(self, env, episodes=500, alpha=0.1, gamma=0.99, 
                    epsilon_start=0.9, epsilon_end=0.01, max_steps=1000):
        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.discretize_state(obs)
            done = False
            total_reward = 0
            steps = 0
            
            # Epsilon decay
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / episodes)
            
            while not done and steps < max_steps:
                steps += 1
                
                # Seleccionar acción
                action = self.next_action(state, epsilon=epsilon, training=True)
                action_idx = self.get_action_index(action)
                
                # Ejecutar acción
                obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = self.discretize_state(obs)
                
                # Q-Learning update
                next_action_idx = np.argmax(self.Q[next_state])
                self.Q[state][action_idx] = self.Q[state][action_idx] + alpha * (
                    reward + gamma * self.Q[next_state][next_action_idx] - self.Q[state][action_idx]
                )
                
                state = next_state
                total_reward += reward
            
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            
            if (episode + 1) % max(1, episodes // 10) == 0:
                avg_reward = np.mean(self.training_history['rewards'][-max(1, episodes // 10):])
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Steps: {steps}")

    def test_agent(self, env, episodes=10, render=False):
        test_rewards = []
        test_steps = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.discretize_state(obs)
            done = False
            total_reward = 0
            steps = 0
            
            while not done : # and steps < 4000:
                steps += 1
                # Usar política óptima (sin exploración)
                action = self.next_action(state, epsilon=0.0, training=False)
                obs, reward, done, _, _ = env.step(np.array([action]))
                state = self.discretize_state(obs)
                total_reward += reward
                
                if render:
                    env.render()
            
            test_rewards.append(total_reward)
            test_steps.append(steps)
            print(f"Test Episode {episode + 1} - Reward: {total_reward:.2f}, Steps: {steps}")
        
        return test_rewards, test_steps
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.training_history['rewards'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward per Episode')
        ax1.grid()
        
        ax2.plot(self.training_history['steps'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.grid()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='agent_model.npy'):
        np.save(filename, self.Q)
        print(f"Modelo guardado en {filename}")
    
    def load_model(self, filename='agent_model.npy'):
        self.Q = np.load(filename)
        print(f"Modelo cargado desde {filename}")
