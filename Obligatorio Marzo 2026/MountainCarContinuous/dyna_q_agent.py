import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class DynaQAgent:
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
        
        # Modelo del entorno: Model[state][action] = (reward, next_state)
        self.Model = defaultdict(dict)
        
        # Estados y acciones observadas
        self.observed_states = set()
        self.observed_actions = defaultdict(set)
        
        self.training_history = {
            'rewards': [],
            'steps': [],
            'successes': []
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
            action = random.choice(self.actions)
        else:
            action_idx = np.argmax(self.Q[state])
            action = self.actions[action_idx]
        
        return action

    def train_agent(self, env, episodes=500, alpha=0.1, gamma=0.99, 
                    epsilon_start=0.9, epsilon_end=0.01, max_steps=1000, planning_steps=10):
        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.discretize_state(obs)
            done = False
            total_reward = 0
            steps = 0
            
            # Epsilon decay lineal
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, episodes - 1))
            
            while not done and steps < max_steps:
                steps += 1
                
                # Seleccionar acción
                action = self.next_action(state, epsilon=epsilon, training=True)
                action_idx = self.get_action_index(action)
                
                # Ejecutar acción en el entorno real
                obs, reward, done, _, _ = env.step(np.array([action]))
                next_state = self.discretize_state(obs)
                
                # Q-Learning update (experiencia real)
                next_action_idx = np.argmax(self.Q[next_state])
                self.Q[state][action_idx] = self.Q[state][action_idx] + alpha * (
                    reward + gamma * self.Q[next_state][next_action_idx] - self.Q[state][action_idx]
                )
                
                # Guardar en el modelo
                self.Model[state][action_idx] = (reward, next_state)
                
                # Registrar estado y acción observados
                self.observed_states.add(state)
                self.observed_actions[state].add(action_idx)
                
                # Fase de planificación: simulación con el modelo
                for _ in range(planning_steps):
                    if len(self.observed_states) > 0:
                        # Seleccionar estado aleatorio observado
                        state_sim = random.choice(list(self.observed_states))
                        
                        # Seleccionar acción aleatoria en ese estado
                        if state_sim in self.observed_actions and len(self.observed_actions[state_sim]) > 0:
                            action_idx_sim = random.choice(list(self.observed_actions[state_sim]))
                            
                            # Obtener transición del modelo
                            if action_idx_sim in self.Model[state_sim]:
                                reward_sim, next_state_sim = self.Model[state_sim][action_idx_sim]
                                
                                # Q-Learning update (experiencia simulada)
                                next_action_idx_sim = np.argmax(self.Q[next_state_sim])
                                self.Q[state_sim][action_idx_sim] = self.Q[state_sim][action_idx_sim] + alpha * (
                                    reward_sim + gamma * self.Q[next_state_sim][next_action_idx_sim] - self.Q[state_sim][action_idx_sim]
                                )
                
                state = next_state
                total_reward += reward
            
            success = int(done and steps < max_steps)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['successes'].append(success)
            
            if (episode + 1) % max(1, episodes // 10) == 0:
                avg_reward = np.mean(self.training_history['rewards'][-max(1, episodes // 10):])
                avg_success = np.mean(self.training_history['successes'][-max(1, episodes // 10):]) * 100
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Steps: {steps}, Success: {avg_success:.1f}%")

    def test_agent(self, env, episodes=10, render=False):
        test_rewards = []
        test_steps = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.discretize_state(obs)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 4000:
                steps += 1
                action = self.next_action(state, epsilon=0.0, training=False)
                obs, reward, done, _, _ = env.step(np.array([action]))
                state = self.discretize_state(obs)
                total_reward += reward
                
                if render:
                    env.render()
            
            success = int(done and steps < 4000)
            test_rewards.append(total_reward)
            test_steps.append(steps)
            success_count += success
            print(f"Test Episode {episode + 1} - Reward: {total_reward:.2f}, Steps: {steps}, Success: {success}")
        
        return test_rewards, test_steps, success_count
    
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
    
    def save_model(self, filename='dyna_agent_model.npy'):
        np.save(filename, self.Q)
        print(f"Modelo guardado en {filename}")
    
    def load_model(self, filename='dyna_agent_model.npy'):
        self.Q = np.load(filename)
        print(f"Modelo cargado desde {filename}")

