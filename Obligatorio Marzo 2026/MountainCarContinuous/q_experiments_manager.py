import numpy as np
import matplotlib.pyplot as plt

from q_learning_agent import QLearningAgent

class ExperimentManager:
    """Clase encargada de coordinar entrenamientos, almacenar los agentes y graficar."""
    def __init__(self):
        self.history = {}  # Diccionario de experimento indexado por su run_id
        
    def train_run(self, env, run_id, x_bins, vel_bins, action_bins, 
                  episodes=500, alpha=0.1, gamma=0.99, 
                  epsilon_start=0.9, epsilon_end=0.01, max_steps=1000):
        
        # Creación del agente usando la clase importada
        agent = QLearningAgent(x_bins=x_bins, vel_bins=vel_bins, action_bins=action_bins)
        print(f"\n[{run_id}] Iniciando experimento con Q-Table de tamaño: {agent.Q.shape}")
            
        local_rewards = []
        local_steps = []
        
        try:
            for episode in range(episodes):
                obs, _ = env.reset()
                state = agent.discretize_state(obs)
                done = False
                total_reward = 0
                steps = 0
                
                epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / episodes)
                
                while not done and steps < max_steps:
                    steps += 1
                    
                    action = agent.next_action(state, epsilon=epsilon, training=True)
                    action_idx = agent.get_action_index(action)
                    
                    obs, reward, done, _, _ = env.step(np.array([action]))
                    next_state = agent.discretize_state(obs)
                    
                    next_action_idx = np.argmax(agent.Q[next_state])
                    agent.Q[state][action_idx] += alpha * (
                        reward + gamma * agent.Q[next_state][next_action_idx] - agent.Q[state][action_idx]
                    )
                    
                    state = next_state
                    total_reward += reward
                
                local_rewards.append(total_reward)
                local_steps.append(steps)
                
                if (episode + 1) % max(1, episodes // 5) == 0:
                    avg_r = np.mean(local_rewards[-max(1, episodes // 5):])
                    print(f"[{run_id}] Ep {episode + 1}/{episodes} - Avg Reward: {avg_r:.2f}, Steps: {steps}")
                    
        except KeyboardInterrupt:
            print(f"\n[AVISO] Interrumpido por el usuario. Guardando progreso alcanzado...")

        self.history[run_id] = {
            'rewards': local_rewards,
            'steps': local_steps,
            'agent': agent,
            'params': {'x_bins': x_bins, 'vel_bins': vel_bins, 'action_bins': action_bins, 'alpha': alpha}
        }
        print(f"--> Experimento '{run_id}' guardado con éxito en el historial.")

    def test_run(self, env, run_id, episodes=5, max_steps=1000):
        if run_id not in self.history:
            print(f"Error: No existe el experimento '{run_id}'")
            return
            
        print(f"\n=== Evaluando en vivo el experimento: {run_id} ===")
        agent = self.history[run_id]['agent']
        
        test_rewards = []
        for episode in range(episodes):
            obs, _ = env.reset()
            state = agent.discretize_state(obs)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                steps += 1
                action = agent.next_action(state, epsilon=0.0, training=False)
                obs, reward, done, _, _ = env.step(np.array([action]))
                state = agent.discretize_state(obs)
                total_reward += reward
                
            test_rewards.append(total_reward)
            print(f" Test Ep {episode + 1} - Reward: {test_rewards[-1]:.2f}, Steps: {steps}")
            
        return test_rewards

    def reset_history(self):
        self.history.clear()
        print("¡Historial de experimentos reiniciado con éxito! El mánager está limpio.")
    
    def plot_comparisons(self):
        if not self.history:
            print("No hay experimentos registrados para graficar.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for name, data in self.history.items():
            p = data['params']
            label_text = f"{name} (Bins: {p['x_bins']}x{p['vel_bins']}x{p['action_bins']})"
            
            ax1.plot(data['rewards'], label=label_text, alpha=0.7)
            ax2.plot(data['steps'], label=name, alpha=0.7)
            
        ax1.set_xlabel('Episodios')
        ax1.set_ylabel('Recompensa Total')
        ax1.set_title('Comparativa: Recompensa por Episodio')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Episodios')
        ax2.set_ylabel('Pasos')
        ax2.set_title('Comparativa: Pasos por Episodio')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, run_id):
        """Grafica el historial de rewards y steps de un modelo específico justo después de entrenar."""
        if run_id not in self.history:
            print(f"Error: El experimento '{run_id}' no existe en el historial.")
            return
            
        # Extraemos los datos de ese experimento en particular
        data = self.history[run_id]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Progreso del experimento: {run_id}", fontsize=12)
        
        # Gráfica de Recompensas
        ax1.plot(data['rewards'], color='steelblue', alpha=0.8)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward per Episode')
        ax1.grid(True)
        
        # Gráfica de Pasos
        ax2.plot(data['steps'], color='coral', alpha=0.8)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()