import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from dyna_q_agent import DynaQAgent


class DynaQExperimentManager:
    """Coordina entrenamientos Dyna-Q, almacena agentes y permite comparar variantes."""

    def __init__(self):
        self.history = {}

    def train_run(self, env, run_id, x_bins, vel_bins, action_bins,
                  planning_steps=10, episodes=500, alpha=0.1, gamma=0.99,
                  epsilon_start=0.9, epsilon_end=0.01, max_steps=1000):

        agent = DynaQAgent(x_bins=x_bins, vel_bins=vel_bins, action_bins=action_bins)
        print(f"\n[{run_id}] Iniciando experimento con Q-Table de tamaño: {agent.Q.shape}, planning_steps={planning_steps}")

        local_rewards = []
        local_steps = []
        local_successes = []

        try:
            for episode in range(episodes):
                obs, _ = env.reset()
                state = agent.discretize_state(obs)
                done = False
                total_reward = 0
                steps = 0

                epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, episodes - 1))

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

                    agent.Model[state][action_idx] = (reward, next_state)
                    agent.observed_states.add(state)
                    agent.observed_actions[state].add(action_idx)

                    for _ in range(planning_steps):
                        if len(agent.observed_states) > 0:
                            state_sim = random.choice(list(agent.observed_states))
                            if state_sim in agent.observed_actions and len(agent.observed_actions[state_sim]) > 0:
                                action_idx_sim = random.choice(list(agent.observed_actions[state_sim]))
                                if action_idx_sim in agent.Model[state_sim]:
                                    reward_sim, next_state_sim = agent.Model[state_sim][action_idx_sim]
                                    next_action_idx_sim = np.argmax(agent.Q[next_state_sim])
                                    agent.Q[state_sim][action_idx_sim] += alpha * (
                                        reward_sim + gamma * agent.Q[next_state_sim][next_action_idx_sim]
                                        - agent.Q[state_sim][action_idx_sim]
                                    )

                    state = next_state
                    total_reward += reward

                success = int(done and steps < max_steps)
                local_rewards.append(total_reward)
                local_steps.append(steps)
                local_successes.append(success)

                if (episode + 1) % max(1, episodes // 5) == 0:
                    window = max(1, episodes // 5)
                    avg_r = np.mean(local_rewards[-window:])
                    avg_s = np.mean(local_successes[-window:]) * 100
                    print(f"[{run_id}] Ep {episode + 1}/{episodes} - Avg Reward: {avg_r:.2f}, Steps: {steps}, Success: {avg_s:.1f}%")

        except KeyboardInterrupt:
            print(f"\n[AVISO] Interrumpido por el usuario. Guardando progreso alcanzado...")

        self.history[run_id] = {
            'rewards': local_rewards,
            'steps': local_steps,
            'successes': local_successes,
            'agent': agent,
            'params': {
                'x_bins': x_bins, 'vel_bins': vel_bins, 'action_bins': action_bins,
                'planning_steps': planning_steps, 'alpha': alpha,
            }
        }
        print(f"--> Experimento '{run_id}' guardado con éxito en el historial.")

    def test_run(self, env, run_id, episodes=5, max_steps=4000):
        if run_id not in self.history:
            print(f"Error: No existe el experimento '{run_id}'")
            return

        print(f"\n=== Evaluando en vivo el experimento: {run_id} ===")
        agent = self.history[run_id]['agent']

        test_rewards = []
        test_steps = []
        success_count = 0

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

            success = int(done and steps < max_steps)
            success_count += success
            test_rewards.append(total_reward)
            test_steps.append(steps)
            print(f" Test Ep {episode + 1} - Reward: {total_reward:.2f}, Steps: {steps}, Success: {success}")

        return test_rewards, test_steps, success_count

    def reset_history(self):
        self.history.clear()
        print("¡Historial de experimentos reiniciado con éxito! El mánager está limpio.")

    def plot_comparisons(self):
        if not self.history:
            print("No hay experimentos registrados para graficar.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        for name, data in self.history.items():
            p = data['params']
            label_text = f"{name} (Bins: {p['x_bins']}x{p['vel_bins']}x{p['action_bins']}, n={p['planning_steps']} steps)"

            ax1.plot(data['rewards'], label=label_text, alpha=0.7)
            ax2.plot(data['steps'], label=name, alpha=0.7)

            window = max(1, len(data['successes']) // 20)
            success_rate = [
                np.mean(data['successes'][max(0, i - window):i + 1]) * 100
                for i in range(len(data['successes']))
            ]
            ax3.plot(success_rate, label=name, alpha=0.7)

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

        ax3.set_xlabel('Episodios')
        ax3.set_ylabel('Tasa de Éxito (%)')
        ax3.set_title('Comparativa: Tasa de Éxito (ventana deslizante)')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def plot_training_history(self, run_id):
        if run_id not in self.history:
            print(f"Error: El experimento '{run_id}' no existe en el historial.")
            return

        data = self.history[run_id]
        p = data['params']

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(
            f"Progreso: {run_id}  |  Bins: {p['x_bins']}x{p['vel_bins']}x{p['action_bins']}, planning_steps={p['planning_steps']}",
            fontsize=11
        )

        ax1.plot(data['rewards'], color='steelblue', alpha=0.8)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward per Episode')
        ax1.grid(True)

        ax2.plot(data['steps'], color='coral', alpha=0.8)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.grid(True)

        window = max(1, len(data['successes']) // 20)
        success_rate = [
            np.mean(data['successes'][max(0, i - window):i + 1]) * 100
            for i in range(len(data['successes']))
        ]
        ax3.plot(success_rate, color='seagreen', alpha=0.8)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Tasa de Éxito (%)')
        ax3.set_title('Tasa de Éxito (ventana deslizante)')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
