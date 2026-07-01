import numpy as np
import matplotlib.pyplot as plt
import copy

class DiscretizationEvaluator:
    """Clase encargada de evaluar, acumular y graficar múltiples configuraciones de discretización por tandas."""
    
    def __init__(self, manager_instance):
        """ Recibe la instancia activa de tu ExperimentManager para interactuar con los agentes. """
        self.manager = manager_instance
        self.results = []  # Lista global que acumula los resultados de todas las tandas ejecutadas

    def evaluate_group(self, env, x_bins_list, vel_bins_list, action_bins_list, train_kwargs, 
                       test_episodes=10, test_max_steps=4000, prefix="Grid"):
        """
        Prueba el grupo (tanda) de combinaciones que le pases y las añade al historial acumulado.
        """
        # Aseguramos que los parámetros vengan en listas, si es un entero único lo envuelve
        x_list = [x_bins_list] if isinstance(x_bins_list, int) else x_bins_list
        v_list = [vel_bins_list] if isinstance(vel_bins_list, int) else vel_bins_list
        a_list = [action_bins_list] if isinstance(action_bins_list, int) else action_bins_list

        total_combinaciones = len(x_list) * len(v_list) * len(a_list)
        count = 0

        print(f"\n=== Iniciando Tanda de {total_combinaciones} combinaciones ===")

        for xb in x_list:
            for vb in v_list:
                for ab in a_list:
                    count += 1
                    # Generamos un ID único para registrarlo en el ExperimentManager
                    run_id = f"{prefix}_x{xb}_v{vb}_a{ab}"
                    
                    print(f"\n[{count}/{total_combinaciones}] Evaluando: {run_id}")
                    
                    # 1. Ejecutamos el entrenamiento delegando en nuestro ExperimentManager modular
                    self.manager.train_run(
                        env=env,
                        run_id=run_id,
                        x_bins=xb,
                        vel_bins=vb,
                        action_bins=ab,
                        **train_kwargs
                    )
                    
                    # 2. Extraemos el agente recién creado para realizar la evaluación sin exploración
                    agent = self.manager.history[run_id]['agent']
                    
                    test_rewards = []
                    test_steps = []
                    successes = 0

                    for _ in range(test_episodes):
                        obs, _ = env.reset()
                        state = agent.discretize_state(obs)
                        done = False
                        total_reward = 0.0
                        steps = 0

                        while not done and steps < test_max_steps:
                            steps += 1
                            action = agent.next_action(state, epsilon=0.0, training=False)
                            obs, reward, done, _, _ = env.step(np.array([action]))
                            state = agent.discretize_state(obs)
                            total_reward += reward

                        test_rewards.append(total_reward)
                        test_steps.append(steps)
                        if done and steps < test_max_steps:
                            successes += 1

                    # 3. Estructuramos el diccionario de métricas
                    result = {
                        'run_id': run_id,
                        'x_bins': xb,
                        'vel_bins': vb,
                        'action_bins': ab,
                        'avg_reward': float(np.mean(test_rewards)),
                        'std_reward': float(np.std(test_rewards)),
                        'avg_steps': float(np.mean(test_steps)),
                        'success_rate': successes / test_episodes,
                        'q_table_size': int(np.prod(agent.Q.shape)),
                        # Extraemos las listas de recompensas y pasos directo desde el mánager
                        'training_history': {
                            'rewards': self.manager.history[run_id]['rewards'],
                            'steps': self.manager.history[run_id]['steps']
                        }
                    }
                    
                    # Lo guardamos en el acumulador global de este objeto
                    self.results.append(result)
                    print(f"       -> Resultado: Avg Reward={result['avg_reward']:.2f} | Success Rate={result['success_rate']:.2f}")

        print(f"\n=== Tanda completada. Historial global cuenta con {len(self.results)} combinaciones registradas. ===")

    def get_results_table(self, top_n=None):
        """Devuelve una lista limpia ordenada por rendimiento."""
        clean = []
        for r in sorted(self.results, key=lambda x: x['avg_reward'], reverse=True):
            clean.append({
                'x_bins': r['x_bins'],
                'vel_bins': r['vel_bins'],
                'action_bins': r['action_bins'],
                'avg_reward': r['avg_reward'],
                'std_reward': r['std_reward'],
                'avg_steps': r['avg_steps'],
                'success_rate': r['success_rate'],
                'q_table_size': r['q_table_size'],
            })
        return clean[:top_n] if top_n is not None else clean

    def print_results_table(self, top_n=None):
        rows = self.get_results_table(top_n=top_n)
        header = f"{'x':>4} {'v':>4} {'a':>4} {'avg_reward':>12} {'std_reward':>12} {'avg_steps':>11} {'success':>8} {'q_size':>12}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['x_bins']:4d} {r['vel_bins']:4d} {r['action_bins']:4d} "
                  f"{r['avg_reward']:12.2f} {r['std_reward']:12.2f} {r['avg_steps']:11.2f} "
                  f"{r['success_rate']:8.2f} {r['q_table_size']:12d}")

    def plot_heatmap(self, metric='avg_reward'):
        if not self.results:
            print("No hay resultados para graficar.")
            return
        action_bins_values = sorted(set(r['action_bins'] for r in self.results))
        fig, axes = plt.subplots(1, len(action_bins_values), figsize=(6 * len(action_bins_values), 5))
        if len(action_bins_values) == 1:
            axes = [axes]

        for ax, ab in zip(axes, action_bins_values):
            subset = [r for r in self.results if r['action_bins'] == ab]
            if not subset: continue
            x_bins_values = sorted(set(r['x_bins'] for r in subset))
            vel_bins_values = sorted(set(r['vel_bins'] for r in subset))

            matrix = np.full((len(vel_bins_values), len(x_bins_values)), np.nan)
            for r in subset:
                i = vel_bins_values.index(r['vel_bins'])
                j = x_bins_values.index(r['x_bins'])
                matrix[i, j] = r[metric]

            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', origin='lower')
            ax.set_xticks(range(len(x_bins_values)))
            ax.set_yticks(range(len(vel_bins_values)))
            ax.set_xticklabels(x_bins_values)
            ax.set_yticklabels(vel_bins_values)
            ax.set_xlabel('x_bins')
            ax.set_ylabel('vel_bins')
            ax.set_title(f'action_bins={ab} - {metric}')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot_ranking(self, metric='avg_reward', top_n=10):
        sorted_res = sorted(self.results, key=lambda r: r[metric], reverse=True)[:top_n]
        labels = [f"x={r['x_bins']}, v={r['vel_bins']}, a={r['action_bins']}" for r in sorted_res]
        values = [r[metric] for r in sorted_res]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_res)), values, color='steelblue')
        plt.yticks(range(len(sorted_res)), labels)
        plt.xlabel(metric)
        plt.title(f'Top {top_n} configuraciones por {metric}')
        plt.gca().invert_yaxis()
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

    def plot_pareto(self, metric='avg_reward'):
        sizes = [r['q_table_size'] for r in self.results]
        rewards = [r[metric] for r in self.results]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(sizes, rewards, c=range(len(self.results)), cmap='viridis', s=100)

        best = max(self.results, key=lambda r: r[metric])
        plt.scatter(best['q_table_size'], best[metric], color='red', s=200, marker='*', label='Mejor')

        plt.xlabel('Tamaño tabla Q')
        plt.ylabel(metric)
        plt.title('Trade-off: tamaño vs rendimiento')
        plt.colorbar(scatter, label='Índice Experimento')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_convergence(self, top_n=3, metric='avg_reward'):
        sorted_res = sorted(self.results, key=lambda r: r[metric], reverse=True)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for r in sorted_res:
            label = f"x={r['x_bins']}, v={r['vel_bins']}, a={r['action_bins']}"
            history = r['training_history']
            axes[0].plot(history['rewards'], label=label, alpha=0.7)
            axes[1].plot(history['steps'], label=label, alpha=0.7)

        axes[0].set_xlabel('Episodio')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward por episodio')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_xlabel('Episodio')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Steps por episodio')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def print_summary(self, metric='avg_reward'):
        if not self.results: return
        best = min(self.results, key=lambda r: r[metric]) if metric == 'avg_steps' else max(self.results, key=lambda r: r[metric])
        print("\n=== MEJOR CONFIGURACIÓN EN EL HISTORIAL ACUMULADO ===")
        print(f"x_bins={best['x_bins']}, vel_bins={best['vel_bins']}, action_bins={best['action_bins']}")
        print(f"avg_reward={best['avg_reward']:.2f}, std_reward={best['std_reward']:.2f}")
        print(f"avg_steps={best['avg_steps']:.2f}, success_rate={best['success_rate']:.2f}")
        print(f"q_table_size={best['q_table_size']}")
        print("\n=== TOP 5 CONFIGURACIONES GENERALES ===")
        self.print_results_table(top_n=5)