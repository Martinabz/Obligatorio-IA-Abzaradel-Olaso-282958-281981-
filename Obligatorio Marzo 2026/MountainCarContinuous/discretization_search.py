import numpy as np
import matplotlib.pyplot as plt


def evaluate_discretization(
    env,
    agent_class,
    x_bins,
    vel_bins,
    action_bins,
    train_kwargs,
    test_episodes=10,
    test_max_steps=4000,
):
    """
    Entrena y evalúa un agente con una configuración de discretización dada.

    Parámetros:
    -----------
    env : gymnasium.Env
        Entorno ya creado por el usuario (por fuera de esta función).
    agent_class : class
        Clase del agente: QLearningAgent o DynaQAgent.
    x_bins, vel_bins, action_bins : int
        Discretización a probar.
    train_kwargs : dict
        Argumentos para pasar a agent.train_agent().
    test_episodes : int
        Cantidad de episodios de evaluación sin exploración.
    test_max_steps : int
        Máximo de pasos por episodio durante evaluación.

    Retorna:
    --------
    dict con métricas y el agente entrenado.
    """
    agent = agent_class(x_bins=x_bins, vel_bins=vel_bins, action_bins=action_bins)
    agent.train_agent(env, **train_kwargs)

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

    return {
        'x_bins': x_bins,
        'vel_bins': vel_bins,
        'action_bins': action_bins,
        'avg_reward': float(np.mean(test_rewards)),
        'std_reward': float(np.std(test_rewards)),
        'avg_steps': float(np.mean(test_steps)),
        'success_rate': successes / test_episodes,
        'q_table_size': int(np.prod(agent.Q.shape)),
        'agent': agent,
        'training_history': agent.training_history,
    }


def find_best_discretization(
    env,
    agent_class,
    x_bins_list,
    vel_bins_list,
    action_bins_list,
    train_kwargs,
    test_episodes=10,
    test_max_steps=4000,
    metric='avg_reward',
):
    """
    Prueba todas las combinaciones de discretización y devuelve resultados
    ordenados según la métrica elegida.

    Métricas soportadas: 'avg_reward', 'success_rate', 'avg_steps' (busca menor).
    """
    results = []
    total = len(x_bins_list) * len(vel_bins_list) * len(action_bins_list)
    count = 0

    for xb in x_bins_list:
        for vb in vel_bins_list:
            for ab in action_bins_list:
                count += 1
                print(f"[{count}/{total}] Probando x_bins={xb}, vel_bins={vb}, action_bins={ab}")
                result = evaluate_discretization(
                    env=env,
                    agent_class=agent_class,
                    x_bins=xb,
                    vel_bins=vb,
                    action_bins=ab,
                    train_kwargs=train_kwargs,
                    test_episodes=test_episodes,
                    test_max_steps=test_max_steps,
                )
                results.append(result)
                print(f"       -> avg_reward={result['avg_reward']:.2f}, "
                      f"avg_steps={result['avg_steps']:.2f}, "
                      f"success_rate={result['success_rate']:.2f}")

    if metric == 'avg_steps':
        best = min(results, key=lambda r: r[metric])
        results = sorted(results, key=lambda r: r[metric])
    else:
        best = max(results, key=lambda r: r[metric])
        results = sorted(results, key=lambda r: r[metric], reverse=True)

    return best, results


def results_table(results, top_n=None):
    """Devuelve una lista de diccionarios limpios para mostrar o imprimir."""
    clean = []
    for r in sorted(results, key=lambda x: x['avg_reward'], reverse=True):
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
    if top_n is not None:
        clean = clean[:top_n]
    return clean


def print_results_table(results, top_n=None):
    """Imprime la tabla de resultados formateada."""
    rows = results_table(results, top_n=top_n)
    header = f"{'x':>4} {'v':>4} {'a':>4} {'avg_reward':>12} {'std_reward':>12} {'avg_steps':>11} {'success':>8} {'q_size':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['x_bins']:4d} {r['vel_bins']:4d} {r['action_bins']:4d} "
              f"{r['avg_reward']:12.2f} {r['std_reward']:12.2f} {r['avg_steps']:11.2f} "
              f"{r['success_rate']:8.2f} {r['q_table_size']:12d}")


def plot_heatmap(results, metric='avg_reward'):
    """Heatmap de x_bins vs vel_bins, un panel por cada action_bins."""
    action_bins_values = sorted(set(r['action_bins'] for r in results))
    fig, axes = plt.subplots(1, len(action_bins_values), figsize=(6 * len(action_bins_values), 5))
    if len(action_bins_values) == 1:
        axes = [axes]

    for ax, ab in zip(axes, action_bins_values):
        subset = [r for r in results if r['action_bins'] == ab]
        x_bins_values = sorted(set(r['x_bins'] for r in subset))
        vel_bins_values = sorted(set(r['vel_bins'] for r in subset))

        matrix = np.full((len(vel_bins_values), len(x_bins_values)), np.nan)
        for r in subset:
            i = vel_bins_values.index(r['vel_bins'])
            j = x_bins_values.index(r['x_bins'])
            matrix[i, j] = r[metric]

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
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


def plot_ranking(results, metric='avg_reward', top_n=10):
    """Gráfico de barras horizontales con las mejores configuraciones."""
    sorted_results = sorted(results, key=lambda r: r[metric], reverse=True)[:top_n]
    labels = [f"x={r['x_bins']}, v={r['vel_bins']}, a={r['action_bins']}" for r in sorted_results]
    values = [r[metric] for r in sorted_results]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_results)), values, color='steelblue')
    plt.yticks(range(len(sorted_results)), labels)
    plt.xlabel(metric)
    plt.title(f'Top {top_n} configuraciones por {metric}')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def plot_pareto(results, metric='avg_reward'):
    """Scatter del trade-off entre tamaño de la tabla Q y rendimiento."""
    sizes = [r['q_table_size'] for r in results]
    rewards = [r[metric] for r in results]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sizes, rewards, c=range(len(results)), cmap='viridis', s=100)

    best = max(results, key=lambda r: r[metric])
    plt.scatter(best['q_table_size'], best[metric], color='red', s=200, marker='*', label='Mejor')

    plt.xlabel('Tamaño tabla Q')
    plt.ylabel(metric)
    plt.title('Trade-off: tamaño vs rendimiento')
    plt.colorbar(scatter, label='configuración')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_convergence(results, top_n=3, metric='avg_reward'):
    """Curvas de reward y steps de las mejores configuraciones durante entrenamiento."""
    sorted_results = sorted(results, key=lambda r: r[metric], reverse=True)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in sorted_results:
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


def print_summary(best, results, metric='avg_reward'):
    """Imprime un resumen rápido de los resultados."""
    print("\n=== MEJOR CONFIGURACIÓN ===")
    print(f"x_bins={best['x_bins']}, vel_bins={best['vel_bins']}, action_bins={best['action_bins']}")
    print(f"avg_reward={best['avg_reward']:.2f}, std_reward={best['std_reward']:.2f}")
    print(f"avg_steps={best['avg_steps']:.2f}, success_rate={best['success_rate']:.2f}")
    print(f"q_table_size={best['q_table_size']}")
    print("\n=== TOP 5 CONFIGURACIONES ===")
    print_results_table(results, top_n=5)
