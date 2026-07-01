import numpy as np
import matplotlib.pyplot as plt


class _MergedManager:
    """Manager virtual que fusiona los historiales de dos o más managers para uso interno."""
    def __init__(self, *managers):
        self.history = {}
        for m in managers:
            self.history.update(m.history)


class HyperparameterEvaluator:
    """
    Evalúa y compara experimentos ya entrenados almacenados en un ExperimentManager
    o DynaQExperimentManager. No entrena: solo analiza el historial y corre tests.
    """

    def __init__(self, manager):
        self.manager = manager
        self.eval_results = {}

    # ------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------

    def evaluate(self, env, run_ids=None, episodes=10, max_steps=4000):
        """
        Corre episodios de test (epsilon=0) para cada run_id y guarda métricas.
        Si run_ids es None evalúa todos los experimentos del manager.
        """
        targets = run_ids if run_ids is not None else list(self.manager.history.keys())

        if not targets:
            print("No hay experimentos en el manager para evaluar.")
            return

        print(f"\n=== Evaluando {len(targets)} experimento(s), {episodes} episodios c/u ===")

        for run_id in targets:
            if run_id not in self.manager.history:
                print(f"[AVISO] '{run_id}' no encontrado en el manager, se omite.")
                continue

            agent = self.manager.history[run_id]['agent']
            test_rewards, test_steps, successes = [], [], 0

            for _ in range(episodes):
                obs, _ = env.reset()
                state = agent.discretize_state(obs)
                done = False
                total_reward = 0.0
                steps = 0

                while not done and steps < max_steps:
                    steps += 1
                    action = agent.next_action(state, epsilon=0.0, training=False)
                    obs, reward, done, _, _ = env.step(np.array([action]))
                    state = agent.discretize_state(obs)
                    total_reward += reward

                test_rewards.append(total_reward)
                test_steps.append(steps)
                if done and steps < max_steps:
                    successes += 1

            self.eval_results[run_id] = {
                'avg_reward':   float(np.mean(test_rewards)),
                'std_reward':   float(np.std(test_rewards)),
                'avg_steps':    float(np.mean(test_steps)),
                'success_rate': successes / episodes,
                'test_rewards': test_rewards,
                'test_steps':   test_steps,
                'params':       self.manager.history[run_id].get('params', {}),
            }
            r = self.eval_results[run_id]
            print(f"  [{run_id}] avg_reward={r['avg_reward']:.2f} ± {r['std_reward']:.2f} | "
                  f"success={r['success_rate']:.0%} | avg_steps={r['avg_steps']:.0f}")

        print("=== Evaluación completada ===\n")

    # ------------------------------------------------------------------
    # TABLA RESUMEN
    # ------------------------------------------------------------------

    def _filter(self, run_ids=None):
        """Devuelve subconjunto de eval_results filtrado por run_ids (o todos si None)."""
        if run_ids is None:
            return dict(self.eval_results)
        return {rid: self.eval_results[rid] for rid in run_ids if rid in self.eval_results}

    def get_summary_table(self, top_n=None, sort_by='avg_reward', run_ids=None):
        """Imprime y devuelve la tabla de resultados ordenada."""
        subset = self._filter(run_ids)
        if not subset:
            print("Aún no se corrió evaluate() o los run_ids indicados no existen.")
            return []

        reverse = sort_by != 'avg_steps'
        rows = sorted(subset.items(), key=lambda x: x[1][sort_by], reverse=reverse)
        if top_n:
            rows = rows[:top_n]

        header = f"{'run_id':<35} {'avg_reward':>12} {'std_reward':>11} {'success':>8} {'avg_steps':>10}  params"
        print(header)
        print("-" * (len(header) + 20))
        table = []
        for run_id, r in rows:
            param_str = "  ".join(f"{k}={v}" for k, v in r['params'].items())
            print(f"{run_id:<35} {r['avg_reward']:12.2f} {r['std_reward']:11.2f} "
                  f"{r['success_rate']:8.0%} {r['avg_steps']:10.0f}  {param_str}")
            table.append({'run_id': run_id, **r})

        return table

    def convergence_episode(self, run_id, threshold=0, window=50):
        """
        Devuelve el episodio en que la reward media (ventana móvil) superó
        `threshold` por primera vez. None si nunca ocurrió.
        """
        rewards = self.manager.history[run_id]['rewards']
        for i in range(window - 1, len(rewards)):
            if np.mean(rewards[i - window + 1: i + 1]) > threshold:
                return i + 1
        return None

    def stability_score(self, run_id, last_n=200):
        """
        Desviación estándar de las rewards en los últimos `last_n` episodios.
        Valor más bajo = agente más estable al final del entrenamiento.
        """
        rewards = self.manager.history[run_id]['rewards']
        tail = rewards[-last_n:] if len(rewards) >= last_n else rewards
        return float(np.std(tail))

    def plot_ranking(self, metric='avg_reward', top_n=10, run_ids=None):
        """Bar chart horizontal con los mejores runs."""
        subset = self._filter(run_ids)
        if not subset:
            print("Ejecutá evaluate() primero.")
            return

        reverse = metric != 'avg_steps'
        rows = sorted(subset.items(), key=lambda x: x[1][metric], reverse=reverse)[:top_n]
        labels = [r[0] for r in rows]
        values = [r[1][metric] for r in rows]

        plt.figure(figsize=(10, max(4, len(labels) * 0.5 + 1)))
        bars = plt.barh(range(len(rows)), values, color='steelblue', alpha=0.85)
        plt.yticks(range(len(rows)), labels)
        plt.xlabel(metric)
        plt.title(f'Ranking por {metric} (top {top_n})')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        for bar, val in zip(bars, values):
            plt.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_reward_vs_steps(self, run_ids=None):
        """Scatter: reward promedio vs pasos promedio en test."""
        subset = self._filter(run_ids)
        if not subset:
            print("Ejecutá evaluate() primero.")
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        for run_id, r in subset.items():
            ax.scatter(r['avg_steps'], r['avg_reward'], s=80, zorder=3)
            ax.annotate(run_id, (r['avg_steps'], r['avg_reward']),
                        textcoords='offset points', xytext=(5, 4), fontsize=7)

        ax.set_xlabel('Pasos promedio (test)')
        ax.set_ylabel('Reward promedio (test)')
        ax.set_title('Eficiencia: Reward vs Pasos en test')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_convergence_curves(self, run_ids=None, smoothing=50):
        """Reward suavizado (media móvil) durante entrenamiento por run."""
        targets = run_ids if run_ids is not None else list(self.manager.history.keys())

        fig, ax = plt.subplots(figsize=(12, 5))
        for run_id in targets:
            if run_id not in self.manager.history:
                continue
            rewards = self.manager.history[run_id]['rewards']
            smoothed = [
                np.mean(rewards[max(0, i - smoothing + 1): i + 1])
                for i in range(len(rewards))
            ]
            ax.plot(smoothed, label=run_id, alpha=0.85)

        ax.set_xlabel('Episodio')
        ax.set_ylabel(f'Reward (media móvil={smoothing})')
        ax.set_title('Curvas de convergencia durante entrenamiento')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_success_rate(self, run_ids=None, window=50):
        """Tasa de éxito acumulada (ventana deslizante) durante entrenamiento."""
        targets = run_ids if run_ids is not None else list(self.manager.history.keys())

        has_any = any(
            'successes' in self.manager.history[rid]
            for rid in targets if rid in self.manager.history
        )
        if not has_any:
            print("Ningún manager en los runs seleccionados guarda 'successes'. "
                  "Usá DynaQExperimentManager para esta métrica.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        for run_id in targets:
            if run_id not in self.manager.history:
                continue
            successes = self.manager.history[run_id].get('successes')
            if not successes:
                continue
            rate = [
                np.mean(successes[max(0, i - window + 1): i + 1]) * 100
                for i in range(len(successes))
            ]
            ax.plot(rate, label=run_id, alpha=0.85)

        ax.set_xlabel('Episodio')
        ax.set_ylabel(f'Tasa de éxito % (ventana={window})')
        ax.set_title('Evolución de la tasa de éxito durante entrenamiento')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_tradeoff(self, x_metric='stability', y_metric='avg_reward', run_ids=None):
        """
        Scatter configurable para justificar la elección final.
        x_metric puede ser 'stability' (calculado internamente), 'avg_steps' o cualquier
        clave de eval_results. y_metric debe ser una clave de eval_results.
        """
        subset = self._filter(run_ids)
        if not subset:
            print("Ejecutá evaluate() primero.")
            return

        x_vals, y_vals, labels = [], [], []
        for run_id, r in subset.items():
            if x_metric == 'stability':
                x = self.stability_score(run_id)
            else:
                x = r.get(x_metric, np.nan)
            y = r.get(y_metric, np.nan)
            x_vals.append(x)
            y_vals.append(y)
            labels.append(run_id)

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)

        best_idx = int(np.argmax(y_arr))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(x_arr, y_arr, s=80, color='steelblue', zorder=3, alpha=0.8)
        ax.scatter(x_arr[best_idx], y_arr[best_idx], s=160, color='red',
                   marker='*', zorder=4, label=f'Mejor: {labels[best_idx]}')

        for i, label in enumerate(labels):
            ax.annotate(label, (x_arr[i], y_arr[i]),
                        textcoords='offset points', xytext=(5, 4), fontsize=7)

        x_label = 'Estabilidad (std últimos 200 ep)' if x_metric == 'stability' else x_metric
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_metric)
        ax.set_title(f'Trade-off: {x_label} vs {y_metric}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


    def print_best(self, metric='avg_reward', run_ids=None):
        """Imprime el mejor run con todos sus parámetros y métricas."""
        subset = self._filter(run_ids)
        if not subset:
            print("Ejecutá evaluate() primero.")
            return

        reverse = metric != 'avg_steps'
        best_id, best = sorted(subset.items(),
                               key=lambda x: x[1][metric], reverse=reverse)[0]

        conv = self.convergence_episode(best_id)
        stab = self.stability_score(best_id)

        print("\n========== MEJOR CONFIGURACIÓN ==========")
        print(f"  run_id       : {best_id}")
        print(f"  Parámetros   : {best['params']}")
        print(f"  avg_reward   : {best['avg_reward']:.2f} ± {best['std_reward']:.2f}")
        print(f"  success_rate : {best['success_rate']:.0%}")
        print(f"  avg_steps    : {best['avg_steps']:.0f}")
        print(f"  Convergencia : episodio {conv if conv else 'no alcanzó umbral'}")
        print(f"  Estabilidad  : std={stab:.3f} (últimos 200 ep)")
        print("==========================================\n")

    def print_full_report(self, top_n=5, run_ids=None):
        """Reporte completo: tabla + mejor + convergencia y estabilidad por run."""
        subset = self._filter(run_ids)
        if not subset:
            print("Ejecutá evaluate() primero.")
            return

        print("\n" + "=" * 60)
        print("         REPORTE DE EVALUACIÓN DE HIPERPARÁMETROS")
        print("=" * 60)

        print(f"\n--- Top {top_n} por avg_reward ---")
        self.get_summary_table(top_n=top_n, run_ids=run_ids)

        print("\n--- Velocidad de convergencia (umbral reward > 0) ---")
        for run_id in subset:
            ep = self.convergence_episode(run_id, threshold=0)
            ep_str = str(ep) if ep else "no convergió"
            print(f"  {run_id:<35} episodio {ep_str}")

        print("\n--- Estabilidad final (std últimos 200 ep de entrenamiento) ---")
        stab_rows = sorted(
            [(rid, self.stability_score(rid)) for rid in subset],
            key=lambda x: x[1]
        )
        for run_id, stab in stab_rows:
            print(f"  {run_id:<35} std={stab:.3f}")

        print()
        self.print_best(run_ids=run_ids)


    def merge(self, other):
        """
        Devuelve un nuevo HyperparameterEvaluator que combina los eval_results
        y el manager.history de este evaluador con los de `other`.
        Los originales no se modifican.
        """
        combined = HyperparameterEvaluator(_MergedManager(self.manager, other.manager))
        combined.eval_results = {**self.eval_results, **other.eval_results}
        return combined
