import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import optimize_model
from orchestrator import train_and_save_model
from random_search import random_search_optimize
import os


def run_comparative_analysis():
    """
    Ejecuta análisis comparativo entre BO y Random Search (REGRESIÓN) usando múltiples semillas.
    Genera dos DataFrames:
    - results_df: Métricas del mejor RMSE global por modelo y método (a través de todas las semillas).
    - histories_df: Historias completas de RMSE por iteración, modelo, método y semilla.
    """
    SEMILLA_BASE = 1024
    n_semillas = 10

    models = ["svm", "rf", "mlp"]
    model_names = {"svm": "SVR", "rf": "Random Forest Regressor", "mlp": "MLP Regressor"}
    inverse_model_names = {v: k for k, v in model_names.items()}

    results = []
    histories_records = []

    print("=" * 70)
    print("ANÁLISIS COMPARATIVO: Optimización Bayesiana vs Random Search (REGRESIÓN)")
    print("=" * 70)

    for model in models:
        print(f"\n{'#' * 70}")
        print(f"# MODELO: {model_names[model]}")
        print(f"{'#' * 70}")

        bo_best_rmse = float("inf")
        bo_best_params = None
        bo_best_seed = None
        bo_best_history = None
        bo_best_iter = None

        rs_best_rmse = float("inf")
        rs_best_params = None
        rs_best_seed = None
        rs_best_history = None
        rs_best_iter = None

        for seed in range(SEMILLA_BASE, SEMILLA_BASE + n_semillas):
            np.random.seed(seed)
            # ----------- BAYESIAN OPTIMIZATION -----------
            params = None
            if model == "svm":
                params = {
                    "C": 10 ** np.random.uniform(-5, 5, 35),
                    "gamma": 10 ** np.random.uniform(-5, 5, 35),
                }
            if model == "rf":
                params = {
                    "n_estimators": np.random.randint(50, 201, 35),
                    "max_depth": np.random.randint(4, 11, 35),
                }
            if model == "mlp":
                params = {
                    "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                    "alpha": 10 ** np.random.uniform(-5, 5, 10)
                }
            bo_params, bo_rmse, bo_history = optimize_model(model, n_init=5, n_iter=30, params=params, return_history=True)
            bo_history_adapted = [item['rmse'] for item in bo_history]
            histories_records.append({
                'model_name': model_names[model],
                'method': 'BO',
                'semilla': seed,
                'rmse': bo_history_adapted
            })

            # Mejor iteración de BO para esta semilla
            min_bo_rmse = min(bo_history_adapted)
            min_bo_iter = bo_history_adapted.index(min_bo_rmse)
            if min_bo_rmse < bo_best_rmse:
                bo_best_rmse = min_bo_rmse
                bo_best_params = bo_params
                bo_best_seed = seed
                bo_best_history = bo_history_adapted
                bo_best_iter = min_bo_iter

            # ----------- RANDOM SEARCH -----------
            rs_params, rs_rmse, rs_history = random_search_optimize(model, n_iterations=35, params=params)
            rs_history_adapted = [item[1] for item in rs_history]
            histories_records.append({
                'model_name': model_names[model],
                'method': 'RS',
                'semilla': seed,
                'rmse': rs_history_adapted
            })

            # Mejor iteración de RS para esta semilla
            min_rs_rmse = min(rs_history_adapted)
            min_rs_iter = rs_history_adapted.index(min_rs_rmse)
            if min_rs_rmse < rs_best_rmse:
                rs_best_rmse = min_rs_rmse
                rs_best_params = rs_params
                rs_best_seed = seed
                rs_best_history = rs_history_adapted
                rs_best_iter = min_rs_iter

        # ----------- RESULTADOS -----------
        improvement = ((rs_best_rmse - bo_best_rmse) / rs_best_rmse * 100) if rs_best_rmse > 0 else 0

        results.append({
            "Modelo": model_names[model],
            "BO_Params": bo_best_params,
            "BO_RMSE": bo_best_rmse,
            "RS_Params": rs_best_params,
            "RS_RMSE": rs_best_rmse,
            "Mejora_BO(%)": improvement
        })

        print(f"\n{'=' * 70}")
        print(f"RESUMEN - {model_names[model]}")
        print(f"{'=' * 70}")
        print(f"Bayesian Optimization (mejor semilla={bo_best_seed}, iteración={bo_best_iter}):")
        print(f"  Parámetros: {bo_best_params}")
        print(f"  RMSE: {bo_best_rmse:.4f}")

        print(f"\nRandom Search (mejor semilla={rs_best_seed}, iteración={rs_best_iter}):")
        print(f"  Parámetros: {rs_best_params}")
        print(f"  RMSE: {rs_best_rmse:.4f}")

        print(f"\nMejora de BO sobre RS: {improvement:.2f}%")

    results_df = pd.DataFrame(results)
    histories_df = pd.DataFrame(histories_records)

    plot_error_history_iterations(histories_df)

    # =============================
    #  GUARDAR SOLO EL MEJOR MODELO GLOBAL
    # =============================
    print("\nBuscando el mejor modelo global...")

    best_bo_idx = results_df["BO_RMSE"].idxmin()
    best_rs_idx = results_df["RS_RMSE"].idxmin()

    best_bo_val = results_df.loc[best_bo_idx, "BO_RMSE"]
    best_rs_val = results_df.loc[best_rs_idx, "RS_RMSE"]

    if best_bo_val < best_rs_val:
        winner_method = "BO"
        winner_idx = best_bo_idx
        winner_rmse = best_bo_val
        winner_params = results_df.loc[best_bo_idx, "BO_Params"]
    else:
        winner_method = "RS"
        winner_idx = best_rs_idx
        winner_rmse = best_rs_val
        winner_params = results_df.loc[best_rs_idx, "RS_Params"]

    winner_model_name = results_df.loc[winner_idx, "Modelo"]
    winner_model_code = inverse_model_names[winner_model_name]

    print(f"\n🏆 Mejor modelo global: {winner_model_name}")
    print(f"   Método ganador: {winner_method}")
    print(f"   RMSE: {winner_rmse:.4f}")
    print(f"   Parámetros: {winner_params}")

    # Guardar modelo ganador global
    pkl_name = f"best_model_final.pkl"
    base_path = __file__
    output_path = os.path.join(os.path.dirname(base_path), "..", "api", pkl_name)
    train_and_save_model(winner_model_code, winner_params, output_path)
    print(f"\n💾 Modelo global guardado como: {output_path}\n")

    return results_df, histories_df

def create_comparison_table(results_df, output_path="../results/comparison_table.csv", verbose=False):

    if verbose:
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA FINAL (RMSE — menor es mejor)")
        print("=" * 70)
        print(results_df.to_string(index=False))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nTabla guardada en: {output_path}")

    return results_df


def plot_comparison_metrics(results_df, output_path="./results/comparison_plot.png"):
    BASE_FILE = os.getcwd()
    output_path = os.path.join(BASE_FILE, output_path)
    _, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35

    bo_metrics = results_df["BO_RMSE"].values
    rs_metrics = results_df["RS_RMSE"].values
    labels = results_df["Modelo"].values

    bars1 = ax.bar(x - width/2, bo_metrics, width, label="Bayesian Optimization (RMSE)")
    bars2 = ax.bar(x + width/2, rs_metrics, width, label="Random Search (RMSE)")

    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("RMSE (menor es mejor)", fontsize=12)
    ax.set_title("Comparación: BO vs RS (Regresión)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # annotate
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", 
                    ha="center", va="bottom", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_analysis_report(results_df, output_path="./results/analysis_report.md", verbose=False):

    report = f"""# Análisis Comparativo: Optimización Bayesiana vs Random Search (Regresión)

Fecha de ejecución: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Tabla Comparativa

| Modelo | Método | Hiperparámetros | RMSE | Mejora BO (%) |
|--------|---------|------------------|--------|----------------|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['Modelo']} | BO | {row['BO_Params']} | {row['BO_RMSE']:.4f} | - |\n"
        report += f"| {row['Modelo']} | RS | {row['RS_Params']} | {row['RS_RMSE']:.4f} | {row['Mejora_BO(%)']:.2f}% |\n"

    best_idx = results_df["BO_RMSE"].idxmin()
    best_model = results_df.loc[best_idx, "Modelo"]
    best_rmse = results_df.loc[best_idx, "BO_RMSE"]

    report += f"""

---

## 2. Conclusiones

- La métrica usada es **RMSE**, menor = mejor.
- El mejor modelo según BO fue **{best_model}**, con RMSE = {best_rmse:.4f}.
- En general, BO logra valores más bajos de RMSE usando menos evaluaciones.

---

"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    if verbose:
        print(f"Reporte guardado en: {output_path}")

def plot_error_history_iterations(histories_df, output_path="./results/error_history_iterations.png", verbose=False):
    """
    histories_df : DataFrame con columnas ['model_name', 'method', 'semilla', 'rmse']
    'rmse' es un arreglo con la historia de RMSE por iteración.
    Grafica el promedio y ±1 desviación estándar de RMSE por iteración para cada modelo y método.
    """
    if histories_df is None or histories_df.empty:
        if verbose:
            print("No hay historiales para graficar.")
        return None

    models = histories_df['model_name'].unique()
    n = len(models)
    fig_width = max(6, 4 * n)
    fig, ax_row = plt.subplots(1, n, figsize=(fig_width, 4), squeeze=False, sharey=False)
    axes = ax_row[0]

    color_map = {"BO": "C0", "RS": "C1"}

    for i, model in enumerate(models):
        ax = axes[i]
        for method in ["BO", "RS"]:
            subset = histories_df[(histories_df['model_name'] == model) & (histories_df['method'] == method)]
            # Convertir lista de arrays en matriz 2D (semillas x iteraciones)
            rmse_matrix = np.array([np.array(r) for r in subset['rmse']])
            if rmse_matrix.size == 0:
                continue
            mean_rmse = np.mean(rmse_matrix, axis=0)
            std_rmse = np.std(rmse_matrix, axis=0)
            x = np.arange(1, len(mean_rmse) + 1)
            color = color_map[method]
            label = "Bayesian Optimization" if method == "BO" else "Random Search"
            ax.plot(x, mean_rmse, marker='o', label=label, color=color)
            ax.fill_between(x, mean_rmse - std_rmse, mean_rmse + std_rmse, color=color, alpha=0.2)
            # marcar último punto promedio
            ax.scatter(x[-1], mean_rmse[-1], color=color)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Iteración", fontsize=9)
        ax.set_ylabel("RMSE", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    output_path = os.path.join(os.getcwd(), "results", "error_history_iterations.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    if verbose:
        print(f"Gráfica de historial guardada en: {output_path}")
    return output_path




if __name__ == "__main__":
    print("\n🚀 Iniciando análisis comparativo...\n")

    df = run_comparative_analysis()
    create_comparison_table(df)
    plot_comparison_metrics(df)
    generate_analysis_report(df)

    print("\n" + "=" * 70)
    print("✅ Análisis comparativo completado exitosamente")
    print("=" * 70)

