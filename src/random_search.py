import numpy as np
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp

def random_search_optimize(model_name, n_iterations=15, params=None, verbose=False):
    """
    Optimiza hiperparámetros usando Random Search (REGRESIÓN).
    
    Objetivo: MINIMIZAR RMSE.
    
    Args:
        model_name: Nombre del modelo ('svm', 'rf', 'mlp')
        n_iterations: Número de combinaciones aleatorias a probar
    
    Returns:
        best_params: Mejor configuración encontrada (menor RMSE)
        best_metric: RMSE mínimo alcanzado
        history: Lista de tuplas (params, rmse)
    """

    # --------------------------------------------------------------
    # Espacio de hiperparámetros dependiendo del modelo
    # --------------------------------------------------------------
    if model_name == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        eval_func = evaluate_svm

    elif model_name == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10]
        }
        eval_func = evaluate_rf

    elif model_name == 'mlp':
        param_grid = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16)],
            'alpha': [1e-4, 1e-3, 1e-2]
        }
        eval_func = evaluate_mlp

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    param_grid = params if params is not None else param_grid
    # --------------------------------------------------------------
    # Crear todas las combinaciones posibles
    # --------------------------------------------------------------
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    if verbose:
        print(f"\n=== Optimizando {model_name.upper()} con Random Search (REGRESIÓN) ===")
        print(f"Espacio de búsqueda: {len(all_combinations)} combinaciones")
        print(f"Iteraciones a evaluar: {n_iterations}")
    
    # Seleccionar combinaciones aleatorias
    n_samples = min(n_iterations, len(all_combinations))
    sampled_indices = np.random.choice(len(all_combinations), size=n_samples, replace=False)
    
    # --------------------------------------------------------------
    # Inicializar mejor resultado (NOW WE MINIMIZE RMSE)
    # --------------------------------------------------------------
    best_params = None
    best_metric = np.inf    # RMSE → minimizar
    history = []
    
    print("\n--- Evaluaciones Random Search ---") if verbose else None
    for i, idx in enumerate(sampled_indices, 1):

        params_tuple = all_combinations[idx]
        params_dict = dict(zip(param_names, params_tuple))
        
        # Evaluar el modelo → devuelve RMSE
        rmse = eval_func(params_dict)
        history.append((params_dict.copy(), rmse))
        
        # Guardar progreso
        print(f"Iteración {i}: {params_dict} -> RMSE: {rmse:.4f}") if verbose else None

        # Actualizar mejor configuración (menor RMSE)
        if rmse < best_metric:
            best_metric = rmse
            best_params = params_dict.copy()

    if verbose:
        print(f"\n=== Mejor configuración encontrada (REGRESIÓN) ===")
        print(f"Parámetros: {best_params}")
        print(f"RMSE mínimo: {best_metric:.4f}")
    
    return best_params, best_metric, history


# --------------------------------------------------------------
# Prueba rápida
# --------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    models = ['svm', 'rf', 'mlp']

    for model in models:
        print(f"\n{'='*60}")
        best_params, best_metric, history = random_search_optimize(model, n_iterations=15)
        print(f"Modelo {model.upper()}: Mejor RMSE = {best_metric:.4f}")
