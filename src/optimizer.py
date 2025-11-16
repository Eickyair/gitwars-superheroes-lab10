import numpy as np
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp, train_and_save_model

# -----------------------------------------------------------
#  Kernel RBF
# -----------------------------------------------------------
def rbf_kernel(x1, x2, length_scale=1.0):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sq_dist = np.sum((x1[:, None] - x2[None, :]) ** 2, axis=2)
    return np.exp(-sq_dist / (2 * length_scale**2))


# -----------------------------------------------------------
#  Fit Gaussian Process
# -----------------------------------------------------------
def fit_gp(X, y, length_scale=1.0, noise=1e-6):
    X = np.atleast_2d(X)
    y = np.atleast_1d(y)

    K = rbf_kernel(X, X, length_scale)
    K_noise = K + noise * np.eye(len(X))

    alpha = np.linalg.solve(K_noise, y)

    return {
        "X_train": X,
        "y_train": y,
        "alpha": alpha,
        "K_noise": K_noise,
        "length_scale": length_scale,
        "noise": noise,
    }


# -----------------------------------------------------------
#  GP Prediction
# -----------------------------------------------------------
def gp_predict(X_train, y_train, X_test, length_scale=1.0, noise=1e-6):
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)

    gp_params = fit_gp(X_train, y_train, length_scale, noise)
    alpha = gp_params["alpha"]
    K_noise = gp_params["K_noise"]

    k_star = rbf_kernel(X_test, X_train, length_scale)
    mu = k_star @ alpha

    k_star_star = rbf_kernel(X_test, X_test, length_scale)

    v = np.linalg.solve(K_noise, k_star.T)
    var = np.diag(k_star_star) - np.sum(k_star * v.T, axis=1)
    var = np.maximum(var, 1e-12)

    return mu, np.sqrt(var)


# -----------------------------------------------------------
#  Acquisition Function UCB (MINIMIZACIÓN)
# -----------------------------------------------------------
def acquisition_ucb(mu, sigma, kappa=2.0):
    """
    Para REGRESIÓN → Queremos MINIMIZAR.
    Usamos LCB = mu - kappa * sigma

    Seleccionamos el mínimo LCB.
    """
    return mu - kappa * sigma


# -----------------------------------------------------------
#  Bayesian Optimization (REGRESIÓN)
# -----------------------------------------------------------
def optimize_model(model_name, n_init=3, n_iter=10, return_history=False, verbose=False,params=None):
    if n_init < 1:
        raise ValueError("n_init debe ser al menos 1")
    # ---- Espacio de parámetros ----
    param_grid = {}
    if model_name == "svm":
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1],
        }
        eval_func = evaluate_svm

    elif model_name == "rf":
        param_grid = {
            "n_estimators": [10,20, 50, 100],
            "max_depth": [2, 4, 6, 8],
        }
        eval_func = evaluate_rf

    elif model_name == "mlp":
        param_grid = {
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "alpha": [1e-4, 1e-3, 1e-2],
        }
        eval_func = evaluate_mlp
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    param_grid = params if params is not None else param_grid

    # ---- Construir combinaciones ----
    from itertools import product

    values = list(param_grid.values())
    combinations = list(product(*values))

    # ---- Convertir parámetros a vector ----
    def params_to_vector(params_tuple):
        vec = []
        for val in params_tuple:
            if isinstance(val, tuple):
                vec.extend(val)
                vec.extend([0] * (2 - len(val)))
            else:
                vec.append(val)
        return np.array(vec, dtype=float)

    # ---- Convertir vector a parámetros ----
    def vector_to_params(vec):
        params = {}
        if model_name == "svm":
            params["C"] = vec[0]
            params["gamma"] = vec[1]

        elif model_name == "rf":
            params["n_estimators"] = int(vec[0])
            params["max_depth"] = int(vec[1])

        elif model_name == "mlp":
            if vec[1] == 0:
                params["hidden_layer_sizes"] = (int(vec[0]),)
            else:
                params["hidden_layer_sizes"] = (int(vec[0]), int(vec[1]))
            params["alpha"] = vec[2]

        return params

    # ---- Inicialización ----
    total = len(combinations)
    if total < n_init:
        raise ValueError("n_init no puede ser mayor que el número total de combinaciones")

    init_idxs = np.random.choice(total, size=min(n_init, total), replace=False)

    X_obs, y_obs, history, idx_obs = [], [], [], []
    if verbose:
        print(f"\n=== Optimization {model_name.upper()} (REGRESIÓN) ===")
        print(f"Total combinaciones: {total}")

    # ---- Random Start ----
    for idx in init_idxs:
        tuple_params = combinations[idx]
        vec = params_to_vector(tuple_params)
        print(vec) if verbose else None
        dict_params = vector_to_params(vec)
        idx_obs.append(idx)
        metric = eval_func(dict_params)  # RMSE

        X_obs.append(vec)
        y_obs.append(metric)
        history.append({"params": dict_params, "rmse": metric})

    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)

    # ---- Bayesian Optimization Loop ----
    for _ in range(n_iter):

        candidates = np.array([params_to_vector(t) for t in combinations])

        # quitar duplicados ya evaluados
        candidates = candidates[~np.array(idx_obs)]

        if len(candidates) == 0:
            # ya se exploro todo el
            # espacio de búsqueda
            break

        mu, sigma = gp_predict(X_obs, y_obs, candidates)

        lcb = acquisition_ucb(mu, sigma)

        best_idx = np.argmin(lcb)
        x_next = candidates[best_idx]
        idx_obs.append(best_idx)

        next_params = vector_to_params(x_next)
        next_rmse = eval_func(next_params)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, next_rmse)

        history.append({"params": next_params, "rmse": next_rmse})

    # ---- Best Result ----
    best_i = np.argmin(y_obs)
    best_rmse = float(y_obs[best_i])
    best_params = history[best_i]["params"]

    if return_history:
        return best_params, best_rmse, history

    return best_params, best_rmse



# Función auxiliar para pruebas
if __name__ == "__main__":
    # Probar con cada modelo
    models = ['svm', 'rf', 'mlp']
    best_model_global,best_metric_global,best_params_global = '', float('inf'), None
    for model in models:
        print(f"\n{'='*60}")
        best_params, best_metric = optimize_model(model, n_init=3, n_iter=3)
        print(f"Modelo {model.upper()}: Mejor métrica = {best_metric:.4f}")
        if best_metric < best_metric_global:
            best_model_global = model
            best_metric_global = best_metric
            best_params_global = best_params
    print(f"\nMejor modelo global: {best_model_global.upper()} con métrica = {best_metric_global:.4f}")