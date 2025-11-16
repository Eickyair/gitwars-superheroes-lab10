import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# ============================
#  Cargar Datos
# ============================

def load_data():
    """
    Carga dataset y prepara X e y para REGRESIÓN.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "../data/data.csv"))

    X = df.drop("power", axis=1).values
    y = df["power"].values 

    return X, y


X_global, y_global = load_data()


# ============================
#  Evaluaciones
# ============================

def evaluate_svm(params):
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(C=params["C"], gamma=params["gamma"])
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse  # minimizar


def evaluate_rf(params):
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


def evaluate_mlp(params):
    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        max_iter=100,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


# ============================
#  Entrenar y Guardar Modelo
# ============================

def train_and_save_model(model_name, best_params, output_path="best_model.pkl"):
    X = X_global
    y = y_global

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_name == "svm":
        model = SVR(**best_params)
    elif model_name == "rf":
        model = RandomForestRegressor(**best_params, random_state=42)
    elif model_name == "mlp":
        model = MLPRegressor(**best_params, max_iter=400, random_state=42)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    model.fit(X_train_scaled, y_train)

    joblib.dump({"model": model, "scaler": scaler, 'params': best_params}, output_path)

    print(f"✅ Modelo guardado correctamente en {output_path}")

    return model, scaler


# Función de prueba
if __name__ == "__main__":
    print("Probando funciones de evaluación (REGRESIÓN)...")

    # Probar SVM (SVR)
    svm_params = {'C': 1.0, 'gamma': 0.01}
    svm_rmse = evaluate_svm(svm_params)
    print(f"\nSVR con {svm_params}: RMSE = {svm_rmse:.4f}")

    # Probar Random Forest Regressor
    rf_params = {'n_estimators': 100, 'max_depth': 6}
    rf_rmse = evaluate_rf(rf_params)
    print(f"RandomForestRegressor con {rf_params}: RMSE = {rf_rmse:.4f}")

    # Probar MLP Regressor
    mlp_params = {'hidden_layer_sizes': (32,), 'alpha': 0.001}
    mlp_rmse = evaluate_mlp(mlp_params)
    print(f"MLPRegressor con {mlp_params}: RMSE = {mlp_rmse:.4f}")


