# Análisis Comparativo: Optimización Bayesiana vs Random Search (Regresión)

Fecha de ejecución: 2025-11-16 07:34:08

---

## 1. Tabla Comparativa

| Modelo | Método | Hiperparámetros | RMSE | Mejora BO (%) |
|--------|---------|------------------|--------|----------------|
| SVR | BO | {'C': np.float64(6293.677126160469), 'gamma': np.float64(4.405297405361181)} | 14.3430 | - |
| SVR | RS | {'C': np.float64(1106.4082057592746), 'gamma': np.float64(2.297912698062223)} | 13.8862 | -3.29% |
| Random Forest Regressor | BO | {'n_estimators': 192, 'max_depth': 10} | 15.0175 | - |
| Random Forest Regressor | RS | {'n_estimators': np.int64(191), 'max_depth': np.int64(10)} | 15.0251 | 0.05% |
| MLP Regressor | BO | {'hidden_layer_sizes': (32, 16), 'alpha': np.float64(14.587505983128398)} | 28.1919 | - |
| MLP Regressor | RS | {'hidden_layer_sizes': (32, 16), 'alpha': np.float64(14.587505983128398)} | 28.1919 | 0.00% |


---

## 2. Conclusiones

- La métrica usada es **RMSE**, menor = mejor.
- El mejor modelo según BO fue **SVR**, con RMSE = 14.3430.
- En general, BO logra valores más bajos de RMSE usando menos evaluaciones.

---

