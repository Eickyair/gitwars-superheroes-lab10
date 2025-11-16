#!/usr/bin/env python3

import sys
import os
import importlib.util
import numpy as np

SEMILLA = 1024
np.random.seed(SEMILLA)
# Ruta absoluta del directorio donde está este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def import_local_module(module_name):
    """
    Importa un módulo .py que esté en la MISMA carpeta que run_analysis.py.
    Esto evita problemas de rutas cuando se ejecuta en Docker o Linux.
    """
    module_path = os.path.join(BASE_DIR, f"{module_name}.py")

    if not os.path.isfile(module_path):
        print(f"❌ No se encontró {module_name}.py en: {BASE_DIR}")
        return None

    print(f"📌 Importando módulo desde: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    print("\n=== EJECUTANDO ANÁLISIS COMPARATIVO (REGRESIÓN - RMSE) ===\n")

    np.random.seed(42)

    # Importar comparative_analysis.py desde la misma carpeta
    comparative = import_local_module("comparative_analysis")

    if comparative is None:
        print("\n ERROR: No se pudo cargar comparative_analysis.py")
        return False

    # Ejecutar análisis
    print("\n Ejecutando análisis comparativo...")
    results_df,histories_df = comparative.run_comparative_analysis()
    comparative.plot_error_history_iterations(histories_df)
    print("\n Creando tabla comparativa...")
    comparative.create_comparison_table(results_df)

    print("\n Generando gráficos...")
    comparative.plot_comparison_metrics(results_df)

    print("\n Generando reporte...")
    comparative.generate_analysis_report(results_df)

    print("\n=== ANÁLISIS COMPLETO ===\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


