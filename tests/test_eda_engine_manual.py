"""Prueba manual/exploratoria del Módulo 2 (eda_engine).

Ejecutar con:
    python tests/test_eda_engine_manual.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import ColumnTypeDetector  # noqa: E402
from src.eda_engine import EDAEngine, OutlierDetector  # noqa: E402


def build_dummy_dataset() -> pd.DataFrame:
    """Genera un dataset con outliers univariados y multivariados conocidos.

    - `ingreso_mensual`: tiene 5 outliers univariados obvios (valores muy
      altos), que el IQR debería detectar sin problema.
    - `edad` y `gasto_mensual`: normales por separado, pero se inyecta 1
      fila donde la combinación es rara (edad muy baja + gasto muy alto),
      que el IQR NO debería marcar (cada valor está dentro de rango en su
      propia columna) pero Isolation Forest sí, por ser multivariado.
    """
    rng = np.random.default_rng(seed=7)
    n = 400

    edad = rng.integers(18, 70, n).astype(float)
    gasto_mensual = (edad * 15 + rng.normal(0, 50, n)).round(2)  # correlacionado con edad
    ingreso_mensual = rng.normal(800_000, 150_000, n).round(2)
    categoria = rng.choice(["A", "B", "C"], n)
    activo = rng.choice(["Sí", "No"], n)

    df = pd.DataFrame(
        {
            "edad": edad,
            "gasto_mensual": gasto_mensual,
            "ingreso_mensual": ingreso_mensual,
            "categoria": categoria,
            "activo": activo,
        }
    )

    # Outliers univariados obvios en ingreso_mensual
    outlier_idx = rng.choice(n, 5, replace=False)
    df.loc[outlier_idx, "ingreso_mensual"] = [5_000_000, 4_800_000, 6_000_000, 5_500_000, 4_900_000]

    # Outlier multivariado sutil: edad baja + gasto alto (raro dado que
    # gasto ~ edad*15), pero cada valor por separado está dentro de rango.
    df.loc[n - 1, "edad"] = 20.0             # dentro de rango normal de edad
    df.loc[n - 1, "gasto_mensual"] = 950.0   # dentro de rango normal de gasto

    # Nulos a propósito para probar missing_value_summary
    df.loc[rng.choice(n, 10, replace=False), "gasto_mensual"] = np.nan

    return df


def main() -> None:
    df = build_dummy_dataset()
    overview = ColumnTypeDetector().profile_dataset(df)

    engine = EDAEngine(df, overview)

    print("=" * 70)
    print("RESUMEN NUMÉRICO")
    print("=" * 70)
    print(engine.numeric_summary())

    print("\n" + "=" * 70)
    print("RESUMEN CATEGÓRICO")
    print("=" * 70)
    for col, table in engine.categorical_summary().items():
        print(f"\n-- {col} --")
        print(table)

    print("\n" + "=" * 70)
    print("MATRIZ DE CORRELACIÓN")
    print("=" * 70)
    print(engine.correlation_matrix())

    print("\n" + "=" * 70)
    print("NULOS")
    print("=" * 70)
    print(engine.missing_value_summary())

    print("\n" + "=" * 70)
    print("DETECCIÓN DE OUTLIERS: IQR vs. ISOLATION FOREST")
    print("=" * 70)
    detector = OutlierDetector(df, numeric_columns=engine.numeric_columns)
    report = detector.compare_methods()
    print(report.summary)

    # --- Validaciones puntuales ---
    print("\n" + "=" * 70)
    print("VALIDACIÓN AUTOMÁTICA")
    print("=" * 70)

    ok_iqr = report.iqr_flags["ingreso_mensual"].sum() >= 5
    print(f"[{'OK' if ok_iqr else 'FALLÓ'}] IQR detecta >=5 outliers en ingreso_mensual "
          f"(detectó {report.iqr_flags['ingreso_mensual'].sum()})")

    fila_multivariada = df.index[-1]
    marcada_por_if = bool(report.isolation_forest_flags.loc[fila_multivariada])
    marcada_por_iqr = bool(report.iqr_flags.loc[fila_multivariada].any())
    print(f"[{'OK' if marcada_por_if else 'INFO'}] Isolation Forest marca la fila "
          f"multivariada inyectada (idx={fila_multivariada}): {marcada_por_if}")
    print(f"[INFO] Esa misma fila NO fue marcada por IQR (esperado): {not marcada_por_iqr}")


if __name__ == "__main__":
    main()
