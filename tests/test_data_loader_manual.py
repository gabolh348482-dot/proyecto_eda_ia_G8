"""Prueba manual/exploratoria del Módulo 1 (data_loader).

Ejecutar con:
    python tests/test_data_loader_manual.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import ColumnTypeDetector, load_and_profile  # noqa: E402


def build_dummy_dataset(path: Path) -> None:
    """Genera un CSV sintético con tipos de columna variados y "sucios"."""
    rng = np.random.default_rng(seed=42)
    n = 300

    df = pd.DataFrame(
        {
            "id_cliente": [f"CUST-{i:05d}" for i in range(n)],          # identifier
            "fecha_compra": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),  # datetime (texto)
            "monto_compra": rng.normal(150, 40, n).round(2),             # numeric_continuous
            "num_items": rng.integers(1, 8, n),                          # numeric_discrete
            "categoria_producto": rng.choice(["Electrónica", "Ropa", "Hogar", "Deportes"], n),  # categorical
            "es_recurrente": rng.choice(["Sí", "No"], n),                # boolean codificada
            "comentario_libre": [f"Cliente comentó detalles variados #{i}" for i in range(n)],  # text
            "region_constante": ["Costa Rica"] * n,                      # constant
        }
    )

    # Inyectamos algunos nulos a propósito
    df.loc[rng.choice(n, 15, replace=False), "monto_compra"] = np.nan

    df.to_csv(path, index=False)


def main() -> None:
    dummy_path = Path(__file__).parent / "_dummy_dataset.csv"
    build_dummy_dataset(dummy_path)

    # --- Prueba 1: carga + perfilado en un solo paso ---
    df, overview = load_and_profile(dummy_path)

    print("=" * 70)
    print("OVERVIEW GENERAL DEL DATASET")
    print("=" * 70)
    print(f"Filas: {overview.n_rows} | Columnas: {overview.n_columns}")
    print(f"Memoria: {overview.memory_usage_mb} MB")
    print(f"Filas duplicadas: {overview.n_duplicated_rows}")
    print(f"% nulos promedio: {overview.total_missing_ratio * 100:.2f}%")

    print("\nConteo de columnas por tipo detectado:")
    for tipo, count in overview.type_counts().items():
        print(f"  - {tipo}: {count}")

    print("\n" + "=" * 70)
    print("DETALLE POR COLUMNA")
    print("=" * 70)
    for name, profile in overview.columns.items():
        print(
            f"{name:22s} | detectado: {profile.detected_type.value:20s} | "
            f"dtype_original: {profile.dtype_original:10s} | "
            f"únicos: {profile.n_unique:4d} | nulos: {profile.missing_ratio*100:5.2f}%"
        )

    # --- Prueba 2: validación puntual de casos esperados ---
    expected = {
        "id_cliente": "identifier",
        "fecha_compra": "datetime",
        "monto_compra": "numeric_continuous",
        "num_items": "numeric_discrete",
        "categoria_producto": "categorical",
        "es_recurrente": "boolean",
        "comentario_libre": "text",
        "region_constante": "constant",
    }

    print("\n" + "=" * 70)
    print("VALIDACIÓN AUTOMÁTICA DE CASOS ESPERADOS")
    print("=" * 70)
    all_ok = True
    for col, expected_type in expected.items():
        actual_type = overview.columns[col].detected_type.value
        status = "OK" if actual_type == expected_type else "FALLÓ"
        if actual_type != expected_type:
            all_ok = False
        print(f"  [{status}] {col}: esperado={expected_type} | obtenido={actual_type}")

    print("\nResultado final:", "TODAS LAS PRUEBAS PASARON ✅" if all_ok else "HAY FALLOS ❌")

    dummy_path.unlink()  # limpieza


if __name__ == "__main__":
    main()
