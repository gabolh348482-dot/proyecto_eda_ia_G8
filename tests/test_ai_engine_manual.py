"""Prueba manual/exploratoria del Módulo 3 (ai_engine).

Ejecutar con:
    python tests/test_ai_engine_manual.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ai_engine import ClusteringEngine, DimensionalityReducer  # noqa: E402


def build_dummy_dataset() -> pd.DataFrame:
    """Genera 3 clusters bien separados en 4 dimensiones + ruido disperso."""
    X, _ = make_blobs(
        n_samples=300, centers=3, n_features=4, cluster_std=1.0, random_state=42
    )
    rng = np.random.default_rng(seed=42)
    noise = rng.uniform(low=X.min() - 15, high=X.max() + 15, size=(15, 4))
    X_full = np.vstack([X, noise])

    columns = [f"feature_{i+1}" for i in range(4)]
    return pd.DataFrame(X_full, columns=columns)


def main() -> None:
    df = build_dummy_dataset()
    numeric_columns = list(df.columns)

    print("=" * 70)
    print("PASO 1: BÚSQUEDA AUTOMÁTICA DE K ÓPTIMO (K-Means)")
    print("=" * 70)
    engine = ClusteringEngine(df, numeric_columns)
    search_result = engine.find_optimal_k(k_min=2, k_max=8)
    print(search_result.summary)
    print(f"\nK sugerido automáticamente: {search_result.suggested_k}")

    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR K-MEANS CON EL K SUGERIDO")
    print("=" * 70)
    kmeans_result = engine.run_kmeans(k=search_result.suggested_k)
    print(f"Clusters encontrados: {kmeans_result.n_clusters}")
    print(f"Silhouette score: {kmeans_result.silhouette}")
    print(f"Tamaño de cada cluster:\n{pd.Series(kmeans_result.labels).value_counts()}")

    print("\n" + "=" * 70)
    print("PASO 3: SUGERENCIA AUTOMÁTICA DE PARÁMETROS PARA DBSCAN")
    print("=" * 70)
    dbscan_params = engine.suggest_dbscan_params()
    print(dbscan_params)

    dbscan_result = engine.run_dbscan(**dbscan_params)
    print(f"\nClusters encontrados (sin contar ruido): {dbscan_result.n_clusters}")
    print(f"Puntos de ruido detectados: {dbscan_result.n_noise}")
    print(f"Silhouette score (excluyendo ruido): {dbscan_result.silhouette}")

    print("\n" + "=" * 70)
    print("PASO 4: REDUCCIÓN DE DIMENSIONALIDAD CON PCA (2D)")
    print("=" * 70)
    reducer = DimensionalityReducer(n_components=2)
    pca_result = reducer.fit_transform(df, numeric_columns)
    print(f"Varianza explicada por componente: {pca_result.explained_variance_ratio.round(4)}")
    print(f"Varianza total explicada (2D): {pca_result.total_variance_explained}%")
    print("\nLoadings (peso de cada variable original en cada componente):")
    print(pca_result.loadings)
    print("\nPrimeras filas de las componentes principales:")
    print(pca_result.components_df.head())

    # --- Validaciones puntuales ---
    print("\n" + "=" * 70)
    print("VALIDACIÓN AUTOMÁTICA")
    print("=" * 70)

    ok_k = search_result.suggested_k == 3
    print(f"[{'OK' if ok_k else 'FALLÓ'}] K sugerido == 3 (dataset generado con 3 centros) "
          f"-> obtenido: {search_result.suggested_k}")

    ok_sil_kmeans = kmeans_result.silhouette > 0.5
    print(f"[{'OK' if ok_sil_kmeans else 'FALLÓ'}] Silhouette de K-Means > 0.5 "
          f"(clusters bien separados) -> obtenido: {kmeans_result.silhouette}")

    ok_dbscan_clusters = dbscan_result.n_clusters == 3
    print(f"[{'OK' if ok_dbscan_clusters else 'FALLÓ'}] DBSCAN encuentra 3 clusters "
          f"-> obtenido: {dbscan_result.n_clusters}")

    ok_dbscan_noise = dbscan_result.n_noise >= 10
    print(f"[{'OK' if ok_dbscan_noise else 'FALLÓ'}] DBSCAN detecta >=10 puntos de ruido "
          f"(se inyectaron 15) -> obtenido: {dbscan_result.n_noise}")

    ok_pca_variance = pca_result.total_variance_explained > 60.0
    print(f"[{'OK' if ok_pca_variance else 'FALLÓ'}] PCA 2D retiene >60% de la varianza "
          f"-> obtenido: {pca_result.total_variance_explained}%")


if __name__ == "__main__":
    main()
