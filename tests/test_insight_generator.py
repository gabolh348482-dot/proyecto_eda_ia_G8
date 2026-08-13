"""
Test end-to-end del Módulo 4: corre el pipeline completo
(1 -> 2 -> 3 -> 4) sobre un dataset sintético con problemas
conocidos a propósito, y verifica que cada regla dispare.
"""
import numpy as np
import pandas as pd

from src.data_loader import ColumnTypeDetector
from src.eda_engine import EDAEngine, OutlierDetector
from src.ai_engine import ClusteringEngine, DimensionalityReducer
from src.insight_generator import InsightGenerator

np.random.seed(42)
n = 300

x = np.random.normal(50, 10, n)
y = x * 1.5 + np.random.normal(0, 3, n)

z = np.random.normal(20, 5, n)
z[:5] = [200, 210, 195, 220, 205]

categoria = np.random.choice(["A", "B", "C"], size=n, p=[0.95, 0.03, 0.02])

con_nulos = np.random.normal(0, 1, n)
mask_nulos = np.random.choice([True, False], size=n, p=[0.4, 0.6])
con_nulos[mask_nulos] = np.nan

df = pd.DataFrame({"x": x, "y": y, "z": z, "categoria": categoria, "con_nulos": con_nulos})

overview = ColumnTypeDetector().profile_dataset(df)
eda = EDAEngine(df, overview)
outlier_report = OutlierDetector(df, eda.numeric_columns).compare_methods()

clustering_engine = ClusteringEngine(df, eda.numeric_columns)
kmeans_search = clustering_engine.find_optimal_k()
clustering_result = clustering_engine.run_kmeans(kmeans_search.suggested_k)
pca_result = DimensionalityReducer().fit_transform(df, eda.numeric_columns)

generator = InsightGenerator(
    eda_engine=eda,
    outlier_report=outlier_report,
    kmeans_search=kmeans_search,
    clustering_result=clustering_result,
    pca_result=pca_result,
)
insights = generator.generate_all()
for ins in insights:
    print(f"[{ins['severity']:>6}] ({ins['type']}) {ins['text']}")

# --- Verificaciones automáticas ---
tipos_generados = {i["type"] for i in insights}
assert "correlation" in tipos_generados, "FALLO: no detectó la correlación fuerte x-y"
assert "null" in tipos_generados, "FALLO: no detectó la columna con 40% de nulos"
assert "categorical_imbalance" in tipos_generados, "FALLO: no detectó el desbalance de categoría"
assert "outlier" in tipos_generados or "outlier_column" in tipos_generados, "FALLO: no detectó outliers"
assert "clustering_k" in tipos_generados, "FALLO: no generó insight de clustering"
assert "pca_variance" in tipos_generados, "FALLO: no generó insight de PCA"

print("\n✅ TODAS LAS VERIFICACIONES PASARON")