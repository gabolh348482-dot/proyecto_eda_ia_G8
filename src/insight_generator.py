"""
Módulo 4: Generador de insights textuales automatizados.

Consume directamente los objetos que ya devuelven los módulos 2 y 3
(EDAEngine, OutlierReport, KMeansSearchResult, ClusteringResult, PCAResult).
"""

from __future__ import annotations


class InsightGenerator:
    def __init__(
        self,
        eda_engine=None,            
        outlier_report=None,        
        kmeans_search=None,         
        clustering_result=None,     
        pca_result=None,            
        corr_threshold: float = 0.7,
        null_threshold_pct: float = 30.0,
        dominant_category_threshold_pct: float = 70.0,
    ):
        self.eda_engine = eda_engine
        self.outlier_report = outlier_report
        self.kmeans_search = kmeans_search
        self.clustering_result = clustering_result
        self.pca_result = pca_result
        self.corr_threshold = corr_threshold
        self.null_threshold_pct = null_threshold_pct
        self.dominant_category_threshold_pct = dominant_category_threshold_pct

    def generate_all(self) -> list[dict]:
        insights: list[dict] = []
        if self.eda_engine is not None:
            insights += self._correlation_insights()
            insights += self._null_insights()
            insights += self._categorical_insights()
        if self.outlier_report is not None:
            insights += self._outlier_insights()
        if self.kmeans_search is not None or self.clustering_result is not None:
            insights += self._clustering_insights()
        if self.pca_result is not None:
            insights += self._pca_insights()
        return insights

    def _correlation_insights(self) -> list[dict]:
        results = []
        corr = self.eda_engine.correlation_matrix()
        if corr.empty:
            return results
        cols = corr.columns
        seen = set()
        for c1 in cols:
            for c2 in cols:
                if c1 == c2 or frozenset((c1, c2)) in seen:
                    continue
                seen.add(frozenset((c1, c2)))
                r = corr.loc[c1, c2]
                if abs(r) >= self.corr_threshold:
                    fuerza = "fuerte" if abs(r) >= 0.85 else "moderada-alta"
                    signo = "positiva" if r > 0 else "negativa"
                    texto = (f"Correlación {fuerza} {signo} (r={r:.2f}) entre "
                             f"'{c1}' y '{c2}'. Es una asociación estadística, "
                             f"no implica causalidad.")
                    results.append({
                        "type": "correlation", "columns": [c1, c2],
                        "value": float(r), "text": texto,
                        "severity": "high" if abs(r) >= 0.85 else "medium",
                    })
        return results

    def _null_insights(self) -> list[dict]:
        results = []
        nulls = self.eda_engine.missing_value_summary()
        for _, row in nulls.iterrows():
            if row["porcentaje_nulos"] >= self.null_threshold_pct:
                texto = (f"La columna '{row['columna']}' ({row['tipo_detectado']}) "
                         f"tiene {row['porcentaje_nulos']}% de valores nulos "
                         f"({row['n_nulos']} filas), un nivel considerable que "
                         f"puede requerir imputación o descartar la columna.")
                results.append({
                    "type": "null", "column": row["columna"],
                    "pct": float(row["porcentaje_nulos"]), "text": texto,
                    "severity": "high" if row["porcentaje_nulos"] >= 50 else "medium",
                })
        return results

    def _categorical_insights(self) -> list[dict]:
        """Detecta desbalance de clases: una sola categoría concentrando la
        mayoría de los registros. Relevante tanto para el EDA (una variable
        con 95% en un solo valor aporta poca información discriminante) como
        para un eventual modelo entrenado sobre esa columna (riesgo de sesgo
        hacia la clase mayoritaria).
        """
        results = []
        cat_summaries = self.eda_engine.categorical_summary()
        for col, table in cat_summaries.items():
            if table.empty:
                continue
            top_row = table.iloc[0]
            top_pct = float(top_row["porcentaje"])
            if top_pct >= self.dominant_category_threshold_pct:
                texto = (f"La columna '{col}' está dominada por la categoría "
                         f"'{top_row['categoria']}', que representa el {top_pct}% "
                         f"de los registros. Esto indica un fuerte desbalance de "
                         f"clases en esa variable.")
                results.append({
                    "type": "categorical_imbalance", "column": col,
                    "dominant_category": str(top_row["categoria"]),
                    "pct": top_pct, "text": texto,
                    "severity": "high" if top_pct >= 90 else "medium",
                })
        return results

    def _outlier_insights(self) -> list[dict]:
        results = []
        summary = self.outlier_report.summary
        for _, row in summary.iterrows():
            if row["n_outliers"] == 0:
                continue
            texto = (f"{row['metodo']}: {row['n_outliers']} filas marcadas "
                     f"como outlier ({row['porcentaje']}% del dataset).")
            results.append({
                "type": "outlier", "method": row["metodo"],
                "n_outliers": int(row["n_outliers"]),
                "pct": float(row["porcentaje"]), "text": texto,
                "severity": "medium",
            })

        per_col = self.outlier_report.iqr_flags.sum()
        for col, n in per_col.items():
            if n > 0:
                texto = f"IQR detectó {int(n)} outliers específicamente en la columna '{col}'."
                results.append({
                    "type": "outlier_column", "column": col, "n_outliers": int(n),
                    "text": texto, "severity": "low",
                })
        return results

    def _clustering_insights(self) -> list[dict]:
        results = []
        if self.kmeans_search is not None:
            texto = (f"El número óptimo de clusters sugerido (codo + silhouette) "
                     f"es k={self.kmeans_search.suggested_k}.")
            results.append({
                "type": "clustering_k", "suggested_k": self.kmeans_search.suggested_k,
                "text": texto, "severity": "info",
            })
        if self.clustering_result is not None:
            cr = self.clustering_result
            calidad = "buena separación" if cr.silhouette and cr.silhouette >= 0.5 else "separación moderada/baja"
            extra = f", {cr.n_noise} puntos de ruido" if cr.algorithm == "dbscan" else ""
            sil_txt = f"{cr.silhouette:.2f}" if cr.silhouette is not None else "no calculable"
            texto = (f"El algoritmo {cr.algorithm.upper()} encontró {cr.n_clusters} "
                     f"clusters (silhouette={sil_txt} — {calidad}){extra}.")
            results.append({
                "type": "clustering_result", "algorithm": cr.algorithm,
                "n_clusters": cr.n_clusters, "silhouette": cr.silhouette,
                "text": texto, "severity": "info",
            })
        return results

    def _pca_insights(self) -> list[dict]:
        results = []
        pr = self.pca_result
        texto = (f"Los primeros {pr.components_df.shape[1]} componentes principales "
                 f"explican {pr.total_variance_explained}% de la varianza total.")
        results.append({
            "type": "pca_variance", "variance_explained": pr.total_variance_explained,
            "text": texto, "severity": "info",
        })
        if "PC1" in pr.loadings.columns:
            top_var = pr.loadings["PC1"].abs().idxmax()
            peso = pr.loadings.loc[top_var, "PC1"]
            texto2 = (f"La variable con mayor peso en el primer componente (PC1) "
                      f"es '{top_var}' (loading={peso}).")
            results.append({
                "type": "pca_loading", "component": "PC1", "variable": top_var,
                "loading": float(peso), "text": texto2, "severity": "info",
            })
        return results