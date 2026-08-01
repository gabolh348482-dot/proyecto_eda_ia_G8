"""
ai_engine.py
============

Módulo 3: Motor de IA — Clustering Dinámico & Reducción de Dimensionalidad.

Componentes:
    - `DimensionalityReducer`: reduce variables numéricas a 2 (o más)
      componentes principales vía PCA, para poder visualizar clusters de
      datasets con muchas dimensiones.
    - `ClusteringEngine`: entrena K-Means (con selección automática de k
      mediante método del codo + silhouette score) y DBSCAN (con selección
      automática de `eps` mediante el gráfico de k-distancias).

No depende de Streamlit ni de ningún framework de UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Utilidad genérica: detección automática de "codo" en una curva
# --------------------------------------------------------------------------- #
def _find_elbow_index(y_values: np.ndarray) -> int:
    """Encuentra el índice del punto de codo en una curva monótona.

    Implementación del método clásico de máxima distancia perpendicular:
    se traza una línea recta entre el primer y el último punto de la curva,
    y se elige el punto con mayor distancia perpendicular a esa línea. Es
    la misma idea detrás de librerías como `kneed`, sin agregar esa
    dependencia externa.

    Args:
        y_values: Array 1D con los valores de la curva (ya ordenados según
            el eje X, por ejemplo, inercia por cada k, o distancias k-NN
            ordenadas ascendentemente).

    Returns:
        Índice (posición en el array) del punto de codo detectado.
    """
    n_points = len(y_values)
    if n_points < 3:
        return 0

    x_values = np.arange(n_points)
    # Normalizar ambos ejes a [0, 1] para que la geometría no dependa de escalas
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min())
    y_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for xi, yi in zip(x_norm, y_norm):
        point = np.array([xi, yi]) - p1
        projection_length = np.dot(point, line_vec_norm)
        projection = projection_length * line_vec_norm
        perpendicular = point - projection
        distances.append(np.linalg.norm(perpendicular))

    return int(np.argmax(distances))


def _prepare_numeric_matrix(df: pd.DataFrame, numeric_columns: list[str]) -> np.ndarray:
    """Imputa (mediana) y escala (estandariza) las columnas numéricas dadas.

    Se usa tanto en clustering como en PCA para garantizar que ambos
    trabajen sobre exactamente los mismos datos preprocesados.
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df[numeric_columns])
    scaler = StandardScaler()
    return scaler.fit_transform(X_imputed)


# --------------------------------------------------------------------------- #
# Reducción de dimensionalidad
# --------------------------------------------------------------------------- #
@dataclass
class PCAResult:
    """Resultado de aplicar PCA a un conjunto de variables numéricas.

    Attributes:
        components_df: DataFrame con las columnas `PC1`, `PC2`, ... y el
            mismo índice que el DataFrame original.
        explained_variance_ratio: Proporción de varianza explicada por cada
            componente.
        loadings: DataFrame que muestra cuánto pesa cada variable original
            en cada componente principal (útil para interpretar clusters).
    """

    components_df: pd.DataFrame
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame

    @property
    def total_variance_explained(self) -> float:
        """Porcentaje total de varianza retenida por los componentes generados."""
        return round(float(self.explained_variance_ratio.sum()) * 100, 2)


class DimensionalityReducer:
    """Aplica PCA sobre columnas numéricas para visualización o preprocesamiento.

    Args:
        n_components: Número de componentes principales a generar
            (2 es el estándar para graficar clusters en un plano).
        random_state: Semilla para reproducibilidad.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, df: pd.DataFrame, numeric_columns: list[str]) -> PCAResult:
        """Ajusta PCA y proyecta los datos a `n_components` dimensiones.

        Args:
            df: DataFrame original.
            numeric_columns: Columnas numéricas a usar como input de PCA.

        Returns:
            `PCAResult` con las componentes, la varianza explicada y las
            loadings por variable original.
        """
        if len(numeric_columns) < self.n_components:
            raise ValueError(
                f"Se requieren al menos {self.n_components} columnas numéricas "
                f"para generar {self.n_components} componentes; se recibieron "
                f"{len(numeric_columns)}."
            )

        X_scaled = _prepare_numeric_matrix(df, numeric_columns)
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        components = pca.fit_transform(X_scaled)

        component_names = [f"PC{i + 1}" for i in range(self.n_components)]
        components_df = pd.DataFrame(components, columns=component_names, index=df.index)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=numeric_columns,
            columns=component_names,
        ).round(3)

        return PCAResult(
            components_df=components_df,
            explained_variance_ratio=pca.explained_variance_ratio_,
            loadings=loadings,
        )


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
@dataclass
class KMeansSearchResult:
    """Resultado de la búsqueda automática del número óptimo de clusters (k).

    Attributes:
        summary: DataFrame con columnas `k`, `inertia`, `silhouette` para
            cada valor de k evaluado (insumo directo para graficar el codo).
        suggested_k: Valor de k sugerido automáticamente.
    """

    summary: pd.DataFrame
    suggested_k: int


@dataclass
class ClusteringResult:
    """Resultado de correr un algoritmo de clustering.

    Attributes:
        algorithm: Nombre del algoritmo usado (`"kmeans"` o `"dbscan"`).
        labels: Array con la etiqueta de cluster asignada a cada fila.
            En DBSCAN, `-1` indica ruido (no pertenece a ningún cluster).
        n_clusters: Número de clusters encontrados (sin contar el ruido).
        n_noise: Número de puntos clasificados como ruido (0 en K-Means).
        silhouette: Silhouette score global (None si no se puede calcular,
            por ejemplo con 1 solo cluster).
        params: Parámetros usados para entrenar el modelo.
    """

    algorithm: str
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette: float | None
    params: dict


class ClusteringEngine:
    """Motor de clustering dinámico sobre variables numéricas.

    Args:
        df: DataFrame original.
        numeric_columns: Columnas numéricas a usar para agrupar. Se
            recomienda excluir identificadores y, opcionalmente, filas
            marcadas como outliers extremos (ver Módulo 2).
    """

    def __init__(self, df: pd.DataFrame, numeric_columns: list[str]) -> None:
        if len(numeric_columns) < 2:
            raise ValueError("Se requieren al menos 2 columnas numéricas para clustering.")
        self.df = df
        self.numeric_columns = numeric_columns
        self.X = _prepare_numeric_matrix(df, numeric_columns)

    def find_optimal_k(self, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> KMeansSearchResult:
        """Evalúa un rango de valores de k y sugiere el óptimo automáticamente.

        Combina el método del codo (inercia) con el silhouette score: el k
        sugerido es el punto de codo de la curva de inercia, pero si el
        silhouette score de un k cercano es claramente superior, se
        prioriza este último (evita sugerir un codo poco informativo en
        datasets sin estructura de cluster clara).

        Args:
            k_min: Mínimo número de clusters a evaluar (>= 2).
            k_max: Máximo número de clusters a evaluar.
            random_state: Semilla para reproducibilidad.

        Returns:
            `KMeansSearchResult` con la tabla comparativa y el k sugerido.
        """
        k_max = min(k_max, len(self.df) - 1)
        k_values = list(range(k_min, k_max + 1))

        inertias, silhouettes = [], []
        for k in k_values:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(self.X)
            inertias.append(model.inertia_)
            silhouettes.append(silhouette_score(self.X, labels))

        summary = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes})

        elbow_idx = _find_elbow_index(np.array(inertias))
        best_silhouette_idx = int(np.argmax(silhouettes))

        # Si el mejor silhouette es notablemente mejor que el del codo, se prioriza.
        suggested_idx = (
            best_silhouette_idx
            if silhouettes[best_silhouette_idx] > silhouettes[elbow_idx] + 0.05
            else elbow_idx
        )
        suggested_k = k_values[suggested_idx]

        return KMeansSearchResult(summary=summary.round(4), suggested_k=suggested_k)

    def run_kmeans(self, k: int, random_state: int = 42) -> ClusteringResult:
        """Entrena K-Means con un número de clusters específico.

        Args:
            k: Número de clusters.
            random_state: Semilla para reproducibilidad.

        Returns:
            `ClusteringResult` con las etiquetas asignadas y métricas.
        """
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(self.X)
        sil = silhouette_score(self.X, labels) if k > 1 else None

        return ClusteringResult(
            algorithm="kmeans",
            labels=labels,
            n_clusters=k,
            n_noise=0,
            silhouette=round(sil, 4) if sil is not None else None,
            params={"k": k, "random_state": random_state},
        )

    def suggest_dbscan_params(self, min_samples: int | None = None) -> dict:
        """Sugiere `eps` y `min_samples` para DBSCAN de forma automática.

        `min_samples` usa la heurística estándar `2 * n_features` si no se
        especifica. `eps` se estima con el gráfico de k-distancias: se
        calcula la distancia de cada punto a su k-ésimo vecino más cercano
        (k = min_samples), se ordenan ascendentemente, y se busca el punto
        de codo de esa curva (el "salto" que separa puntos densos de ruido).

        Args:
            min_samples: Mínimo de puntos para formar una región densa.
                Si es `None`, se calcula automáticamente.

        Returns:
            Diccionario `{"eps": float, "min_samples": int}`.
        """
        if min_samples is None:
            min_samples = max(4, 2 * self.X.shape[1])

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(self.X)
        distances, _ = neighbors.kneighbors(self.X)

        k_distances = np.sort(distances[:, -1])
        elbow_idx = _find_elbow_index(k_distances)
        eps = float(k_distances[elbow_idx])

        return {"eps": round(eps, 4), "min_samples": min_samples}

    def run_dbscan(self, eps: float, min_samples: int) -> ClusteringResult:
        """Entrena DBSCAN con los parámetros dados.

        Args:
            eps: Radio máximo de vecindad para considerar puntos como densos.
            min_samples: Mínimo de puntos para formar una región densa.

        Returns:
            `ClusteringResult`. `n_clusters` no cuenta el ruido (-1).
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X)

        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int(np.sum(labels == -1))

        # El silhouette score requiere al menos 2 clusters (sin contar ruido)
        # y no puede calcularse sobre puntos de ruido.
        sil = None
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 1:
                sil = round(silhouette_score(self.X[mask], labels[mask]), 4)

        return ClusteringResult(
            algorithm="dbscan",
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette=sil,
            params={"eps": eps, "min_samples": min_samples},
        )
