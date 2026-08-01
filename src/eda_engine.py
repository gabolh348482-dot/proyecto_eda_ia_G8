"""
eda_engine.py
=============

Módulo 2: Motor de EDA Automatizado & Detección de Outliers.

Este módulo consume la salida del Módulo 1 (`DatasetOverview`) para no
tener que re-detectar tipos de columna, y expone dos componentes:

    - `EDAEngine`: estadística descriptiva automatizada (numérica,
      categórica, correlaciones, análisis de nulos).
    - `OutlierDetector`: detección de outliers combinando un método
      estadístico univariado (IQR) con un método de Machine Learning
      multivariado (Isolation Forest), y una comparación entre ambos.

No depende de Streamlit ni de ningún framework de UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from src.data_loader import ColumnType, DatasetOverview


# --------------------------------------------------------------------------- #
# Motor de EDA (estadística descriptiva)
# --------------------------------------------------------------------------- #
class EDAEngine:
    """Genera estadística descriptiva automatizada a partir de un dataset
    ya perfilado por el Módulo 1.

    Args:
        df: DataFrame original (salida de `load_dataset`).
        overview: `DatasetOverview` generado por `ColumnTypeDetector`.
    """

    def __init__(self, df: pd.DataFrame, overview: DatasetOverview) -> None:
        self.df = df
        self.overview = overview
        self.numeric_columns = self._columns_of_types(
            [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]
        )
        self.categorical_columns = self._columns_of_types(
            [ColumnType.CATEGORICAL, ColumnType.BOOLEAN]
        )

    def _columns_of_types(self, types: list[ColumnType]) -> list[str]:
        """Une los nombres de columna que pertenecen a alguno de los tipos dados."""
        columns: list[str] = []
        for column_type in types:
            columns.extend(self.overview.columns_by_type(column_type))
        return columns

    def numeric_summary(self) -> pd.DataFrame:
        """Calcula estadísticas descriptivas para todas las columnas numéricas.

        Incluye, además de lo estándar (`describe`), asimetría y curtosis,
        útiles para decidir si conviene tratar los outliers con métodos
        robustos (IQR) o paramétricos (Z-score).

        Returns:
            DataFrame indexado por columna, con una fila de estadísticas
            por cada variable numérica. Devuelve un DataFrame vacío si no
            hay columnas numéricas.
        """
        if not self.numeric_columns:
            return pd.DataFrame()

        rows = []
        for col in self.numeric_columns:
            series = self.df[col].dropna()
            rows.append(
                {
                    "columna": col,
                    "count": int(series.count()),
                    "media": series.mean(),
                    "mediana": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "q1": series.quantile(0.25),
                    "q3": series.quantile(0.75),
                    "max": series.max(),
                    "asimetria": stats.skew(series) if len(series) > 2 else np.nan,
                    "curtosis": stats.kurtosis(series) if len(series) > 2 else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("columna").round(3)

    def categorical_summary(self, top_n: int = 10) -> dict[str, pd.DataFrame]:
        """Genera tablas de frecuencia para columnas categóricas/booleanas.

        Args:
            top_n: Número máximo de categorías a mostrar individualmente;
                el resto se agrupa bajo la etiqueta "Otros".

        Returns:
            Diccionario `{nombre_columna: tabla_de_frecuencias}`, donde cada
            tabla tiene columnas `categoria`, `frecuencia` y `porcentaje`.
        """
        summaries: dict[str, pd.DataFrame] = {}
        for col in self.categorical_columns:
            counts = self.df[col].value_counts(dropna=True)
            if len(counts) > top_n:
                top = counts.iloc[:top_n]
                otros = pd.Series({"Otros": counts.iloc[top_n:].sum()})
                counts = pd.concat([top, otros])

            table = counts.rename("frecuencia").reset_index()
            table.columns = ["categoria", "frecuencia"]
            table["porcentaje"] = (table["frecuencia"] / len(self.df) * 100).round(2)
            summaries[col] = table
        return summaries

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Calcula la matriz de correlación entre variables numéricas.

        Args:
            method: `"pearson"` (relaciones lineales) o `"spearman"`
                (relaciones monótonas, más robusto a outliers).

        Returns:
            DataFrame cuadrado con la matriz de correlación. Vacío si hay
            menos de 2 columnas numéricas.
        """
        if len(self.numeric_columns) < 2:
            return pd.DataFrame()
        return self.df[self.numeric_columns].corr(method=method).round(3)

    def missing_value_summary(self) -> pd.DataFrame:
        """Genera un ranking de columnas por porcentaje de valores nulos.

        Returns:
            DataFrame ordenado descendentemente por `porcentaje_nulos`,
            incluyendo solo columnas que tienen al menos un nulo.
        """
        rows = [
            {
                "columna": name,
                "n_nulos": profile.n_missing,
                "porcentaje_nulos": round(profile.missing_ratio * 100, 2),
                "tipo_detectado": profile.detected_type.value,
            }
            for name, profile in self.overview.columns.items()
            if profile.n_missing > 0
        ]
        if not rows:
            return pd.DataFrame(columns=["columna", "n_nulos", "porcentaje_nulos", "tipo_detectado"])
        return pd.DataFrame(rows).sort_values("porcentaje_nulos", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Detección de outliers
# --------------------------------------------------------------------------- #
@dataclass
class OutlierReport:
    """Resultado consolidado de la detección de outliers.

    Attributes:
        iqr_flags: DataFrame booleano (misma forma que las columnas numéricas)
            donde `True` indica que ese valor es outlier según IQR.
        isolation_forest_flags: Serie booleana a nivel de fila; `True` indica
            que la fila completa fue marcada como anomalía multivariada.
        summary: Tabla comparativa de conteos/porcentajes por método.
    """

    iqr_flags: pd.DataFrame
    isolation_forest_flags: pd.Series
    summary: pd.DataFrame

    def rows_flagged_by_both(self) -> pd.Index:
        """Índices de filas marcadas como outlier por ambos métodos a la vez.

        Una fila cuenta como "marcada por IQR" si al menos una de sus
        columnas numéricas fue detectada como outlier univariado.
        """
        iqr_any_row = self.iqr_flags.any(axis=1)
        both = iqr_any_row & self.isolation_forest_flags
        return self.isolation_forest_flags[both].index


class OutlierDetector:
    """Detecta outliers combinando un método estadístico y uno de ML.

    Args:
        df: DataFrame original.
        numeric_columns: Columnas numéricas a evaluar (típicamente
            `EDAEngine.numeric_columns`). Se recomienda excluir
            `numeric_discrete` de baja cardinalidad si se desea un análisis
            puramente continuo, pero por defecto se aceptan todas.
    """

    def __init__(self, df: pd.DataFrame, numeric_columns: list[str]) -> None:
        if not numeric_columns:
            raise ValueError("Se requiere al menos una columna numérica para detectar outliers.")
        self.df = df
        self.numeric_columns = numeric_columns

    def detect_iqr(self, factor: float = 1.5) -> pd.DataFrame:
        """Detecta outliers univariados usando el rango intercuartílico (IQR).

        Un valor se marca como outlier si cae fuera de
        `[Q1 - factor*IQR, Q3 + factor*IQR]`, evaluado columna por columna
        de forma independiente.

        Args:
            factor: Multiplicador del IQR. El estándar es 1.5; usar 3.0
                para un criterio más conservador ("outliers extremos").

        Returns:
            DataFrame booleano con la misma forma que `df[numeric_columns]`.
        """
        flags = pd.DataFrame(index=self.df.index)
        for col in self.numeric_columns:
            series = self.df[col]
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            flags[col] = (series < lower) | (series > upper)
        return flags.fillna(False)

    def detect_isolation_forest(
        self,
        contamination: float | str = "auto",
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> pd.Series:
        """Detecta anomalías multivariadas con Isolation Forest.

        A diferencia del IQR (que evalúa cada columna por separado),
        Isolation Forest considera la combinación de todas las variables
        numéricas simultáneamente: un punto puede ser normal en cada
        variable individual pero anómalo en su combinación.

        Args:
            contamination: Proporción esperada de outliers en los datos.
                `"auto"` deja que el algoritmo lo estime.
            random_state: Semilla para reproducibilidad.
            n_estimators: Número de árboles del ensamble.

        Returns:
            Serie booleana indexada igual que `df`; `True` = outlier.
        """
        # Isolation Forest no acepta NaN: se imputa con la mediana en una
        # copia auxiliar, sin modificar el DataFrame original del usuario.
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(self.df[self.numeric_columns])

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        predictions = model.fit_predict(X)  # -1 = outlier, 1 = normal
        return pd.Series(predictions == -1, index=self.df.index, name="is_outlier_if")

    def compare_methods(self, iqr_factor: float = 1.5) -> OutlierReport:
        """Ejecuta ambos métodos de detección y arma un reporte comparativo.

        Args:
            iqr_factor: Multiplicador del IQR (ver `detect_iqr`).

        Returns:
            `OutlierReport` con las máscaras de cada método y una tabla
            resumen con conteos y porcentajes.
        """
        iqr_flags = self.detect_iqr(factor=iqr_factor)
        if_flags = self.detect_isolation_forest()

        iqr_any_row = iqr_flags.any(axis=1)
        n_total = len(self.df)

        summary = pd.DataFrame(
            [
                {
                    "metodo": "IQR (univariado)",
                    "n_outliers": int(iqr_any_row.sum()),
                    "porcentaje": round(iqr_any_row.sum() / n_total * 100, 2),
                },
                {
                    "metodo": "Isolation Forest (multivariado)",
                    "n_outliers": int(if_flags.sum()),
                    "porcentaje": round(if_flags.sum() / n_total * 100, 2),
                },
                {
                    "metodo": "Coincidencia entre ambos métodos",
                    "n_outliers": int((iqr_any_row & if_flags).sum()),
                    "porcentaje": round((iqr_any_row & if_flags).sum() / n_total * 100, 2),
                },
            ]
        )

        return OutlierReport(iqr_flags=iqr_flags, isolation_forest_flags=if_flags, summary=summary)
