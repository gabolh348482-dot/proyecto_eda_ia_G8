"""
data_loader.py
===============

Módulo 1: Carga Dinámica y Detección de Tipos de Variable.

Responsabilidades:
    1. Cargar datasets desde distintos formatos (CSV, Excel, JSON) de forma
       robusta, aceptando tanto rutas de archivo como buffers en memoria
       (compatible con `st.file_uploader` de Streamlit).
    2. Detectar automáticamente el tipo semántico de cada columna
       (numérica continua/discreta, categórica, texto libre, datetime,
       booleana, identificador), más allá del dtype crudo de pandas.
    3. Generar un resumen general (profiling inicial) del dataset.

Este módulo no depende de Streamlit ni de ningún framework de UI,
por lo que puede probarse e importarse de forma completamente aislada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np
import pandas as pd

# Tipo genérico para admitir tanto rutas como buffers (UploadedFile de Streamlit)
FileSource = Union[str, Path, BinaryIO]


# --------------------------------------------------------------------------- #
# Excepciones específicas del módulo
# --------------------------------------------------------------------------- #
class UnsupportedFileFormatError(Exception):
    """Se lanza cuando la extensión del archivo no está soportada."""


class EmptyDatasetError(Exception):
    """Se lanza cuando el dataset cargado no contiene filas."""


# --------------------------------------------------------------------------- #
# Enum de tipos semánticos de columna
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    """Categorías semánticas que puede tomar una columna del dataset."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"  # una sola categoría/valor en toda la columna


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
def load_dataset(source: FileSource, filename_hint: str | None = None) -> pd.DataFrame:
    """Carga un dataset desde un archivo o buffer, detectando su formato.

    Soporta CSV (con detección de separador), Excel (.xlsx, .xls) y JSON.
    Está diseñado para aceptar tanto una ruta de sistema de archivos como
    un objeto tipo archivo en memoria (por ejemplo, el retornado por
    `st.file_uploader`), en cuyo caso `filename_hint` es obligatorio para
    poder inferir la extensión.

    Args:
        source: Ruta al archivo o buffer binario/de texto con el contenido.
        filename_hint: Nombre de archivo (con extensión) usado para inferir
            el formato cuando `source` es un buffer sin atributo `.name`.

    Returns:
        DataFrame de pandas con los datos cargados.

    Raises:
        UnsupportedFileFormatError: Si la extensión no es reconocida.
        EmptyDatasetError: Si el archivo se lee correctamente pero no
            contiene filas de datos.

    Example:
        >>> df = load_dataset("ventas.csv")
        >>> df = load_dataset(uploaded_file, filename_hint=uploaded_file.name)
    """
    extension = _resolve_extension(source, filename_hint)

    if extension == "csv":
        df = _read_csv_robust(source)
    elif extension in {"xlsx", "xls"}:
        df = pd.read_excel(source)
    elif extension == "json":
        df = pd.read_json(source)
    else:
        raise UnsupportedFileFormatError(
            f"Formato '.{extension}' no soportado. Usa CSV, Excel o JSON."
        )

    if df.empty:
        raise EmptyDatasetError("El archivo se cargó pero no contiene filas de datos.")

    # Normalización básica de nombres de columnas (espacios sobrantes)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _resolve_extension(source: FileSource, filename_hint: str | None) -> str:
    """Determina la extensión del archivo a partir del path o del hint."""
    name = filename_hint
    if name is None:
        # Objetos tipo UploadedFile de Streamlit exponen `.name`
        name = getattr(source, "name", None) or str(source)
    return Path(name).suffix.lower().lstrip(".")


def _read_csv_robust(source: FileSource) -> pd.DataFrame:
    """Lee un CSV probando separadores comunes si el default falla en inferirlos.

    pandas puede inferir el separador con `sep=None` y `engine="python"`,
    pero esto falla en algunos archivos con codificaciones no estándar,
    por lo que se agrega un fallback explícito.
    """
    try:
        return pd.read_csv(source, sep=None, engine="python")
    except Exception:
        # Si `source` es un buffer ya consumido, hay que rebobinarlo
        if hasattr(source, "seek"):
            source.seek(0)
        for sep in (",", ";", "\t", "|"):
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                return pd.read_csv(source, sep=sep, encoding="utf-8", engine="python")
            except Exception:
                continue
        raise


# --------------------------------------------------------------------------- #
# Detección automática de tipos de columna
# --------------------------------------------------------------------------- #
@dataclass
class ColumnProfile:
    """Metadatos de una columna individual tras el análisis de tipo."""

    name: str
    detected_type: ColumnType
    dtype_original: str
    n_unique: int
    unique_ratio: float
    n_missing: int
    missing_ratio: float


@dataclass
class DatasetOverview:
    """Resumen general del dataset (profiling inicial)."""

    n_rows: int
    n_columns: int
    memory_usage_mb: float
    n_duplicated_rows: int
    total_missing_ratio: float
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def type_counts(self) -> dict[str, int]:
        """Cuenta cuántas columnas hay por cada tipo detectado."""
        counts: dict[str, int] = {}
        for profile in self.columns.values():
            key = profile.detected_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def columns_by_type(self, column_type: ColumnType) -> list[str]:
        """Devuelve los nombres de columna que pertenecen a un tipo dado."""
        return [
            name
            for name, profile in self.columns.items()
            if profile.detected_type == column_type
        ]


class ColumnTypeDetector:
    """Detecta el tipo semántico de cada columna de un DataFrame.

    La detección usa una heurística en cascada porque el dtype crudo de
    pandas no es suficiente para reconocer, por ejemplo, fechas guardadas
    como texto o columnas categóricas codificadas como enteros.

    Args:
        categorical_cardinality_threshold: Proporción máxima de valores
            únicos respecto al total de filas para considerar una columna
            de texto/entero como categórica en lugar de identificador/texto.
        discrete_unique_threshold: Número máximo de valores únicos para que
            una columna numérica se considere discreta en vez de continua.
        datetime_parse_success_threshold: Proporción mínima de valores que
            deben parsearse correctamente como fecha para aceptar la columna
            como datetime.
    """

    def __init__(
        self,
        categorical_cardinality_threshold: float = 0.05,
        discrete_unique_threshold: int = 15,
        datetime_parse_success_threshold: float = 0.9,
    ) -> None:
        self.categorical_cardinality_threshold = categorical_cardinality_threshold
        self.discrete_unique_threshold = discrete_unique_threshold
        self.datetime_parse_success_threshold = datetime_parse_success_threshold

    def profile_dataset(self, df: pd.DataFrame) -> DatasetOverview:
        """Genera el perfil completo (overview + por columna) del dataset.

        Args:
            df: DataFrame ya cargado (ver `load_dataset`).

        Returns:
            `DatasetOverview` con el resumen general y el detalle por columna.
        """
        column_profiles = {
            col: self._profile_column(df[col], total_rows=len(df)) for col in df.columns
        }

        overview = DatasetOverview(
            n_rows=len(df),
            n_columns=len(df.columns),
            memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024**2), 3),
            n_duplicated_rows=int(df.duplicated().sum()),
            total_missing_ratio=round(float(df.isna().mean().mean()), 4),
            columns=column_profiles,
        )
        return overview

    def _profile_column(self, series: pd.Series, total_rows: int) -> ColumnProfile:
        """Clasifica una columna individual y calcula sus estadísticas base."""
        n_missing = int(series.isna().sum())
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        unique_ratio = n_unique / total_rows if total_rows > 0 else 0.0

        detected_type = self._detect_type(series, non_null, n_unique, unique_ratio)

        return ColumnProfile(
            name=str(series.name),
            detected_type=detected_type,
            dtype_original=str(series.dtype),
            n_unique=n_unique,
            unique_ratio=round(unique_ratio, 4),
            n_missing=n_missing,
            missing_ratio=round(n_missing / total_rows, 4) if total_rows > 0 else 0.0,
        )

    def _detect_type(
        self,
        series: pd.Series,
        non_null: pd.Series,
        n_unique: int,
        unique_ratio: float,
    ) -> ColumnType:
        """Heurística en cascada para determinar el tipo semántico."""
        if n_unique == 0:
            return ColumnType.CONSTANT
        if n_unique == 1:
            return ColumnType.CONSTANT

        # 1. Booleanas explícitas o codificadas (2 valores únicos "tipo flag")
        if self._is_boolean_like(non_null, n_unique):
            return ColumnType.BOOLEAN

        is_string_like = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        )

        # 2. Datetime nativo o parseable desde texto
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        if is_string_like and self._is_parseable_as_datetime(non_null):
            return ColumnType.DATETIME

        # 3. Numéricas (continua vs. discreta)
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= self.discrete_unique_threshold:
                return ColumnType.NUMERIC_DISCRETE
            return ColumnType.NUMERIC_CONTINUOUS

        # 4. String-like: identificador, categórica o texto libre.
        # Antes de mirar la cardinalidad, se revisa si el contenido "parece"
        # texto natural (varias palabras) en vez de un código/ID de un token,
        # porque un ID (alta cardinalidad, sin espacios) y una descripción
        # libre (alta cardinalidad, con espacios) requieren tratamiento distinto.
        if is_string_like and self._looks_like_free_text(non_null):
            return ColumnType.TEXT
        if unique_ratio >= 0.95:
            return ColumnType.IDENTIFIER
        if unique_ratio <= self.categorical_cardinality_threshold or n_unique <= 50:
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT

    @staticmethod
    def _is_boolean_like(non_null: pd.Series, n_unique: int) -> bool:
        """Detecta booleanos reales o codificados (0/1, Sí/No, True/False...)."""
        if n_unique != 2:
            return False
        if pd.api.types.is_bool_dtype(non_null):
            return True
        normalized = {str(v).strip().lower() for v in non_null.unique()}
        boolean_pairs = [
            {"true", "false"},
            {"0", "1"},
            {"si", "no"},
            {"sí", "no"},
            {"yes", "no"},
            {"y", "n"},
        ]
        return normalized in boolean_pairs

    @staticmethod
    def _looks_like_free_text(
        non_null: pd.Series, sample_size: int = 100, avg_word_count_threshold: float = 3.0
    ) -> bool:
        """Determina si una columna de texto contiene oraciones/descripciones
        en lugar de códigos cortos (IDs) o etiquetas categóricas.

        Se basa en el número promedio de palabras por valor: un ID como
        "CUST-00042" o una categoría como "Electrónica" tienen 1 palabra,
        mientras que una descripción libre normalmente tiene varias.
        """
        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        word_counts = sample.astype(str).str.split().str.len()
        return bool(word_counts.mean() >= avg_word_count_threshold)

    @staticmethod
    def _is_parseable_as_datetime(non_null: pd.Series, sample_size: int = 200) -> bool:
        """Intenta parsear una muestra de la columna como fecha.

        Se evita parsear enteros/floats puros como fecha (pandas a veces los
        interpreta como timestamps Unix, generando falsos positivos).
        """
        if pd.api.types.is_numeric_dtype(non_null):
            return False

        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(sample, errors="coerce")

        success_ratio = parsed.notna().mean()
        return bool(success_ratio >= 0.9)


# --------------------------------------------------------------------------- #
# Función de conveniencia (fachada simple para uso rápido / Streamlit)
# --------------------------------------------------------------------------- #
def load_and_profile(
    source: FileSource, filename_hint: str | None = None
) -> tuple[pd.DataFrame, DatasetOverview]:
    """Carga un dataset y genera su perfil en un solo paso.

    Args:
        source: Ruta o buffer del archivo.
        filename_hint: Nombre con extensión, requerido si `source` es un buffer.

    Returns:
        Tupla `(df, overview)` lista para pasar al Módulo 2 (EDA Engine).
    """
    df = load_dataset(source, filename_hint=filename_hint)
    overview = ColumnTypeDetector().profile_dataset(df)
    return df, overview
