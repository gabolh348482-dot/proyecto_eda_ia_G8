import streamlit as st

from src.data_loader import load_and_profile
from src.eda_engine import EDAEngine, OutlierDetector
from src.ai_engine import ClusteringEngine, DimensionalityReducer
from src.insight_generator import InsightGenerator

st.set_page_config(page_title="EDA + IA", layout="wide")

for key in ["df", "overview", "eda_engine", "outlier_report",
            "kmeans_search", "clustering_result", "pca_result", "last_insights"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", [
    "1. Cargar datos", "2. EDA", "3. Clustering", "4. Insights", "5. Exportar reporte"
])

if pagina == "1. Cargar datos":
    archivo = st.file_uploader("Sube tu CSV/Excel/JSON", type=["csv", "xlsx", "json"])
    if archivo is not None:
        df, overview = load_and_profile(archivo, filename_hint=archivo.name)
        st.session_state.df = df
        st.session_state.overview = overview
        st.success(f"Cargado: {overview.n_rows} filas, {overview.n_columns} columnas")
        st.dataframe(df.head(20))
        st.write("Tipos detectados:", overview.type_counts())

elif pagina == "2. EDA":
    if st.session_state.df is None:
        st.warning("Primero carga un dataset en la pestaña 1.")
    else:
        if st.button("Correr EDA"):
            eda = EDAEngine(st.session_state.df, st.session_state.overview)
            st.session_state.eda_engine = eda
            if eda.numeric_columns:
                detector = OutlierDetector(st.session_state.df, eda.numeric_columns)
                st.session_state.outlier_report = detector.compare_methods()

        eda = st.session_state.eda_engine
        if eda is not None:
            st.subheader("Estadística numérica")
            st.dataframe(eda.numeric_summary())
            st.subheader("Correlaciones")
            st.dataframe(eda.correlation_matrix())
            st.subheader("Nulos")
            st.dataframe(eda.missing_value_summary())
            if st.session_state.outlier_report is not None:
                st.subheader("Outliers (IQR vs Isolation Forest)")
                st.dataframe(st.session_state.outlier_report.summary)

elif pagina == "3. Clustering":
    eda = st.session_state.eda_engine
    if eda is None:
        st.warning("Corre el EDA primero (pestaña 2).")
    elif len(eda.numeric_columns) < 2:
        st.error("Se requieren al menos 2 columnas numéricas para clustering.")
    else:
        if st.button("Correr Clustering + PCA"):
            engine = ClusteringEngine(st.session_state.df, eda.numeric_columns)
            search = engine.find_optimal_k()
            result = engine.run_kmeans(search.suggested_k)
            pca = DimensionalityReducer().fit_transform(st.session_state.df, eda.numeric_columns)
            st.session_state.kmeans_search = search
            st.session_state.clustering_result = result
            st.session_state.pca_result = pca

        if st.session_state.clustering_result is not None:
            st.write("k sugerido:", st.session_state.kmeans_search.suggested_k)
            st.dataframe(st.session_state.kmeans_search.summary)
            st.write("Varianza explicada por PCA:",
                     st.session_state.pca_result.total_variance_explained, "%")
            plot_df = st.session_state.pca_result.components_df.copy()
            plot_df["cluster"] = st.session_state.clustering_result.labels
            st.scatter_chart(plot_df, x="PC1", y="PC2", color="cluster")

elif pagina == "4. Insights":
    if st.session_state.eda_engine is None:
        st.warning("Corre el EDA primero.")
    else:
        gen = InsightGenerator(
            eda_engine=st.session_state.eda_engine,
            outlier_report=st.session_state.outlier_report,
            kmeans_search=st.session_state.kmeans_search,
            clustering_result=st.session_state.clustering_result,
            pca_result=st.session_state.pca_result,
        )
        insights = gen.generate_all()
        st.session_state.last_insights = insights
        for ins in insights:
            st.markdown(f"- **[{ins['severity']}]** {ins['text']}")

elif pagina == "5. Exportar reporte":
    if not st.session_state.last_insights:
        st.warning("Genera los insights primero (pestaña 4).")
    else:
        from src.report_exporter import export_html
        if st.button("Generar reporte HTML"):
            path = export_html(st.session_state.last_insights)
            st.success(f"Reporte generado: {path}")
            with open(path, "rb") as f:
                st.download_button("Descargar HTML", f, file_name="reporte.html")