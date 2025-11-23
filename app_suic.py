import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Carga del Modelo (Al inicio del script) ---
# Intentamos cargar el modelo K-Means y el Escalador serializado.
try:
    # Cargar el archivo .joblib que contiene el modelo K-Means y el Escalador
    model_data = joblib.load('clustering_model.joblib')
    kmeans = model_data['kmeans_model']
    scaler = model_data['scaler']
    features = model_data['features'] # Lista con el orden correcto de las features de entrenamiento
    MODEL_LOADED = True
except FileNotFoundError:
    kmeans, scaler, features = None, None, None
    MODEL_LOADED = False
    #st.error("Archivo 'clustering_model.joblib' no encontrado.")
except Exception as e:
    # Captura cualquier otro error
    st.error(f"Error al cargar el modelo: {e}")
    kmeans, scaler, features = None, None, None
    MODEL_LOADED = False

# --- Lógica de Análisis por Cluster (Basada en Paso 10) ---
# Definición de las propiedades y la interpretación para cada cluster.
CLUSTER_PROPERTIES = {
    0: {
        'nombre': "Cluster 0: Riesgo de Salud Mental Crítico",
        'prioridad': "ATENCIÓN INMEDIATA EN SALUD MENTAL",
        'color': 'red',
        'comportamiento': "Este grupo presenta el **riesgo de suicidio más alto** (especialmente en hombres). Necesita estrategias de intervención de crisis y apoyo psicológico inmediato."
    },
    1: {
        'nombre': "Cluster 1: Bajo Riesgo General",
        'prioridad': "PREVENCIÓN GENERAL Y MANTENIMIENTO",
        'color': 'green',
        'comportamiento': "Este es el grupo con el **menor riesgo** en todas las métricas. El foco debe ser mantener las políticas de salud preventivas y monitorear el bienestar general."
    },
    2: {
        'nombre': "Cluster 2: Riesgo de Mortalidad por ECNT Elevado",
        'prioridad': "ATENCIÓN EN ENFERMEDADES CRÓNICAS (ECNT)",
        'color': 'orange',
        'comportamiento': "El riesgo de suicidio es bajo, pero la **Probabilidad de Muerte por ECNT es la más alta**. Se requiere atención prioritaria en programas de prevención de enfermedades cardiovasculares, cáncer y diabetes."
    }
}

def analyze_cluster(cluster_id):
    return CLUSTER_PROPERTIES.get(cluster_id)

# --- Interfaz Streamlit Principal ---
st.set_page_config(page_title="Modelo de Segmentación de Riesgo de Salud", layout="wide")

st.title("🧠 Herramienta de Segmentación de Riesgo de Salud")
st.markdown("---")
st.markdown("Esta aplicación predice el **Cluster de Atención** de un país (o región) basándose en sus tasas de mortalidad y suicidio, utilizando el modelo K-Means.")

if not MODEL_LOADED:
    st.error("🚨 **ERROR:** El modelo no se pudo cargar. Asegúrese de que el archivo `clustering_model.joblib` existe en el mismo directorio.")
elif MODEL_LOADED:
    st.header("1. Ingreso de Tasas del País/Región")
    st.markdown("Ingrese las 4 tasas clave (valores crudos) para clasificar el tipo de riesgo del país:")

    # Crear la interfaz de entrada para las 4 features
    col1, col2 = st.columns(2)
    
    with col1:
        # Probabilidad de Muerte por ECNT en Hombres
        prob_h = st.number_input("Prob. Muerte ECNT (Hombres) [%]", min_value=0.0, max_value=100.0, value=35.0, step=0.1, help="Probabilidad de morir entre 30 y 70 años por ECNT en hombres.")
        # Tasa de Suicidio Cruda en Hombres
        suicidio_h = st.number_input("Tasa Suicidio Cruda (Hombres) [por 100k]", min_value=0.0, max_value=100.0, value=10.0, step=0.1, help="Tasa de suicidio cruda en hombres.")
    
    with col2:
        # Probabilidad de Muerte por ECNT en Mujeres
        prob_m = st.number_input("Prob. Muerte ECNT (Mujeres) [%]", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Probabilidad de morir entre 30 y 70 años por ECNT en mujeres.")
        # Tasa de Suicidio Cruda en Mujeres
        suicidio_m = st.number_input("Tasa Suicidio Cruda (Mujeres) [por 100k]", min_value=0.0, max_value=100.0, value=3.0, step=0.1, help="Tasa de suicidio cruda en mujeres.")
        
    st.markdown("---")
    
    # --- Predicción y Resultado ---
    if st.button("Clasificar Cluster de Atención", type="primary"):
        
        # 1. Construcción explícita del DataFrame de entrada (Solución CRÍTICA a la desalineación)
        input_data = pd.DataFrame({
            'Prob_Muerte_ECNT_Hombres': [prob_h],
            'Prob_Muerte_ECNT_Mujeres': [prob_m],
            'Tasa_Suicidio_Cruda_Hombres': [suicidio_h],
            'Tasa_Suicidio_Cruda_Mujeres': [suicidio_m]
        })
        
        # 2. Reordenar las columnas para que coincidan EXACTAMENTE con el orden de entrenamiento
        input_data = input_data[features] 

        # 3. Escalar los datos de entrada
        input_scaled = scaler.transform(input_data)
        
        # 4. Predecir el Cluster
        cluster_pred = kmeans.predict(input_scaled)[0]
        
        # 5. Obtener el análisis
        analysis = analyze_cluster(cluster_pred)

        st.success("✅ Análisis Generado")
        st.header(f"2. Resultado de la Segmentación")
        
        # Mostrar el resultado con estilos
        st.markdown(f"**Cluster Asignado:** <span style='font-size: 1.5em;'>**{cluster_pred}**</span>", unsafe_allow_html=True)
        st.title(analysis['nombre'])
        
        st.markdown(f"**Prioridad de Atención:** <span style='color: {analysis['color']}; font-size: 1.5em;'>**{analysis['prioridad']}**</span>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("Análisis de Comportamiento y Recomendaciones")
        st.markdown(analysis['comportamiento'])
        
        st.info("\n\n*El modelo de K-Means agrupa a este país con otros que tienen un perfil de riesgo estadístico similar.*")