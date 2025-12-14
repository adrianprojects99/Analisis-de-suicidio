# Analisis-de-suicidio

#  Tarea y Laboratorio 4: Segmentación de Salud Mental y Suicidio Global (Clustering)

##  Descripción del Proyecto

Este repositorio contiene la solución para el Laboratorio 4 del curso de Inteligencia Artificial. El proyecto se centra en el análisis exploratorio y la aplicación de **Clustering (Aprendizaje No Supervisado)** al conjunto de datos de **Salud Mental Global, Uso de Sustancias y Suicidio** de Kaggle.

El objetivo principal es realizar una **segmentación** de las poblaciones para dividirlas en "n" clusters, con el fin de agruparlas para una **atención médica especializada** según su estado mental y riesgos asociados.



##  Dataset

* **Nombre:** Global Suicide Mental Health Substance Use Disorder
* **Fuente:** [Kaggle - thedevastator](https://www.kaggle.com/datasets/thedevastator/global-suicide-mental-health-substance-use-disor)

##  Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis y Datos:** `pandas`, `numpy`
* **Modelado ML (Clustering):** `scikit-learn`
* **Evaluación de Clusters:** `yellowbrick` / `scikit-learn`
* **Visualización:** `matplotlib`, `seaborn`
* **Interfaz de Usuario (UI):** Gradio / Streamlit (<Elegir uno>)

##  Estructura del Repositorio

* `data/`: Contiene el conjunto de datos original.
* `notebooks/`: Archivos `.ipynb` con el proceso de EDA, preprocesamiento, análisis de clustering, y la interpretación de clusters.
* `model/`: Carpeta para almacenar el modelo de clustering entrenado y serializado (e.g., `kmeans_model.pkl`).
* `app/`: Contiene el código de la interfaz de usuario web (`app.py`).
* `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

