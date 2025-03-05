**#Ejecuta la Aplicación Directamente**
https://upgraded-chainsaw-q75wr7r4q4vh5pq-8501.app.github.dev/

# Food Detection App (app.py)

Este README describe exclusivamente el archivo `app.py`, que constituye el punto de entrada de la aplicación web para la detección y clasificación de alimentos. La aplicación está desarrollada en Python utilizando Streamlit y está diseñada para que el usuario cargue una imagen, la cual es procesada por un modelo de deep learning pre-entrenado para identificar alimentos y, en función de la etiqueta obtenida, mostrar información nutricional asociada.

---

## Funcionalidades Principales

- **Carga de Imagen:** Permite al usuario seleccionar y subir una imagen (formatos comunes como JPEG o PNG) desde su dispositivo.
- **Preprocesamiento:** La imagen se redimensiona y normaliza para ajustarse al tamaño de entrada requerido por el modelo.
- **Inferencia con el Modelo de IA:** Se carga un modelo de red neuronal (basado en una arquitectura similar a VGG) previamente entrenado para clasificar alimentos. El modelo procesa la imagen y retorna la etiqueta del alimento detectado.
- **Visualización de Resultados:** Una vez completada la inferencia, la aplicación muestra en la interfaz la imagen original y el resultado obtenido, junto con los datos nutricionales asociados a la clase reconocida.

---

## Requisitos

- **Python** ≥ 3.7
- **Streamlit**
- **NumPy**
- **Pillow (PIL)**
- **TensorFlow** o **PyTorch** (dependiendo de la implementación del modelo)

*(Las dependencias específicas se pueden instalar utilizando un archivo `requirements.txt` incluido en el proyecto.)*

---

## Cómo Ejecutar la Aplicación

1. **Instalación de dependencias:**  
   Asegúrate de tener Python instalado y crea un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/Mac
   venv\Scripts\activate      # En Windows

  pip install -r requirements.txt

  streamlit run app.py
  


