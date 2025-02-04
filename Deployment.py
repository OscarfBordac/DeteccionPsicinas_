import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import io

# Inicialización de Roboflow y modelo
@st.cache_resource
def init_roboflow():
    rf = Roboflow(api_key="HMtpnDxMMpTmWHSXsxKV")
    return rf.workspace("ofbordac").project("piscinas-3dwyx").version(7).model

def draw_boxes(image, predictions):
    """Dibuja las cajas de predicción en la imagen"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for pred in predictions:
        # Obtener coordenadas
        x1 = int(pred['x'] - pred['width'] / 2)
        y1 = int(pred['y'] - pred['height'] / 2)
        x2 = int(x1 + pred['width'])
        y2 = int(y1 + pred['height'])

        # Dibujar rectángulo
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Añadir etiqueta con confianza
        label = f"{pred['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st.title("Detector de Piscinas")

# Cargar modelo
model = init_roboflow()

# Configuración
confidence_threshold = st.sidebar.slider(
    "Umbral de Confianza (%)",
    min_value=0,
    max_value=100,
    value=40
)

# Subir imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        # Leer imagen y convertir a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        # Mostrar imagen original
        col1, col2 = st.columns(2)
        with col1:
            st.header("Imagen Original")
            st.image(image, width=500)

        # Hacer predicción
        with st.spinner('Analizando imagen...'):
            predictions = model.predict(image_np, confidence=confidence_threshold).json()

        # Mostrar imagen con detecciones
        with col2:
            st.header("Detecciones")
            result_image = draw_boxes(image, predictions['predictions'])
            st.image(result_image, width=500)

        # Mostrar resultados detallados
        st.header("Resultados Detallados")
        st.write(f"Se encontraron {len(predictions['predictions'])} Piscinas")

        # Crear tabla de resultados
        if len(predictions['predictions']) > 0:
            results_data = []
            for i, pred in enumerate(predictions['predictions'], 1):
                results_data.append({
                    "Pileta #": i,
                    "Confianza": f"{pred['confidence']:.2f}%",
                    "Centro X": f"{pred['x']:.1f}",
                    "Centro Y": f"{pred['y']:.1f}",
                    "Ancho": f"{pred['width']:.1f}",
                    "Alto": f"{pred['height']:.1f}"
                })

            # Mostrar tabla
            st.table(results_data)

            # Mostrar coordenadas detalladas
            with st.expander("Ver coordenadas detalladas"):
                for i, pred in enumerate(predictions['predictions'], 1):
                    x1 = pred['x'] - pred['width'] / 2
                    y1 = pred['y'] - pred['height'] / 2
                    x2 = x1 + pred['width']
                    y2 = y1 + pred['height']
                    st.write(f"Pileta #{i}:")
                    st.write(f"- Esquina superior izquierda: ({x1:.1f}, {y1:.1f})")
                    st.write(f"- Esquina inferior derecha: ({x2:.1f}, {y2:.1f})")

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        st.error("Por favor, intenta con otra imagen")
