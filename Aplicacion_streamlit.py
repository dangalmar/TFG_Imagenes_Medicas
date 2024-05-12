import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./DenseNet-201_trained_with_data_augmentation_and_adam.h5')

def preprocess_image(image):
    img = np.array(image) / 255.0  # Normalizar los valores de píxeles
    img = np.expand_dims(img, axis=0)  # Añadir una dimensión para batch
    return img

# Configurar la interfaz de la aplicación con estilos mejorados
st.title('Clasificador de Cáncer de Piel')
st.markdown("---")

# Permitir al usuario cargar una imagen con un botón y estilos mejorados
st.write('Por favor, sube una imagen para clasificar si es maligna o benigna')
uploaded_image = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'], 
                                   help="Sube una imagen con un formato válido (jpg, jpeg, png)")

# Mostrar la imagen cargada con estilos mejorados
if uploaded_image is not None:
    st.markdown("---")
    st.subheader('Imagen cargada:')
    image = Image.open(uploaded_image)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Preprocesar la imagen
    img_array = preprocess_image(image)

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar el resultado de la predicción con estilos mejorados
    st.markdown("---")
    st.subheader('Resultado de la clasificación:')
    porcentageMalignant = prediction[0][0] * 100
    porcentageBenign = (1 - prediction[0][0]) * 100
    if prediction[0][0] > 0.5:
        st.error('La imagen muestra un tumor maligno con una probabilidad del ' + str(porcentageMalignant) + "%")
    else:
        st.success('La imagen muestra un tumor benigno con una probabilidad del ' + str(porcentageBenign) + "%")
