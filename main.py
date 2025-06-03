import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model = tf.keras.models.load_model('my_model.keras')

st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload an image of a digit (28x28 grayscale)")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert (black on white)
    image = image.resize((28, 28))
    st.image(image, caption='Input Image', width=150)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"**Prediction:** {predicted_class}")

