import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("pneumonia_densenet121.keras")

st.title("Pneumonia Detection using Deep Learning")

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error(f"Prediction: PNEUMONIA ({prediction[0][0]*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: NORMAL ({(1-prediction[0][0])*100:.2f}% confidence)")

