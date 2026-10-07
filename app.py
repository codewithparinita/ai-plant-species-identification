
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Plant species names
class_names = [
    "Dandelion",
    "Daisy",
    "Tulips",
    "Sunflowers",
    "Roses"
]

# Load the saved AI model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_species_model.h5")

model = load_model()

# Application title
st.title("🌱 AI-Based Plant Species Identification")

st.write(
    "Upload a flower image to identify its species "
    "using Artificial Intelligence."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Plant Image",
             use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict species
    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index]) * 100

    st.success(
        f"🌿 Predicted Species: {class_names[predicted_index]}"
    )

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.subheader("Top Predictions")

    top_indices = np.argsort(prediction)[::-1][:5]

    for index in top_indices:
        st.write(
            f"**{class_names[index]}:** "
            f"{prediction[index] * 100:.2f}%"
        )

st.caption(
    "This model identifies five flower categories. "
    "Predictions may be incorrect for unfamiliar images."
)
