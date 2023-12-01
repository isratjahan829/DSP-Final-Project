import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model using st.cache_data
@st.cache_data(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('tomato.h5')
    return model

model = load_model()

st.write("""
# Tomato Leaf Disease Classification Using Image Recognition
""")

st.write("""
# Tomato Leaf Disease Classification Using Image Recognition
""")
st.text(" CPE 027-CPE41S4 - Digital Signal Processing and Application")
st.text(" Members:")
st.text(" Aragon, Roujienald")
st.text(" Fernandez, Rhenz Nathaniel")
st.text(" Geslani, Grant Guriel")
st.text(" Instructor: Engr. Bonry Dorado")

# Add file uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if file is None:
    st.text("Please upload an image file")
else:
    # Process the uploaded image
    image = Image.open(file)
    
    # Check the type of the image
    st.text(f"Image Type: {type(image)}")
    
    # Perform the resizing and preprocessing
    size = (224, 224)
    
    # Ensure the image is an instance of PIL.Image.Image
    if isinstance(image, Image.Image):
        image = ImageOps.fit(image, size, Image.ANTIALIAS)  # Use Image.ANTIALIAS directly
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]

        # Make prediction using the loaded model
        prediction = model.predict(img_reshape)

        # Define class names
        class_names = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
            'Tomato_healthy'
        ]

        # Display the predicted class
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"OUTPUT: {predicted_class}")
    else:
        st.text("Invalid image format. Please upload a valid image.")


