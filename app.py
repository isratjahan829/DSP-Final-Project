import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('tomato.h5')
  return model
model=load_model()
st.write("""
# Tomato Leaf Disease Classification Using
 Image Recognition """
)
st.text(" CPE 027-CPE41S4 - Digital Signal Processing and Application")
st.text(" Members:")
st.text(" Aragon, Roujienald")
st.text(" Fernandez, Rhenz Nathaniel")
st.text(" Geslani, Grant Guriel")
st.text(" Instructor: Engr. Bonry Dorado")
file=st.file_uploader("Choose tomato leaf photo from your computer",type=["jpg"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Pepper__bell___Bacterial_spot',
                'Pepper__bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
    
