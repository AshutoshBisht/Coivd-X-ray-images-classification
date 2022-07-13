import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import os
import streamlit as st

m1 = load_model('model_covid.h5')
tf.keras.models.save_model(m1,'my_model.hdf5')
model = tf.keras.models.load_model('my_model.hdf5')


st.write("""
         # Covid-19 prediction using CXR(Chest X-Ray) images
         """
         )
st.write("This is a simple image classification web app to predict whether is person is nomral of COVID-19 infected using CXR(Chest X-Ray images")


    
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
 size = (150,150)    
 image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
 image = np.asarray(image)
 img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
 img_reshape = img_resize[np.newaxis,...]
    
 prediction = model.predict(img_reshape)
 return prediction
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if( prediction.round() == 1):
     st.write("Prediction: Normal")
    else:
     st.write("Prediction: Covid Positive!!")

