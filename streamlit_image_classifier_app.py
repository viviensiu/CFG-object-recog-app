#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:28:25 2023

@author: VS
"""

# =============================================================================
# Source code reference:
# https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
#     
# =============================================================================
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st  
from PIL import Image, ImageOps
import numpy as np

# Define constants
class_names= {0: 'bee', 1:'keyboard', 2:'laptop', 3:'letter_M', 4:'letter_T', 
               5:'monitor', 6:'mouse', 7:'soccer_ball', 8:'train'}
IMAGE_SIZE=224

# load model
@st.cache_resource
def get_model():
    model=load_model('CFG-image-classifier-model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model = get_model()

# create file uploader app
st.markdown("<h1 style='text-align: center; color: black;'>Object Recognition Application</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>This application attempts to predict one of these nine objects below from your image. </h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'> Objects: Bee, keyboard, laptop, letter M, letter T, monitor, mouse, soccer ball, train</h6>", unsafe_allow_html=True)

file = st.file_uploader("Try it out yourself by uploading an image!", type=["jpg", "png"])

# predict object based on uploaded image
def import_and_predict(image_data, model):
    # reformat image to be accepted by model as input
    size = (IMAGE_SIZE, IMAGE_SIZE)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    
    # predict on reformatted image
    prediction = model.predict(img_reshape, verbose=0)
    score = tf.nn.softmax(prediction[0])
    
    # for debugging purpose
    # The following will only print in the python cmd when you execute 
    # streamlit run <this app>
# =============================================================================
#     print("PREDICTION DTYPE", type(prediction))
#     print("SCORE:", score)
#     print("PREDICTION ARRAY:", prediction)
# =============================================================================
    
    output = "Predicted object: {}, with a probability of {:.2f} percent."
    output = output.format(class_names[np.argmax(score)], 100 * np.max(score))
    output = output.replace("_", " ")
    return output

if file is None:
    st.text("No image file uploaded. Please upload one.")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption = file.name, use_column_width='auto')
    predictions = import_and_predict(image, model)
    st.markdown("<h2 style='text-align: center; color: black;'>Object recognition results: </h2>", unsafe_allow_html=True)
    st.info(predictions)