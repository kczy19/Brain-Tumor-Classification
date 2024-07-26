import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import streamlit as st

model = load_model('BrainTumor10EpochsCategorical.h5')
st.sidebar.success('Model loaded successfully')

def get_class_name(class_no):
    return "No Brain Tumor" if class_no == 0 else "Yes Brain Tumor"

def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_index = np.argmax(result, axis=1)
    return class_index

# Streamlit app
st.title('🧠 Brain Tumor Detection')
st.markdown("""
    <style>
    .main {
        background-color: #424549;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .css-1d391kg {
        padding-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

st.header('Upload an MRI Image')

uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    file_path = os.path.join('uploads', uploaded_file.name)
    os.makedirs('uploads', exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded MRI Image.', use_column_width=True, channels="RGB")
    st.write("") 
    st.markdown("<h3 style='text-align: center;'>Classifying...</h3>", unsafe_allow_html=True)

    value = get_result(file_path)
    result = get_class_name(value[0])
    
    st.markdown(f"<h2 style='text-align: center;'>Prediction: {result}</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h4 style='text-align: center; color: grey;'>Please upload an MRI image to start the classification.</h4>", unsafe_allow_html=True)
