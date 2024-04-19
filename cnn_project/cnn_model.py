import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
image = Image.open("thumbnail-bb4636136d3415e1fe16927d44e7d648-1140x618.jpeg")
st.image(image,width=800)
model = load_model('model2.h5')
def preprocess_image(image):
    img = imread(image)
    img_resized = resize(img, (150, 150, 1))
    return np.expand_dims(img_resized, axis=0)
st.title("TRAFFIC LIGHT :red[ CLASSIFIER]")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file,caption='Uploaded Image.', use_column_width=True)
    with st.spinner('Making predictions...'):
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
    st.write('The traffic light displayed above is:')
    class_names = ['back', 'green', 'red', 'yellow']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(predicted_class)
