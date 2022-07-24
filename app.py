import streamlit as st
import keras
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np

st.title("Binary image Classification")
st.header("Chakka Manga Classification")
st.text("Upload an image for image classification as Jackfruit or mango")

dic = {0 : 'red', 1 : 'green'}

def teachable_machine_classification(img,  weights_file):
    model = keras.models.load_model(weights_file)
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = np.asarray(img) 
    img = img / 255 
    img = img.reshape(1, 224,224,3)
    p = np.argmax(model.predict(img), axis=1)
    return p 

uploaded_file = st.file_uploader("Choose an image ...", type=[ 'png', "jpg","jpeg"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        p = teachable_machine_classification(image, 'keras_model.h5')
        prediction = dic[p[0]]
        if prediction == 'red:
            st.markdown('This is likely to be a Red Algae')
        else if prediction == 'green:
            st.markdown('This is more likely to be a Green Algae.')
