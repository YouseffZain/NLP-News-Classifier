import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

st.title("News Category Classifier")

model = load_model("news_model.h5")
vec = pickle.load(open("vectorizer.pkl","rb"))
labels = pickle.load(open("labels.pkl","rb"))

txt = st.text_input("Enter headline:")

if st.button("Predict"):
    if txt.strip():
        x = vec.transform([txt]).toarray()
        pred = model.predict(x)[0]
        st.success(labels[np.argmax(pred)])
