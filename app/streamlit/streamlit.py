import streamlit as st
import requests

API = "http://localhost:8000/translate"  # local first

st.title("English → Japanese Translation (Custom Transformer)")

text = st.text_area("Enter text")

if st.button("Translate"):
    if text.strip():
        res = requests.post(API, json={"text": text})
        st.write("### Translation:")
        st.success(res.json()["translation"])
