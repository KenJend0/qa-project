import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Question Answering – SQuAD")

model_name = st.selectbox(
    "Choisir le modèle",
    ["distilbert", "bert", "roberta"]
)

context = st.text_area("Contexte", height=200)
question = st.text_input("Question")

if st.button("Obtenir la réponse"):
    if context and question:
        payload = {
            "context": context,
            "question": question,
            "model_name": model_name
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            st.success("Réponse")
            st.write(response.json()["answer"])
        else:
            st.error("Erreur lors de la prédiction")
    else:
        st.warning("Veuillez remplir le contexte et la question")
