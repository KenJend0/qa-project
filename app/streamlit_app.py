import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

st.title("Question Answering – SQuAD")

model_name = st.selectbox(
    "Choisir le modèle",
    ["distilbert", "bert", "roberta"]
)

context = st.text_area("Contexte", height=200)
question = st.text_input("Question")

# Disable button if context or question is missing
if st.button("Obtenir la réponse", disabled=not(context and question)):
    if context and question:
        payload = {
            "context": context,
            "question": question,
            "model_name": model_name
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Réponse trouvée")
                st.write(f"**Réponse:** {result['answer']}")
                
                # Display score and latency
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score (confiance)", f"{result['score']:.3f}")
                with col2:
                    st.metric("Temps d'inférence", f"{result['latency_ms']:.1f} ms")
            else:
                error_detail = response.json().get("detail", "Erreur inconnue")
                st.error(f"❌ Erreur API: {error_detail}")
        except requests.exceptions.ConnectionError:
            st.error("❌ API non disponible. Assurez-vous que le serveur FastAPI est lancé.")
        except requests.exceptions.Timeout:
            st.error("❌ Délai d'attente dépassé. Le serveur met trop de temps à répondre.")
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")
    else:
        st.warning("⚠️ Veuillez remplir le contexte et la question")
