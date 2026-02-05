# Question Answering – Fine-tuning SQuAD

Projet DataScale – QA extractif avec Transformers.

## Objectifs
- Fine-tuner 3 modèles NLP sur SQuAD
- Comparer précision et temps d’inférence
- Déployer une application Streamlit (HF Spaces)

## Stack
- PyTorch, Hugging Face Transformers
- FastAPI + Streamlit

## 🚀 Interface utilisateur

### Backend FastAPI
```bash
uvicorn app.api:app --reload
```
API disponible sur http://127.0.0.1:8000

### Frontend Streamlit
```bash
streamlit run app/streamlit_app.py
```
Interface disponible sur http://localhost:8501
