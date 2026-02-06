"""
QA SQuAD – Streamlit App for Hugging Face Spaces
Fused architecture (no separate FastAPI)
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
import time

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QA SQuAD",
    layout="centered",
    initial_sidebar_state="collapsed"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths - use local models or download from Hub
MODELS_CONFIG = {
    "distilbert": "models/distilbert",
    "bert": "models/bert",
    "roberta": "models/roberta",
}

# HuggingFace Hub fallback (if local models don't exist)
# You'll need to upload your fine-tuned models to your HF profile first
HF_MODELS = {
    "distilbert": "KenJend0/qa-squad-distilbert",  # Change to your HF username
    "bert": "KenJend0/qa-squad-bert",
    "roberta": "KenJend0/qa-squad-roberta",
}

# ═══════════════════════════════════════════════════════════════════════════
# Model Loading (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_name):
    """Load tokenizer and model with caching."""
    local_path = MODELS_CONFIG.get(model_name)
    
    if not model_name:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Try loading from local path first (for development)
    if os.path.exists(local_path):
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(local_path, local_files_only=True)
    else:
        # Download from HuggingFace Hub if local not available
        hf_model_id = HF_MODELS.get(model_name)
        if not hf_model_id:
            raise ValueError(f"Model {model_name} not found in HF Hub config")
        
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(hf_model_id)
    
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model


def answer_question(tokenizer, model, question: str, context: str):
    """
    Answer a question given context using the model.
    Returns (answer_text, confidence_score, inference_time_ms)
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )
    
    # RoBERTa doesn't use token_type_ids
    if "token_type_ids" in inputs:
        # Keep token_type_ids for BERT/DistilBERT, but check model architecture
        model_type = model.config.model_type
        if model_type == "roberta":
            inputs.pop("token_type_ids", None)
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency_ms = (time.time() - start_time) * 1000
    
    # Extract span
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    
    max_answer_length = 15
    best_score = -1e9
    best_start, best_end = 0, 0
    
    # Find best answer span
    for start_idx in range(len(start_logits)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start, best_end = start_idx, end_idx
    
    # Decode answer
    answer = tokenizer.decode(
        inputs["input_ids"][0][best_start:best_end + 1],
        skip_special_tokens=True
    )
    
    # Fallback if empty
    if not answer.strip():
        answer = "No answer found"
    
    return answer, best_score, latency_ms


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

st.title("🤖 Question Answering – SQuAD")
st.markdown("*Powered by BERT, DistilBERT, RoBERTa*")

col1, col2 = st.columns([2, 1])
with col1:
    model_name = st.selectbox(
        "📊 Choisir le modèle",
        ["distilbert", "bert", "roberta"],
        help="DistilBERT: rapide | BERT: précis | RoBERTa: robuste"
    )
with col2:
    st.metric("⚙️ Device", "GPU" if torch.cuda.is_available() else "CPU")

st.divider()

context = st.text_area(
    "📖 Contexte",
    height=200,
    placeholder="Collez le texte dans lequel chercher la réponse..."
)

question = st.text_input(
    "❓ Question",
    placeholder="Posez votre question ici..."
)

st.divider()

if st.button("🔍 Obtenir la réponse", type="primary", use_container_width=True):
    if not context or not question:
        st.warning("⚠️ Veuillez remplir le contexte ET la question")
    else:
        # Load model
        with st.spinner("⏳ Chargement du modèle..."):
            try:
                tokenizer, model = load_model(model_name)
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
                st.stop()
        
        # Get answer
        with st.spinner("⏳ Calcul de la réponse..."):
            try:
                answer, score, latency_ms = answer_question(tokenizer, model, question, context)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'inférence: {str(e)}")
                st.stop()
        
        # Display results
        st.success("✅ Réponse trouvée!")
        st.markdown(f"### {answer}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score (confiance)", f"{score:.2f}")
        with col2:
            st.metric("Temps inférence", f"{latency_ms:.1f} ms")
