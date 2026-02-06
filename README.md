# Question Answering – Fine-tuning SQuAD

Projet DataScale M2 UVSQ – Fine-tuning de modèles QA extractifs sur SQuAD.

## Liens du projet

- **Dépôt GitHub** : `https://github.com/KenJend0/qa-project`
- **Démo Hugging Face Spaces** : `https://huggingface.co/spaces/KenJend0/qa-squad-transformers`

---

## Objectifs
- Fine-tuner 3 modèles NLP (BERT, DistilBERT, RoBERTa) sur SQuAD
- Comparer performances : EM, F1, Precision, Recall, AUC, temps d'inférence
- Développer une interface utilisateur (FastAPI + Streamlit)
- Déployer sur Hugging Face Spaces

## Stack technique
- PyTorch, Hugging Face Transformers, Datasets
- FastAPI (backend API)
- Streamlit (frontend interactif)
- pdfplumber (extraction texte PDF)

## Structure du projet

```
qa-project/
├── notebooks/              # Notebooks Jupyter (exploration, training, évaluation)
│   ├── 00_exploration_data.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_training_*.ipynb
│   ├── 03_evaluation_*.ipynb
│   ├── 04_comparison_final.ipynb
│   └── outputs/           # Résultats, checkpoints, datasets tokenisés
├── app/                   # Application locale
│   ├── api.py            # Backend FastAPI
│   └── streamlit_app.py  # Frontend Streamlit
├── hf/                    # Version pour déploiement Hugging Face Spaces
├── livrables/             # Documents rendus (explications.md)
└── requirements.txt       # Dépendances Python
```

## Installation

```bash
# Cloner le dépôt
git clone <repo_url>
cd qa-project

# Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt
```

# Installer dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Lancer le backend FastAPI

```bash
uvicorn app.api:app --reload
```

API disponible sur http://127.0.0.1:8000

### 2. Lancer l'interface Streamlit

Dans un nouveau terminal :

```bash
streamlit run app/streamlit_app.py
```

Interface disponible sur http://localhost:8501

### 3. Utiliser l'interface

L'interface permet de :
- Choisir un modèle (BERT, DistilBERT ou RoBERTa)
- Charger un contexte via fichier PDF/TXT ou saisie manuelle
- Poser une question
- Obtenir la réponse avec score de confiance et temps d'inférence

## Modèles fine-tunés

Les trois modèles sont fine-tunés sur un sous-ensemble de SQuAD :
- **BERT-base** : baseline, bon compromis performance/vitesse
- **DistilBERT** : version allégée, plus rapide, légèrement moins précis
- **RoBERTa-base** : pré-entraînement amélioré, meilleures performances attendues

Les checkpoints sont stockés dans `notebooks/outputs/checkpoints/`.

## Métriques d'évaluation

- Exact Match (EM) : pourcentage de réponses exactement correctes
- F1-score : mesure token-level entre prédiction et référence
- Precision, Recall, AUC : transformation binaire (EM=1 si correct)
- Temps d'inférence : latence moyenne par question (ms)

## Déploiement

Le dossier `hf/` contient la version adaptée pour Hugging Face Spaces.

## Livrables

- Notebooks complets avec exploration, preprocessing, training, évaluation
- Application interactive (FastAPI + Streamlit)
- Documentation technique (`livrables/explications.md`)
- Résultats comparatifs des 3 modèles (`notebooks/outputs/comparison_results.json`)
