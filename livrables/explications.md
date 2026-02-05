## Étape 1 – Dataset et exploration

Nous utilisons le dataset SQuAD (Stanford Question Answering Dataset), 
un jeu de données de référence pour le question answering extractif.

Chaque exemple est composé :
- d’un contexte (paragraphe),
- d’une question,
- d’une réponse sous forme de span dans le contexte 
  (texte + position de début).

L’objectif du modèle est de prédire les indices de début et de fin
de la réponse dans le texte.

Avant l’entraînement, nous explorons le dataset et vérifions la
cohérence entre le texte de la réponse et sa position dans le contexte.

Nous utilisons le tokenizer associé au modèle Transformer afin de
préparer les entrées du réseau de neurones.

## Étape 2 – Preprocessing et alignement des labels

Avant l’entraînement, les données doivent être prétraitées afin d’être
compatibles avec les modèles Transformer.

Les contextes étant parfois longs, ils sont découpés en plusieurs
segments à l’aide d’un stride. Pour chaque segment, les positions de
début et de fin de la réponse sont alignées avec les tokens générés
par le tokenizer.

Lorsque la réponse n’est pas contenue dans un segment, le modèle est
entraîné à prédire le token [CLS], ce qui permet de gérer ces cas.

## Étape 3 – Fine-tuning (Trainer)

Nous fine-tunons un modèle Transformer pour le question answering extractif.
Le modèle prédit deux distributions : début et fin de réponse dans le contexte.

L’entraînement est réalisé avec l’API Trainer (Transformers) afin d’avoir
un pipeline reproductible : paramètres d’entraînement, logs, sauvegarde
du meilleur checkpoint et évaluation à chaque époque.

Les artefacts (checkpoints et logs) sont stockés dans le dossier outputs/.

## Étape 4 – Évaluation

Le modèle fine-tuné est évalué sur le jeu de validation du dataset SQuAD.

Les performances sont mesurées à l’aide des métriques Exact Match (EM)
et F1-score, qui sont des métriques standards pour le question answering
extractif.
Afin de calculer les métriques Precision, Recall, ROC et AUC, la tâche
de question answering extractif est ramenée à un problème de
classification binaire.

Une prédiction est considérée comme correcte si elle correspond
exactement à la réponse de référence (Exact Match = 1).
Le score de confiance est obtenu à partir des logits de début et de fin
de réponse produits par le modèle.

Cette approche permet de comparer les modèles à l'aide de métriques
classiques de classification tout en conservant la tâche de QA.
Le temps d’inférence moyen par question est également mesuré afin
d’évaluer les performances du modèle en conditions d’utilisation réelle.

## Comparaison de plusieurs modèles

Afin de comparer les performances, le même pipeline d’entraînement
et d’évaluation est appliqué à trois architectures différentes :
DistilBERT, BERT-base et RoBERTa-base.

Seul le modèle pré-entraîné est modifié, les données et les
hyperparamètres restant identiques afin de garantir une comparaison
équitable.
## Synthèse des résultats

Les résultats montrent un compromis clair entre performances et coût
de calcul.

DistilBERT offre les temps d'inférence les plus faibles, tandis que
RoBERTa obtient généralement les meilleures performances en termes
de F1-score et d'AUC.

BERT-base constitue un compromis équilibré entre qualité des réponses
et temps d'exécution.

Ces résultats confirment l'impact du choix de l'architecture sur les
performances d'un système de question answering.

## Interface utilisateur

Une interface utilisateur interactive a été développée à l'aide de
FastAPI pour le backend et Streamlit pour le frontend.

Le backend expose une API permettant de charger les modèles fine-tunés
et de générer des réponses à partir d'un contexte et d'une question.

Le frontend Streamlit permet à l'utilisateur de saisir un contexte,
poser une question et choisir le modèle de question answering à utiliser.