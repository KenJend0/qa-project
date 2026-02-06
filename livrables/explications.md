### Étape 1 – Dataset et exploration

Nous utilisons le dataset SQuAD (Stanford Question Answering Dataset),
un jeu de données de référence pour le question answering extractif.

Chaque exemple est composé :

d’un contexte (paragraphe),

d’une question,

d’une réponse sous forme de span dans le contexte
(texte de la réponse et position de début).

L’objectif du modèle est de prédire les indices de début et de fin
de la réponse directement dans le texte du contexte.

Avant l’entraînement, nous explorons le dataset afin de vérifier la
cohérence entre le texte des réponses et leurs positions annotées
dans les contextes.

Nous utilisons le tokenizer associé au modèle Transformer
afin de transformer les textes en séquences de tokens exploitables
par le réseau de neurones.

### Étape 2 – Preprocessing et alignement des labels

Avant l’entraînement, les données doivent être prétraitées afin d’être
compatibles avec les modèles Transformer.

Les contextes pouvant dépasser la longueur maximale autorisée par le
modèle, ils sont découpés en plusieurs segments à l’aide d’un
stride glissant.
Pour chaque segment, les positions de début et de fin de la réponse
sont réalignées avec les tokens générés par le tokenizer.

Lorsque la réponse n’est pas contenue dans un segment donné, le modèle
est entraîné à prédire le token [CLS], ce qui permet de gérer
correctement ces cas lors de l’apprentissage.

### Étape 3 – Fine-tuning (Trainer)

Nous fine-tunons un modèle Transformer pour la tâche de
question answering extractif.
Le modèle apprend à prédire deux distributions de probabilités :

une pour le début de la réponse,

une pour la fin de la réponse.

L’entraînement est réalisé à l’aide de l’API Trainer de la
bibliothèque Transformers, afin de disposer d’un pipeline
reproductible incluant :

la gestion des hyperparamètres,

le suivi des logs,

la sauvegarde du meilleur modèle,

l’évaluation à chaque époque.

Les artefacts d’entraînement (checkpoints et logs) sont stockés dans
le dossier outputs/.

### Étape 4 – Évaluation

Le modèle fine-tuné est évalué sur le jeu de validation du dataset SQuAD.

Les performances sont mesurées à l’aide des métriques
Exact Match (EM) et F1-score, qui sont les métriques de
référence pour le question answering extractif.

Afin de calculer les métriques Precision, Recall, ROC et
AUC, la tâche de question answering extractif est ramenée à un
problème de classification binaire.

Une prédiction est considérée comme correcte si la réponse prédite
correspond exactement à la réponse de référence
(Exact Match = 1).
Un score de confiance est associé à chaque prédiction, calculé à
partir de la somme des logits de début et de fin produits par le modèle.

Cette approche permet de comparer les modèles à l’aide de métriques
classiques de classification tout en conservant la spécificité de la
tâche de question answering.

Le temps d’inférence moyen par question est également mesuré afin
d’évaluer les performances des modèles en conditions d’utilisation
réelle.

### Comparaison de plusieurs modèles

Afin de comparer les performances, le même pipeline d’entraînement
et d’évaluation est appliqué à trois architectures différentes :

DistilBERT,

BERT-base,

RoBERTa-base.

Seul le modèle pré-entraîné est modifié ; les données, la procédure de
prétraitement et les hyperparamètres restent identiques afin de garantir
une comparaison équitable entre les architectures.

### Optimisation du temps d'entraînement

Afin de réduire le temps de calcul tout en conservant une comparaison
pertinente entre les modèles, l’entraînement a été réalisé sur un
sous-ensemble représentatif du dataset SQuAD
(2000 exemples pour l’entraînement et 500 pour la validation).

Cette approche permet d’analyser les différences de performances entre
les architectures tout en respectant les contraintes de temps et de
ressources matérielles.
Les tendances observées sur ce sous-ensemble restent représentatives
du comportement des modèles sur l’ensemble complet du dataset.

### Synthèse des résultats

Les résultats mettent en évidence un compromis clair entre performances
et coût de calcul.

DistilBERT offre les temps d’inférence les plus faibles, au prix
de performances légèrement inférieures.
RoBERTa-base obtient globalement les meilleures performances,
notamment en termes de F1-score et d’AUC.
BERT-base constitue un compromis équilibré entre qualité des
réponses et temps d’exécution.

Ces résultats confirment l’impact du choix de l’architecture sur les
performances d’un système de question answering.

### Optimisation de l'extraction des réponses

Lors des tests interactifs, nous avons observé que le modèle pouvait
parfois extraire une entité incorrecte lorsque plusieurs réponses
plausibles étaient présentes dans le contexte.

Ce comportement est inhérent au question answering extractif, où le
modèle sélectionne les tokens ayant les logits les plus élevés sans
contrainte explicite sur la longueur ou la cohérence de la réponse.

Afin d’améliorer la qualité des prédictions, une contrainte sur la
longueur maximale des réponses (15 tokens) a été introduite.
L’algorithme considère désormais l’ensemble des spans possibles
respectant cette limite et sélectionne celui ayant le meilleur score
combiné (somme des logits de début et de fin).

Cette approche est conforme aux recommandations des exemples officiels
de Hugging Face et améliore significativement la précision des
réponses extraites.

### Interface utilisateur

Une interface utilisateur interactive a été développée à l’aide de
FastAPI pour le backend et Streamlit pour le frontend.

Le backend expose une API permettant de charger les modèles fine-tunés
et de générer des réponses à partir d’un contexte et d’une question.

Le frontend Streamlit permet à l’utilisateur de :

saisir un contexte,

poser une question,

sélectionner le modèle de question answering à utiliser.