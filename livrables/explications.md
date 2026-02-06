### Étape 1 – Dataset et exploration

Nous utilisons le dataset SQuAD (Stanford Question Answering Dataset),
un jeu de données de référence pour le question answering extractif.

Chaque exemple est composé :

d'un contexte (paragraphe),

d'une question,

d'une réponse sous forme de span dans le contexte
(texte de la réponse et position de début).

L'objectif du modèle est de prédire les indices de début et de fin
de la réponse directement dans le texte du contexte.

Avant l'entraînement, nous explorons le dataset afin de vérifier la
cohérence entre le texte des réponses et leurs positions annotées
dans les contextes.

Nous utilisons le tokenizer associé au modèle Transformer
afin de transformer les textes en séquences de tokens exploitables
par le réseau de neurones.

### Étape 2 – Preprocessing et alignement des labels

Avant l'entraînement, les données doivent être prétraitées afin d'être
compatibles avec les modèles Transformer.

Les contextes pouvant dépasser la longueur maximale autorisée par le
modèle, ils sont découpés en plusieurs segments à l'aide d'un
stride glissant.
Pour chaque segment, les positions de début et de fin de la réponse
sont réalignées avec les tokens générés par le tokenizer.

Lorsque la réponse n'est pas contenue dans un segment donné, le modèle
est entraîné à prédire le token [CLS], ce qui permet de gérer
correctement ces cas lors de l'apprentissage.

### Étape 3 – Fine-tuning (Trainer)

Nous fine-tunons un modèle Transformer pour la tâche de
question answering extractif.
Le modèle apprend à prédire deux distributions de probabilités :

une pour le début de la réponse,

une pour la fin de la réponse.

L'entraînement est réalisé à l'aide de l'API Trainer de la
bibliothèque Transformers, afin de disposer d'un pipeline
reproductible incluant :

la gestion des hyperparamètres,

le suivi des logs,

la sauvegarde du meilleur modèle,

l'évaluation à chaque époque.

Les artefacts d'entraînement (checkpoints et logs) sont stockés dans
le dossier outputs/.

### Étape 4 – Évaluation

Le modèle fine-tuné est évalué sur le jeu de validation du dataset SQuAD.

Les performances sont mesurées à l'aide des métriques
Exact Match (EM) et F1-score, qui sont les métriques de
référence pour le question answering extractif.

Afin de calculer les métriques Precision, Recall et AUC,
nous avons transformé la tâche en problème de classification binaire :
une prédiction est correcte si elle correspond exactement à la réponse
de référence (Exact Match = 1), incorrecte sinon.
Un score de confiance est calculé comme la somme des logits de début
et de fin.

Cette transformation présente des limites : elle ne reflète pas la
nature continue du problème de QA extractif et ignore les réponses
partiellement correctes. Les métriques EM et F1 restent donc les
indicateurs principaux de performance.

Le temps d'inférence moyen par question est également mesuré.

### Comparaison de plusieurs modèles

Afin de comparer les performances, le même pipeline d'entraînement
et d'évaluation est appliqué à trois architectures différentes :

DistilBERT,

BERT-base,

RoBERTa-base.

Seul le modèle pré-entraîné est modifié ; les données, la procédure de
prétraitement et les hyperparamètres restent identiques afin de garantir
une comparaison équitable entre les architectures.

### BERT-base

BERT-base est un modèle de référence pour le question answering extractif.
Il comporte 12 couches Transformer avec attention bidirectionnelle.

Nous l'utilisons comme point de comparaison principal.

### DistilBERT

DistilBERT est une version allégée de BERT obtenue par distillation.
Il ne possède que 6 couches Transformer, ce qui réduit le temps de calcul.

L'objectif est de conserver des performances acceptables tout en étant
plus rapide et moins coûteux en ressources.
DistilBERT permet d'évaluer le compromis entre rapidité et qualité.

### RoBERTa-base

RoBERTa-base utilise la même architecture que BERT-base, mais avec
une méthode de pré-entraînement améliorée :
davantage de données, suppression du Next Sentence Prediction,
masquage dynamique.

Ces modifications conduisent généralement à de meilleures performances
sur les tâches de NLP, au prix d'un temps de calcul plus élevé.

### Comparaison globale

Les trois modèles présentent des caractéristiques différentes :

- DistilBERT : plus rapide, moins précis
- BERT-base : compromis performances / temps d'exécution
- RoBERTa-base : meilleures performances attendues, plus lent

### Optimisation du temps d'entraînement

Pour réduire le temps de calcul, l'entraînement a été réalisé sur
un sous-ensemble du dataset SQuAD
(2000 exemples pour l'entraînement et 500 pour la validation).

Cette approche permet de comparer les architectures avec
des ressources limitées. Les performances absolues sont
réduites par rapport à un entraînement complet, mais les
tendances relatives entre modèles restent observables.

### Synthèse des résultats

Les résultats montrent un compromis entre performances et coût de calcul.

DistilBERT offre les temps d'inférence les plus faibles, avec des
performances réduites.
BERT-base constitue un équilibre entre qualité et temps d'exécution.

Les scores absolus sont faibles en raison du sous-ensemble
d'entraînement réduit.

### Optimisation de l'extraction des réponses

Lors des tests, le modèle pouvait extraire des spans de longueur
excessive ou incorrects.

Pour améliorer la qualité, une contrainte de longueur maximale
(15 tokens) a été ajoutée.
L'algorithme considère tous les spans possibles respectant cette
limite et sélectionne celui ayant le meilleur score
(somme des logits de début et de fin).

Cette approche est utilisée dans les exemples officiels de
Hugging Face.

### Interface utilisateur

Une interface utilisateur interactive a été développée à l'aide de
FastAPI pour le backend et Streamlit pour le frontend.

Le backend expose une API permettant de charger les modèles fine-tunés
et de générer des réponses à partir d'un contexte et d'une question.

Le frontend Streamlit permet à l'utilisateur de :

saisir un contexte,

poser une question,

sélectionner le modèle de question answering à utiliser.

---

### Liens du projet

- **Dépôt GitHub** : `https://github.com/KenJend0/qa-project`
- **Démo Hugging Face Spaces** : `https://huggingface.co/spaces/KenJend0/qa-squad-transformers`
