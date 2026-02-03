# Transaction Risk Scoring – Fraud Detection (Machine Learning)

## Contexte
Dans le secteur bancaire, la détection de fraude est un enjeu critique :  
les transactions frauduleuses sont très rares mais coûteuses, tandis que les fausses alertes génèrent une charge opérationnelle importante.

Ce projet a pour objectif de construire un **système de scoring de risque** permettant d’estimer la probabilité qu’une transaction bancaire soit frauduleuse, à l’aide de techniques de **machine learning supervisé**.

---

## Objectifs du projet
- Analyser un jeu de données réel de transactions bancaires
- Gérer un **fort déséquilibre de classes** (fraude très minoritaire)
- Entraîner un modèle de classification interprétable
- Évaluer les performances avec des **métriques adaptées au contexte bancaire**
- Mettre en place un pipeline reproductible (entraînement, évaluation, prédiction)

---

## Pre-requis techniques

- Python 3 (execution de scripts, bases du langage)
- Utilisation d un environnement virtuel (venv)
- Bases en analyse de donnees (pandas, numpy)
- Notions de machine learning supervise
  - separation train / test
  - classification binaire
- Utilisation de scikit-learn
- Comprendre les metriques de base (ROC AUC, precision, recall)

---

## Librairies utilisees

- pandas : manipulation et analyse de donnees
- numpy : calculs numeriques
- scikit-learn : modelisation et evaluation
- matplotlib : visualisation des resultats
- joblib : sauvegarde et chargement du modele

--

## Données
**Important**  
Le fichier `data/creditcard.csv` n est **pas versionne dans ce depot** afin de respecter les limites de taille de GitHub

### Comment recuperer le dataset
1. Telecharger le dataset sur Kaggle  
   Credit Card Fraud Detection
2. Placer le fichier CSV dans le dossier :
data/creditcard.csv
Aucune autre modification n est necessaire.

---

- Dataset : **Credit Card Fraud Detection (Kaggle)**
- Environ 285 000 transactions
- Taux de fraude ≈ **0,17 %**
- Variables :


  - `Time` : temps écoulé depuis la première transaction (en secondes)
  - `Amount` : montant de la transaction
  - `V1` à `V28` : variables anonymisées
  - `Class` : 0 = transaction normale, 1 = fraude

Le dataset est volontairement **déséquilibré**, ce qui reflète un cas réel en banque.

---

## Méthodologie

### 1. Préparation et feature engineering
- Extraction de l’heure de la transaction à partir de la variable `Time`
- Transformation logarithmique du montant (`log_amount`) pour réduire l’asymétrie
- Séparation stricte des données en jeu d’entraînement et de test (split stratifié)
- Normalisation des variables numériques à l’aide d’un `StandardScaler`
  - le scaler est entraîné uniquement sur le jeu d’entraînement

---

### 2. Modélisation
- Modèle utilisé : **Régression Logistique**
- Justification :
  - modèle robuste
  - interprétable
  - adapté aux contraintes réglementaires du secteur bancaire
- Gestion du déséquilibre des classes via `class_weight="balanced"`

Le modèle produit un **score de risque** correspondant à la probabilité de fraude.

---

### 3. Évaluation
Les performances sont évaluées sur un jeu de test indépendant.

Métriques principales :
- **ROC-AUC** : capacité globale de séparation fraude / non-fraude
- **PR-AUC** : métrique prioritaire compte tenu du fort déséquilibre
- Matrice de confusion au seuil choisi
- Precision / Recall pour la classe fraude

Un seuil de décision initial est fixé à **0,30**, afin de privilégier la détection des fraudes (rappel élevé), tout en restant ajustable selon les contraintes métier.

---

## Résultats principaux

- ROC-AUC : **0.974**
- PR-AUC : **0.718**
- Taux de fraude : **0.17 %**

Ces résultats montrent que le modèle est capable de produire des alertes pertinentes malgré la rareté des fraudes.

---

## Organisation du projet
```BASH
ANALYSE-DE-DONNEES-DE-TRANSACTION-BANCAIRES/
├── data/
│ └── creditcard.csv
├── models/
│ └── model.joblib
├── outputs/
│ ├── metrics.json
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── pr_curve.png
│ └── feature_importance.png
│ └── classification_report.json
├── src/
│ ├── config.py
│ ├── data_utils.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
├── requirements.txt
└── README.md
```

---

## Exécution

### Installation
```bash
pip install -r requirements.txt
```
### Entraînement du modèle
```bash
python -m src.train
```
### Évaluation
```bash
python -m src.evaluate
```
### Limites et améliorations possibles
 - Tester d’autres modèles (Random Forest, Gradient Boosting)
 - Optimisation automatique du seuil de décision
 - Ajout de techniques d’IA explicable plus avancées (SHAP)
 - Déploiement sous forme d’API ou d’interface utilisateur

### Auteur
Projet réalisé par Omar Amoudi dans le cadre d’un apprentissage personnel en Machine Learning appliqué aux données bancaires, niveau Licence 3 Informatique.
