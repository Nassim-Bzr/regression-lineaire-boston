# TP Régression Linéaire - Boston Housing

## Description
Ce projet implémente une analyse complète de régression linéaire (simple et multiple) sur le dataset Boston Housing pour prédire les prix des maisons à Boston.

## Objectifs du TP
1. Charger et explorer le dataset Boston Housing
2. Afficher les statistiques descriptives des données
3. Nettoyer les données (gestion des valeurs manquantes, normalisation)
4. Construire un modèle de régression linéaire simple (1 variable)
5. Construire un modèle de régression linéaire multiple (plusieurs variables)
6. Évaluer les performances avec R² et RMSE
7. Visualiser les relations et les prédictions

## Dataset
Le dataset `train.csv` contient des données sur les ventes de maisons à Boston avec les colonnes suivantes :

- **CRIM** : Taux de criminalité par ville
- **ZN** : Proportion de terrains résidentiels zonés pour des lots de plus de 25 000 sq.ft
- **INDUS** : Proportion d'acres commerciaux non liés au commerce de détail par ville
- **CHAS** : Variable fictive Charles River (= 1 si le tronçon limite la rivière; 0 sinon)
- **NOX** : Concentration d'oxydes nitriques (parties par 10 million)
- **RM** : Nombre moyen de pièces par logement
- **AGE** : Proportion de logements occupés par leur propriétaire construits avant 1940
- **DIS** : Distances pondérées aux cinq centres d'emploi de Boston
- **RAD** : Indice d'accessibilité aux autoroutes radiales
- **TAX** : Taux d'imposition foncière à plein valeur par 10 000 dollars
- **PTRATIO** : Ratio élèves-enseignant par ville
- **B** : 1000(Bk - 0.63)^2 où Bk est la proportion de personnes de couleur par ville
- **LSTAT** : % de statut inférieur de la population
- **MEDV** : Valeur médiane des maisons occupées par leur propriétaire en 1000 dollars (variable cible)

## Installation

### Prérequis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Ou avec un fichier requirements.txt :
```bash
pip install -r requirements.txt
```

## Utilisation

### Option 1 : Script Python
Exécuter le script complet :
```bash
python main.py
```

### Option 2 : Notebook Jupyter
Ouvrir et exécuter le notebook interactif :
```bash
jupyter notebook regression_lineaire.ipynb
```

## Structure du projet
```
.
├── main.py                      # Script Python complet
├── regression_lineaire.ipynb    # Notebook Jupyter interactif
├── train.csv                    # Dataset Boston Housing
└── README.md                    # Ce fichier
```

## Résultats attendus

### Régression Linéaire Simple
- Utilise uniquement la variable **RM** (nombre moyen de pièces)
- Permet de comprendre la relation entre le nombre de pièces et le prix

### Régression Linéaire Multiple
- Utilise 7 variables : `rm`, `lstat`, `ptratio`, `dis`, `nox`, `tax`, `crim`
- Améliore significativement les performances par rapport au modèle simple

### Métriques d'évaluation
- **R² (Coefficient de détermination)** : Mesure la proportion de variance expliquée
- **RMSE (Root Mean Squared Error)** : Mesure l'erreur moyenne de prédiction en milliers de dollars

### Visualisations générées
1. Matrice de corrélation des variables
2. Graphique de régression simple (RM vs Prix)
3. Prédictions vs Valeurs réelles (Simple et Multiple)
4. Analyse des résidus
5. Importance des caractéristiques

## Concepts clés

### Régression Linéaire Simple
Modèle : `y = ax + b`
- Une seule variable explicative
- Facile à interpréter
- Performances limitées

### Régression Linéaire Multiple
Modèle : `y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`
- Plusieurs variables explicatives
- Capture mieux la complexité
- Nécessite la normalisation des données

### Normalisation
Utilisation de `StandardScaler` pour :
- Mettre toutes les variables à la même échelle
- Améliorer la convergence
- Faciliter la comparaison des coefficients

## Auteur
TP réalisé dans le cadre du cours de Machine Learning / Python IA - IPSSI

## Licence
Projet éducatif
