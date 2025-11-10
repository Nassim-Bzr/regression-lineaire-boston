import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
sns.set_palette("husl")

df = pd.read_csv("train.csv")
print("Dimensions du dataset:", df.shape)
df.head()

print("=== STATISTIQUES DESCRIPTIVES ===")
df.describe()

# Informations sur les types de données et valeurs manquantes
print("\n=== INFORMATIONS SUR LE DATASET ===")
df.info()

# Vérification des valeurs manquantes
print("\n=== VALEURS MANQUANTES ===")
missing = df.isnull().sum()
print(missing[missing > 0])
if missing.sum() == 0:
    print("Aucune valeur manquante détectée !")

# Suppression de la colonne ID (non pertinente pour la prédiction)
df_clean = df.drop("ID", axis=1)

# Gestion des valeurs manquantes (si nécessaire)
# Ici, pas de valeurs manquantes, mais on pourrait faire:
# df_clean = df_clean.fillna(df_clean.mean())

print("Dataset nettoyé - Shape:", df_clean.shape)

# Matrice de corrélation
plt.figure(figsize=(14, 10))
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation des variables")
plt.tight_layout()
plt.show()

# Corrélation avec la cible (medv)
print("\n=== CORRÉLATION AVEC LE PRIX (medv) ===")
corr_with_target = correlation_matrix["medv"].sort_values(ascending=False)
print(corr_with_target)

# Sélection de la caractéristique RM (nombre moyen de pièces)
X_simple = df_clean[["rm"]]
y = df_clean["medv"]

# Division en ensembles d'entraînement et de test (80/20)
X_train_simple, X_test_simple, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print(f"Taille ensemble d'entraînement: {len(X_train_simple)}")
print(f"Taille ensemble de test: {len(X_test_simple)}")

# Création et entraînement du modèle de régression linéaire simple
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

print("=== MODÈLE DE RÉGRESSION LINÉAIRE SIMPLE ===")
print(f"Coefficient (pente): {model_simple.coef_[0]:.4f}")
print(f"Intercept (ordonnée à l'origine): {model_simple.intercept_:.4f}")
print(
    f"\nÉquation: medv = {model_simple.coef_[0]:.4f} * rm + {model_simple.intercept_:.4f}"
)


# Prédictions
y_pred_simple = model_simple.predict(X_test_simple)

# Évaluation
r2_simple = r2_score(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

print("=== PERFORMANCE DU MODÈLE SIMPLE ===")
print(f"R² Score: {r2_simple:.4f}")
print(f"RMSE: {rmse_simple:.4f} (milliers de dollars)")

# Sélection de plusieurs caractéristiques pertinentes
features = ["rm", "lstat", "ptratio", "dis", "nox", "tax", "crim"]
X_multiple = df_clean[features]

# Division en ensembles d'entraînement et de test
X_train_multiple, X_test_multiple, y_train_m, y_test_m = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_multiple)
X_test_scaled = scaler.transform(X_test_multiple)

print("Variables sélectionnées:", features)

# Création et entraînement du modèle de régression linéaire multiple
model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train_m)

print("=== MODÈLE DE RÉGRESSION LINÉAIRE MULTIPLE ===")
print(f"Intercept: {model_multiple.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(features, model_multiple.coef_):
    print(f"  {feature:10s}: {coef:8.4f}")

# Prédictions
y_pred_multiple = model_multiple.predict(X_test_scaled)

# Évaluation
r2_multiple = r2_score(y_test_m, y_pred_multiple)
rmse_multiple = np.sqrt(mean_squared_error(y_test_m, y_pred_multiple))

print("=== PERFORMANCE DU MODÈLE MULTIPLE ===")
print(f"R² Score: {r2_multiple:.4f}")
print(f"RMSE: {rmse_multiple:.4f} (milliers de dollars)")

# Tableau comparatif
comparison = pd.DataFrame(
    {
        "Modèle": ["Régression Simple (RM)", "Régression Multiple"],
        "R² Score": [r2_simple, r2_multiple],
        "RMSE": [rmse_simple, rmse_multiple],
        "Nombre de features": [1, len(features)],
    }
)

print("\n=== COMPARAISON DES MODÈLES ===")
print(comparison.to_string(index=False))

print(f"\nAmélioration du R²: {((r2_multiple - r2_simple) / r2_simple * 100):.2f}%")
print(f"Réduction du RMSE: {((rmse_simple - rmse_multiple) / rmse_simple * 100):.2f}%")

# Visualisation de la régression simple
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test, alpha=0.6, label="Valeurs réelles")
plt.plot(X_test_simple, y_pred_simple, color="red", linewidth=2, label="Prédictions")
plt.xlabel("RM (Nombre moyen de pièces)")
plt.ylabel("MEDV (Prix en milliers de $)")
plt.title("Régression Linéaire Simple: RM vs Prix")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Graphique des prédictions vs valeurs réelles - Modèle Simple
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_simple, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title(f"Régression Simple\nR² = {r2_simple:.4f}")
plt.grid(True, alpha=0.3)

# Graphique des prédictions vs valeurs réelles - Modèle Multiple
plt.subplot(1, 2, 2)
plt.scatter(y_test_m, y_pred_multiple, alpha=0.6)
plt.plot(
    [y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], "r--", lw=2
)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title(f"Régression Multiple\nR² = {r2_multiple:.4f}")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Résidus du modèle multiple
residuals = y_test_m - y_pred_multiple

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_multiple, residuals, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Graphique des résidus")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.title("Distribution des résidus")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Importance des caractéristiques (coefficients)
feature_importance = pd.DataFrame(
    {"Feature": features, "Coefficient": model_multiple.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 6))
colors = ["green" if x > 0 else "red" for x in feature_importance["Coefficient"]]
plt.barh(
    feature_importance["Feature"],
    feature_importance["Coefficient"],
    color=colors,
    alpha=0.7,
)
plt.xlabel("Coefficient (données normalisées)")
plt.title("Importance des caractéristiques dans le modèle multiple")
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.show()

print(feature_importance)
