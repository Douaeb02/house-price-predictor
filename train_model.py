import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import pickle

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1) Génération du jeu de données (1000 échantillons, 7 features)
n = 1000
square_feet = np.random.randint(400, 4000, size=n)
bedrooms = np.random.randint(1, 6, size=n)
bathrooms = np.random.randint(1, 4, size=n)
age_years = np.random.randint(0, 60, size=n)
lot_size = np.random.randint(200, 10000, size=n)
garage_spaces = np.random.randint(0, 4, size=n)
neighborhood_score = np.round(np.random.uniform(0.0, 10.0, size=n),2)

# formule de prix 
prix_base = 20000
price = (prix_base
         + square_feet * 150
         + bedrooms * 20000
         + bathrooms * 15000
         - age_years * 2000
         + lot_size * 2
         + garage_spaces * 10000
         + neighborhood_score * 3000
         + np.random.normal(0, 20000, size=n)  # bruit
        )

df = pd.DataFrame({
    "square_feet": square_feet,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "lot_size": lot_size,
    "garage_spaces": garage_spaces,
    "neighborhood_score": neighborhood_score,
    "price": price
})

# 2) Split train/test
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# 3) Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Entraînement RandomForest
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 5) Évaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.4f}")

# 6) Sauvegarde
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Enregistrer un petit CSV
df.to_csv("data/synthetic_houses.csv", index=False)

print("Modèle, scaler et features sauvegardés dans ./models/")
