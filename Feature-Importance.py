# Import der Bibliotheken
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

# Definieren der Ein- und Ausgangsgrößen
file_path = 'Extruder_data_all.csv'
data = pd.read_csv(file_path, delimiter=';')

# Laden des Extrusionsdatensatzes
input_columns = ['Drehzahl [1/min]', 'T (Schnecke)', 'T (Stiftzone)', 'T (Werk-zeug)', 'T (Speisewalze)', 'T (Einzug)']
target_columns = ['T 1 [C]  (0,0 mm)', 'T 2 [C]  (10,0 mm)', 'T 3 [C]  (20,0 mm)', 
                  'T 4 [C]  (30,0 mm)', 'T 5 [C]  (40,0 mm)', 'T 6 [C]  (50,0 mm)', 
                  'T 7 [C]  (60,0 mm)', 'T 8 [C]', 'T b', 'T a']

# Aufteilung der Daten in Training- und Testdaten
X = data[input_columns]
y = data[target_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, rand-om_state=42)

# Random Forest Model, Importance Score und Features 
rf = RandomForestRegressor(n_estimators=2000, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
num_features_to_keep = 6
top_indices = sorted_indices[:num_features_to_keep]
indices = np.argsort(importances)[::-1]

# Featurebezeichnung 
top_features = X_train.columns[top_indices]
top_scores = importances[top_indices]
print("Features und deren Importance-Score:")
for feature, score in zip(top_features, top_scores):
    print(f"{feature}: {score:.4f}")

# Darstellung des Balkendiagramms
plt.figure()
plt.title("Feature-Importance")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
