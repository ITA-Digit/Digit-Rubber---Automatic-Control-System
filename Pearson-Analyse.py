# Import der Bibliotheken
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Font und DPI-Einstellungen
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

# Laden des Extrusionsdatensatzes
file_path = 'Extruder_data_all.csv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter=';')  # Adjust the delimiter if necessary

# Ausklammern irrelevanter Paramater 
columns_to_exclude = ['D_Y_rechts', 'D_Y_mitte', 'D_Y_links', '1 deviation', '2 deviation', 'Datensatz', 'time [s]', 'T Wz']
data = data.drop(columns=columns_to_exclude, errors='ignore')

# Berechnung der Pearson-Matrix
correlation_matrix = data.corr()

# Darstellung der Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Pearson-Korrelations Heatmap')
plt.show()
