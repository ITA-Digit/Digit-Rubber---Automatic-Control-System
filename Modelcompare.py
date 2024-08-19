# Import der Bibliotheken
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Einstellung des Seeds zur Reproduzierbarkeit
Seed_1 = 1
os.environ['PYTHONHASHSEED']=str(Seed_1)
np.random.seed(Seed_1)
tf.random.set_seed(Seed_1)
import random
random.seed(Seed_1)

# Laden des Extrusionsdatensatzes
file_path = 'Extruder_data_all.csv'
data = pd.read_csv(file_path, delimiter=';')

# Definieren der Ein- und Ausgangsgrößen
input_columns = ['Drehzahl [1/min]', 'T (Werkzeug)']
target_columns = ['T 1 [C]  (0,0 mm)', 'T 2 [C]  (10,0 mm)', 'T 3 [C]  (20,0 mm)',
                  'T 4 [C]  (30,0 mm)', 'T 5 [C]  (40,0 mm)', 'T 6 [C]  (50,0 mm)',
                  'T 7 [C]  (60,0 mm)', 'T 8 [C]', 'T b', 'T a']

# Aufteilung der Daten in Training- und Testdaten 
X = data[input_columns]
y = data[target_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, rand-om_state=42)

# Datenvorverarbeitung mittels MinMaxScaler
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Lineares Regressions-Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f'MSE - Lineare Regression: {mse_linear}')

# Neuronenanzahl der KNN
Neuron=32

# FFN Model
ffn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(Neuron, activation='relu', in-put_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])

ffn_model.compile(optimizer='adam', loss='mean_squared_error')
ffn_model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=32, vali-dation_split=0.2)

y_pred_scaled_ffn = ffn_model.predict(X_test_scaled)
y_pred_ffn = scaler_y.inverse_transform(y_pred_scaled_ffn)
mse_ffn = mean_squared_error(y_test, y_pred_ffn)
print(f'MSE - FFN: {mse_ffn}')

# Datenshaping für RNN und LSTM
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# RNN Model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(Neuron, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])

rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train_rnn, y_train_scaled, epochs=10, batch_size=32, valida-tion_split=0.2)

y_pred_scaled_rnn = rnn_model.predict(X_test_rnn)
y_pred_rnn = scaler_y.inverse_transform(y_pred_scaled_rnn)
mse_rnn = mean_squared_error(y_test, y_pred_rnn)
print(f'MSE - RNN: {mse_rnn}')

# LSTM Model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(Neuron, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_rnn, y_train_scaled, epochs=10, batch_size=32, valida-tion_split=0.2)

y_pred_scaled_lstm = lstm_model.predict(X_test_rnn)
y_pred_lstm = scaler_y.inverse_transform(y_pred_scaled_lstm)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

print(f'MSE - LSTM: {mse_lstm}')
print(f'MSE - RNN: {mse_rnn}')
print(f'MSE - FFN: {mse_ffn}')
print(f'MSE - Linear Regression: {mse_linear}')
