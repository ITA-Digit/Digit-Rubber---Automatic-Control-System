# Import der Bibliotheken
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
import joblib
from keras.models import model_from_json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, me-an_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Einstellung des Seeds zur Reproduzierbarkeit 
Seed_1 = 42
MODEL_NAME = 'Model_1'
DATA_FILE = 'Data_Mischer.csv'
SCALER_FILE = f'{MODEL_NAME}_scaler.gz'
os.environ['PYTHONHASHSEED']=str(Seed_1)
np.random.seed(Seed_1)
tf.random.set_seed(Seed_1)
import random
random.seed(Seed_1)

# Laden des Extrusionsdatensatzes
def load_data(file_path, delimiter=','):
    return pd.read_csv(file_path, delimiter=delimiter)

# Definieren der Ein- und Ausgangsgrößen
def preprocess_data(input_data):
    Stellgroeßen = input_data[['Stempeldruck [bar]', 'Stempelweg [mm]', 'Dreh-zahl [1/min]', 'Temp. TCU-1 Sa/Kl/St [C]', 'Temp. TCU-2 Rotor [C]','Chargenzeit [s]']]
    Messgroeßen = input_data[['Mischungstemp. [C]']]

    return Stellgroeßen, Messgroeßen

# Berechnung der Bewertungsmetriken (MAE, MSE, MAPE)
def calculate_metrics(actual, predicted):
    if isinstance(actual, pd.DataFrame):
        actual = actual.to_numpy()
    if isinstance(predicted, pd.DataFrame):
        predicted = predicted.to_numpy()
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mae, mse, mape

# Berechnung durchschnittlicher MAE, MSE, MAPE
def calculate_and_display_metrics(actual_train, predicted_train, actual_test, predicted_test):

    # Metrik für die Trainingsdaten
    mae_train, mse_train, mape_train = calculate_metrics(actual_train, predic-ted_train)
    print(f"\nTraining Metrics:\nMSE: {mse_train:.4f}\nMAE: {ma-e_train:.4f}\nMAPE: {mape_train:.4f}%")

    # Metrik für die Validierungsdaten
    mae_test, mse_test, mape_test = calculate_metrics(actual_test, predic-ted_test)
    print(f"\nTest Metrics:\nMSE: {mse_test:.4f}\nMAE: {mae_test:.4f}\nMAPE: {mape_test:.4f}%")

# Einstellung der Schriftart, Font, cm, Graphen
cm = 1/(2.54)
fpath = Path(mpl.get_data_path(), r'C:\Users\marco\OneDrive\Desktop\Stuff\Font\ArialMTStd-Light.otf')
def graphen_einstellungen():
    plt.rcParams['axes.edgecolor'] = "#000000"
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams['axes.labelcolor'] = "#000000"
    plt.rcParams['xtick.color'] = "#000000"
    plt.rcParams['ytick.color'] = "#000000"
    plt.rcParams['font.size'] = '12'

# Datenvorverarbeitung mittels MinMaxScaler
def scale_datasets(data_train, data_test):
    scaler = MinMaxScaler()
    data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), co-lumns=data_train.columns)
    data_test_scaled = pd.DataFrame(scaler.transform(data_test), co-lumns=data_test.columns)
    joblib.dump(scaler, SCALER_FILE)
    return data_train_scaled, data_test_scaled

# Darstellung der Verlustverlaufskurve
def plot_loss(epoch_count, training_loss, validation_loss, ylim=None, xlim=None):
    plt.figure(figsize=(20, 10), dpi=300)
    plt.plot(epoch_count, training_loss, 'r-', label='Training Loss', c='tab:orange',)
    plt.plot(epoch_count, validation_loss, 'b-', label='Validation Loss', c='b',)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    plt.grid(axis='both', linestyle='-', alpha=1.0, color='black')
    plt.show()

# Darstellung des Residuum-Plots
def plot_residuals(predictions, residuals, data_label, color):
    plt.figure(figsize=(20, 10), dpi=300)
    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    plt.scatter(predictions, residuals, c=color, label=f'{data_label} Data')
    plt.title(f'Residual Plot for {data_label} Data')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(axis='both', linestyle='-', alpha=1.0, color='black')
    plt.ylim(ymin=-20, ymax=20)
    plt.show()
    
# Main-Funktion
def main():

    # Laden des Datensatzes
    input_data = load_data(DATA_FILE)
    Stellgroeßen, Messgroeßen = preprocess_data(input_data)
    data = Stellgroeßen
    target = Messgroeßen

    # Trainings- und Validierungssplit (80/20)
    data_train = data.iloc[1:1386] # 80 %
    data_test = data.iloc[1386:1733] # 20 %
    target_train = target.iloc[1:1386] # 80 %
    target_test = target.iloc[1386:1733] # 20 %

    # Ausgabe des Trainings- und Validierungssplit
    print('Groeße des Train/Test-Splits')
    print(data_train.shape, target_train.shape)
    print(data_test.shape, target_test.shape)

    # Normalisieren der Daten durch MinMaxScaler
    data_train_scaled, data_test_scaled = scale_datasets(data_train, da-ta_test)
    
    # EarlyStopping: Abbruch, wenn sich die Prädiktionsgenauigkeit nicht ver-bessert
    early_stopping = EarlyStopping(monitor='val_loss', patience=250, resto-re_best_weights=True) 
    # ModelCheckpoint callbacks - Speichern des Modells mit der höchsten Ge-nauigkeit
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', sa-ve_best_only=True)

    # Modellarchitektur
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, kernel_initializer='normal',
                                  activation=tf.keras.activations.relu),  
            tf.keras.layers.Dense(128, activation=tf.keras.activations.relu,
                                  activity_regularizer=tf.keras.regularizers.l1(0.0061)),
            tf.keras.layers.Dense(128, activation=tf.keras.activations.relu,
                                  activity_regularizer=tf.keras.regularizers.l1(0.0061)),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
        ])
    
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                      optimizer=tf.keras.optimizers.Adam(
                          learning_rate=0.003688),  
                      metrics=['MAPE'])

    history = model.fit(
                        data_train_scaled.values,
                        target_train.values,
                        epochs=3133,
                        batch_size=64,
                        validation_data=(data_test_scaled.values, target_test.values),
                        callbacks=[early_stopping, checkpoint]
                        )
    
    # Ausgabe des Epochs bei dem die höchste Prädiktionsgenauigkeit erreicht wurde
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    print(f"Hoehste Genauigkeit des Validierungsverlust bei Epoch ereicht: {best_epoch}")

    # Speicherung des Modells und Gewichte für den Einsatz in der Steuerung
    model_json = model.to_json()
    with open(MODEL_NAME + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(MODEL_NAME + '.h5')

    # Historisierung des Trainings für den Plot
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']  
    epoch_count = range(1, len(training_loss) + 1)

    # Anwendung des Modells auf Training- und Validierungsdaten
    target_pred = model.predict(data_test_scaled.values)
    train_pred = model.predict(data_train_scaled.values)

    # Vorbereitung für grafische Darstellung 
    target.reset_index(inplace=True)
    train_pred = pd.DataFrame(train_pred, columns=target.iloc[:, 1:].columns)
    train_pred.reset_index(inplace=True)
    train_pred['index'] = train_pred['index']
    rows_train = len(train_pred.axes[0])

    target_pred = pd.DataFrame(target_pred, columns=target.iloc[:, 1:].columns)
    target_pred.reset_index(inplace=True)
    target_pred['index'] = target_pred['index'] + rows_train

    target = target.iloc[1:1733]
    target['index'] = target['index'] - 0
    
    # Darstellung des Realverlaufs, Trainings- und Validierungsverläufe
    for (columnName, ColumnData) in target.iteritems():
            if columnName != 'index':
                fig = plt.figure(figsize=(33.59*cm, 16.59*cm), dpi=300)
                ax1 = fig.add_subplot(111)
                
                for axis in ['top','bottom','left','right']:
                    ax1.spines[axis].set_linewidth(2)

                plt.plot(target['index'], target[columnName], 'o',
                         ms=1, c='k', alpha=1, linewidth=3)  # plot of target
                plt.plot(train_pred['index'], train_pred[columnName],
                         'o', ms=1, c='tab:orange', alpha=1, linewidth=3)
                plt.plot(target_pred['index'], target_pred[columnName],
                         'o', ms=1, c='b', alpha=1, linewidth=3)  # plot of target_pred
                plt.grid(axis='both', linestyle='-', alpha=1.0, color='black')
    
                plt.xticks(font=fpath)
                plt.yticks(font=fpath)
                plt.xlabel('t in s', font=fpath)
                plt.ylabel(columnName+' '+'in mm', font=fpath)
                plt.tight_layout()
                
                ax1.set_xlim(xmin=0)
                ax1.set_ylim(ymin=30, ymax=115)
                
                plt.show()

                # Berechnung der Metriken MSE, MAPE, MAE
                if columnName != 'index':
                    # Trainingsberechnung
                    r2_score_model = sk.metrics.r2_score(target_train[columnName], train_pred[columnName])
                    mae_model, mse_model, mape_model = calcula-te_metrics(target_train[columnName], train_pred[columnName])
                    print(f"\nTRANING: Trainingsmetriken - {columnName}:\nMSE: {mse_model}\nR2 Score: {r2_score_model}\nMAE: {mae_model}\nMAPE: {mape_model}%")
                    
                    # Validierungsberechnung 
                    r2_score_pred = sk.metrics.r2_score(target_test[columnName], target_pred[columnName])
                    mae_pred, mse_pred, mape_pred = calcula-te_metrics(target_test[columnName], target_pred[columnName])
                    print(f"\nPREDICTION: Validierungsmetriken - {columnNa-me}:\nMSE: {mse_pred}\nR2 Score: {r2_score_pred}\nMAE: {ma-e_pred}\nMAPE: {mape_pred}%")


    # Berechnnung der durschnittlichen MAE, MSE, MAPE
    target_pred = model.predict(data_test_scaled.values)
    train_pred = model.predict(data_train_scaled.values)
    calculate_and_display_metrics(target_train, train_pred, target_test, tar-get_pred)

    # Darstellung Verlustverläufe
    plot_loss(epoch_count, training_loss, validation_loss, ylim=(0, 10) ,xlim=(0, 3133))

    # Darstellung Residuuen
    train_predictions = model.predict(data_train_scaled.values)
    test_predictions = model.predict(data_test_scaled.values)
    train_residuals = target_train.values - train_predictions
    test_residuals = target_test.values - test_predictions
    plot_residuals(train_predictions, train_residuals, 'Training', 'orange')
    plot_residuals(test_predictions, test_residuals, 'Validierung', 'blue')

main()
