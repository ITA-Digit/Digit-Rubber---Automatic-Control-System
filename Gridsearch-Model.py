# Import der Bibliotheken
import numpy as np
import joblib
import time
from tensorflow.keras.models import model_from_json

# Model- und Scaler-Files
MODEL_NAME = 'Model_1'
SCALER_FILE = f'{MODEL_NAME}_scaler.gz'
INCREMENT = 1.0  # Step increment for each input

# Aufruf des Data-Mining-Models
def load_model(json_file, weights_file):
    with open(json_file, 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_file)
    return model

# Definieren der Objektfunktion
def objective_function(inputs, model, target_index, target_value, scaler):
    
    # Vorhersage der Outputs auf Grundlage der skaldierten Inputs 
    predicted_outputs = model.predict(inputs.reshape(1, -1))
    # Berechnung der Abweichung zum Zielwert 
    return (predicted_outputs[0][target_index] - target_value) ** 2

# Gridsearch Algorithmus 
def optimize_inputs(model, scaler, target_index, target_value, bounds):
    min_error = float('inf')
    best_inputs = None
    
    # Erzeugen des Grids mit den spezifizierten Inkrementen 
    grid_x = np.arange(bounds[0][0], bounds[0][1] + INCREMENT, INCRE-MENT)
    grid_y = np.arange(bounds[1][0], bounds[1][1] + INCREMENT, INCRE-MENT)
    
    for x in grid_x:
        for y in grid_y:
            scaled_inputs = scaler.transform(np.array([[x, y]]))[0]
            error = objective_function(scaled_inputs, model, target_index, tar-get_value, scaler)
            if error < min_error:
                min_error = error
                best_inputs = [x, y]
    
    return best_inputs, min_error

def main():
    
    # Starten der Zeitberechnung 
    start_time = time.time()
    
    # Aufruf der Model und Scaler
    model = load_model(MODEL_NAME + '.json', MODEL_NAME + '.h5')
    scaler = joblib.load(SCALER_FILE)
    
    # Auswahl der zu berechnenden Variable 
    
    target_index = 3  # Index 3 entspricht der Variable T4
    target_value = 95 # Zielwert 

    # Definieren der Grid-Grenzen
    real_bounds = [(10, 30), (70, 90)]  # Input bounds

    # Optimieren des Inputs durch Gridsearch 
    optimized_inputs, min_error = optimize_inputs(model, scaler, target_index, target_value, real_bounds)

    # Validierung der Vorhersage / Berechnung der Abweichung 
    scaled_optimized_inputs = scaler.transform(np.array([optimized_inputs]))
    predicted_outputs = model.predict(scaled_optimized_inputs)

    # Beenden der Zeitberechnung 
    end_time = time.time()  

    # Ausgabe der Ergebnisse 
    print("Optimierte Eingangsparameter: ", optimized_inputs)
    print("MSE:", min_error)
    print("Resultierende Ausgangsgrößen:", predicted_outputs)
    print(f"'T 4' wird vorhergsagt folgenden Wert zu erreichen: {predic-ted_outputs[0][target_index]:.2f} °C")
    print("Gesamtzeit: {:.2f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
