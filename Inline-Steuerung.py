# Import der Bibliotheken
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import paho.mqtt.client as mqtt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
from keras.models import model_from_json
import joblib
import numpy as np
from termcolor import colored

# Setzen der Verbindungsparameter
pd.set_option('display.max_columns', None)
input_url = "http://192.168.51.52:8086" # Bitmotec-System
input_token = "SGVHL3FgBy0nXfe6uO9kkBN11ViguRRPR0A9ckyHBr0q-sQVDMTOIAXH_7UH_zhZtcR20VX3qqxW3JfdpuYOaA=="
input_org = "bitmoteco"
input_bucket = "data"
client = InfluxDBClient(url=input_url, token=input_token, org=input_org)
query_api = client.query_api()

# Funktion zum Einlesen der Daten aus der InfluxDB
def fetch_data_from_influxdb():
    
    start = pd.Timestamp.utcnow() - pd.Timedelta(seconds=10)
    stop = pd.Timestamp.utcnow() - pd.Timedelta(seconds=8)
    
    # Transformation in das ISO-Zeitformat
    start_iso = start.replace(microsecond=0).isoformat()
    stop_iso = stop.replace(microsecond=0).isoformat()

    # Definieren des Messkörpers und der InfluxQuery
    measurement = 'temperaturmessschwert'
    time_interval = "1s"
    flux_query = f'from(bucket: "{input_bucket}")     |> range(start: {start_iso}, stop: {stop_iso})     |> filter(fn: (r) => r["_measurement"] == "{measurement}")     |> aggregateWindow(every: {time_interval}, fn: mean, createEmpty: false)     |> yield(name: "mean")     |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    
    list_df = query_api.query_data_frame(org=input_org, query=flux_query)
    df1 = list_df[0].iloc[0:1,:]
    
    # Sonderbedingung zum Einlesen der Data, da diese Teilweise in unter-schiedlichen Formaten bzw. Dataframes übergeben werden 
    if len(df1.columns) == 14:
        return list_df[0].iloc[0:1,:] if list_df else None  #
    else:
        return list_df[1].iloc[0:1,:] if list_df else None  #
    

# Definieren der MQTT-Broker callback functions
def on_connect(client, userdata, flags, rc):
    client.subscribe("subscribe_topic_placeholder")

def on_message(client, userdata, msg):
    pass  

# Initialisieren des MQTT-Clients
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set("user", "user") 
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
ip_master = "192.168.51.52"
mqtt_client.connect(ip_master, 1883, 60)
mqtt_client.loop_start()

# Wertübergabe der Drehzahl durch die MQTT-Publish-Methode
def mqtt_publish(payload):
    mqtt_client.publish("external_target_rpm", payload)
    
    
# Userinterface der Steuerung
class DataFrameHeaderSelector:
    def __init__(self, root, dataframe):
       	
        #Erstellen des Benutzerfensters und Speichern der Eingaben
        self.root = root
        self.dataframe = dataframe
        self.selected_header = None
        self.tolerance_input_entries = {}  
        self.tolerance_values = [None, None, None, None]
        self.display_task = None 

        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.header_listbox = tk.Listbox(self.frame, selectmode=tk.SINGLE)
        self.header_listbox.grid(row=0, column=0, padx=5, sticky="nsew")

        for header in self.dataframe.columns:
            self.header_listbox.insert(tk.END, header)

        # Eingabe der Toleranzgrenzen
        self.tolerance_labels = ["Obere Toleranzgrenze", "Oberer Vorgabewert", "Unterer Vorgabewert", "Untere Toleranzgrenze"]
        for idx, label in enumerate(self.tolerance_labels):
            self.create_tolerance_entry(label, row=idx + 1, column=0)
		
        # Button zum Starten nach der Eingabe
        self.start_button = ttk.Button(self.frame, text="Start", com-mand=self.submit_selection)
        self.start_button.grid(row=len(self.tolerance_labels) + 4, column=0, padx=5, pady=10, sticky="nsew")

        self.result_text = tk.Text(self.frame, width=40, height=10, wrap=tk.WORD)
        self.result_text.grid(row=0, column=1, rowspan=len(self.tolerance_labels) + 2, padx=10, pady=10, sticky="nsew")
        
        # Toggle-Checkbox, um zwischen Zielwert und Toleranzgrenzen (2 Mo-di) entscheiden zu können - zu beginn nicht ausgewählt
        self.target_value_var = tk.BooleanVar(value=False)
        self.target_value_checkbox = ttk.Checkbutton(self.frame, text="Zielwert", variable=self.target_value_var, com-mand=self.toggle_target_value_fields)
        self.target_value_checkbox.grid(row=len(self.tolerance_labels) + 1, co-lumn=0, padx=5, pady=5, sticky="nsew")
 
        # Definieren des Zielwert-feld 
        self.target_value_entry = ttk.Entry(self.frame, state=tk.DISABLED)
        self.target_value_entry.grid(row=len(self.tolerance_labels) + 2, co-lumn=0, padx=5, pady=5, sticky="nsew")

        # Zeit initialisierung für die Wartezeit der Steuerung nach einem Eingriff
        self.last_intervention_time = 0

    # Toggle-Checkbox Zielwert 
    def toggle_target_value_fields(self):
        
        # Aktivieren bzw. Deaktivieren der Zielwert-Funktion
        if self.target_value_var.get():
            self.target_value_entry.config(state=tk.NORMAL)
            
            # Dekaktivieren der Toleranzwert-Funktion, wenn Zielwert-Funktion ausgewählt ist 
            self.toggle_tolerance_fields(tk.DISABLED)
        else:
            self.target_value_entry.config(state=tk.DISABLED)
            #Aktivieren der Toleranzwert-Funktion, wenn Zielwert-Funktion nicht ausgewählt ist 
            self.toggle_tolerance_fields(tk.NORMAL)
    
    # Toggle-Checkbox Toleranzgrenzen 
    def toggle_tolerance_fields(self, state):
        # Aktivieren bzw. Deaktivieren der Toleranzgrenzen-Funktion
        for entry in self.tolerance_input_entries.values():
            entry.config(state=state)
            
    # Eingabe der Toleranzgrenzen 
    def create_tolerance_entry(self, label, row, column):
        entry_frame = ttk.Frame(self.frame)
        entry_frame.grid(row=row, column=column, padx=5, pady=5, sti-cky="nsew")

        label_widget = ttk.Label(entry_frame, text=label)
        label_widget.pack()

        entry = ttk.Entry(entry_frame)
        entry.pack()

        self.tolerance_input_entries[label] = entry
        

    #Submit-Funktion zu Parameterübergabe 
    def submit_selection(self):
        selected_index = self.header_listbox.curselection()
        if len(selected_index) == 1:
            self.selected_header = self.header_listbox.get(selected_index[0])
            # Überprüfung, ob Zielwert-Funktion aktiv ist 
            if self.target_value_var.get():
                # Abrufen und Verarbeiten des Zielwerts
                target_value = self.target_value_entry.get()
                try:
                    target_value = float(target_value)   # Umwandlung in Float, falls notwendig 
                    console_message = f"Zielwert: {target_value}"
                    print(console_message)
                    self.steuerung(target_value)
                    
                # Fehlermeldung, wenn keine Zahl für Zielwert gewählt wurde
                except ValueError:  
                    messagebox.showerror("Error", "Invalid input for Zielwert. Plea-se enter a numeric value.")
                    return
            else:
                # Wenn das die Zielwert-Funktion nicht aktiviert ist, werden die ei-gegebenen Toleranzgrenzen übernommen
                self.get_tolerance_values()
                self.display_result()
                self.stop_display_rows()
                self.display_rows()
        
        # Hinweismeldung, wenn keine Steuergröße ausgewählt wurde 
        else:
            messagebox.showerror("Error", "Please select one header.")
        
    # Übernahme der Toleranzgrenzen 
    def get_tolerance_values(self):
        tolerance_values = []
        for label, entry in self.tolerance_input_entries.items():
            value = entry.get()
            if value.strip() == "":
                messagebox.showerror("Error", f"Please enter a value for {label}.")
                return
            try:
                value = float(value)
                tolerance_values.append(value)
            except ValueError:
                messagebox.showerror("Error", f"Invalid input for {label}. Please enter a numeric value.")
                return

        # Eingabenresitriktion
        if not tolerance_values[0] >= tolerance_values[1] >= tolerance_values[2] >= tolerance_values[3]:
            messagebox.showerror("Error", "The tolerance values must follow the condition: Obere Toleranzgrenze > Oberer Vorgabewert > Unterer Vorgabe-wert > Untere Toleranzgrenze.")
            return

        self.tolerance_values = tolerance_values
        
    # Ausgabe des Ergebnisses 
    def display_result(self):
        self.result_text.delete(1.0, tk.END)
        if self.selected_header is not None:
            self.result_text.insert(tk.END, f"Selected Header: {self.selected_header}\n")
            self.result_text.insert(tk.END, "Tolerance Values:\n")
            for label, value in zip(self.tolerance_labels, self.tolerance_values):
                self.result_text.insert(tk.END, f"{label}: {value}\n")
        else:
            messagebox.showerror("Error", "Please select one header.")

    # Anzeige der aktuellen Messwerte     
    def display_rows(self):
        if self.display_task is not None:
            self.root.after_cancel(self.display_task)

        self.result_text.delete(1.0, tk.END)  # Clear the existing text
        self.result_text.insert(tk.END, f"Selected Header: {self.selected_header}\n")
        self.result_text.insert(tk.END, "Tolerance Values:\n")
        for label, value in zip(self.tolerance_labels, self.tolerance_values):
            self.result_text.insert(tk.END, f"{label}: {value}\n")
        self.result_text.insert(tk.END, "\nCurrent Row:\n")
        
        row = fetch_data_from_influxdb()
        self.result_text.insert(tk.END, f"{row}\n")
        self.root.update()
        
        # Überprüfung der Toleranzgrenzen 
        self.check_tolerance_conditions(row)
        self.display_task = self.root.after(1000, lambda: self.display_rows())  # 1 Sekunden Delay 

#########hier 
        
    # Vergleich der IST-Werte mit den Toleranzgrenzen 
    def check_tolerance_conditions(self, row):
   
        upper_tolerance = self.tolerance_values[0]
        lower_tolerance = self.tolerance_values[3]
        
        print(row)
        row.to_csv('data_fetch_test.csv', index = False, header= True, sep=';', decimal='.')
    
        # Get the value of the selected header from the current row
        selected_header_value = row.loc[0,self.selected_header]
    
        # Überprüfen, ob der IST-Wert die obere oder untere Toleranzgrenze unterschreitet
        if selected_header_value > upper_tolerance:
            current_time = time.time()
            
            
            # Delay: Überprüfen, ob die Zeit seit dem letzten Eingriff mindestens x Sekunden beträgt, 
            # Dadurch werden Mehrfacheingriffe für die selbe Überschreitung vermeiden und die geänderten Stellgrößen können such auf das System aus-wirken     
            if current_time - self.last_intervention_time >= 30:
                self.last_intervention_time = current_time
                messagebox.showinfo("Toleranzen überschritten", "Toleranzen überschritten (Obere Toleranzgrenze).")
                vorgabewert = self.tolerance_values[1]
                self.steuerung(vorgabewert)
            else:
                return True
        elif selected_header_value < lower_tolerance:
            current_time = time.time()
        
            # Analoges Vorgehn für untere Toleranzgrenze 
            if current_time - self.last_intervention_time >=30:
                self.last_intervention_time = current_time
                messagebox.showinfo("Toleranzen überschritten", "Toleranzen überschritten (Untere Toleranzgrenze).")
                vorgabewert = self.tolerance_values[2]
                self.steuerung(vorgabewert)
            else:
                return True

    # Steuerungseingriff 
    def steuerung(self, vorgabewert):
        
            # Ausgabe, wenn Toleranzgrenze überschritten wird
            console_message = colored("Messgröße überschritten\nSteuerung greift ein", 'red')
            print(console_message)
            
            # Laden der Modelle 
            json_file = f"Modell_{self.selected_header}.json"
            h5_file = f"Modell_{self.selected_header}.h5"
            scaler_file = f"Modell_{self.selected_header}_scaler.gz"
            variables = load_neural_network(vorgabewert, json_file, h5_file, sca-ler_file, 'MeanSquaredError', 'Adam', 'mse')
            
            # Übergabe der Parameter an weiteres Modell 
            parameters = load_neural_network(variables, 'Mo-dell_Data_Mining.json', 'Modell_Data_Mining.h5', 'Mo-dell_Data_Mining_scaler.gz', 'MeanSquaredError', 'Adam', 'mse')
    
    
            # Ausgangsgrößen
            t1_c_name = "T 1 [C] (0,0 mm)"
            t2_c_name = "T 2 [C] (10,0 mm)"
            t3_c_name = "T 3 [C] (20,0 mm)"
            t4_c_name = "T 4 [C] (30,0 mm)"
            t5_c_name = "T 5 [C] (40,0 mm)"
            t6_c_name = "T 6 [C] (50,0 mm)"
            t7_c_name = "T 7 [C] (60,0 mm)"
            t8_c_name = "T 8 [C]"
            p_wz_name = "p Wz"
            p_b_name = "p b"
            p_a_name = "p a"
            t_b_name = "T b"
            t_a_name = "T a"

    
        		# Schleifenparameter
            header_index = self.dataframe.columns.get_loc(self.selected_header)  
            anad = 10 # Abbruch nach dem die schleife nach 10 Iterationen 
            step_size = 1
            deviation = 1
            i = 1
            new_input = vorgabewert
            
            # Extrahieren der Eingangsgrößen 
            drehzahl_value, t_schneke_value = variables
            drehzahl_name = "Drehzahl"
            t_schneke_name = "T (Schneke)"
            console_message = colored("Einzustellende Stellgrößen", 'red')
            print(console_message)
            print(f"{drehzahl_name}: {drehzahl_value}\n{t_schneke_name}: {t_schneke_value}")
            console_message = colored("Erreichte Messgrößen", 'red')
            print(console_message)
            print(f"{t1_c_name}: {parameters[0]}\n{t2_c_name}: {parame-ters[1]}\n{t3_c_name}: {parameters[2]}\n{t4_c_name}: {parame-ters[3]}\n{t5_c_name}: {parameters[4]}\n{t6_c_name}: {parame-ters[5]}\n{t7_c_name}: {parameters[6]}\n{t8_c_name}: {parame-ters[7]}\n{p_wz_name}: {parameters[8]}\n{p_b_name}: {parame-ters[9]}\n{p_a_name}: {parameters[10]}\n{t_b_name}: {parame-ters[11]}\n{t_a_name}: {parameters[12]}")
            
            # Schleifeniteration
            while abs(vorgabewert - parameters[header_index-6]) > deviation:
                new_input = new_input + step_size * (vorgabewert-parameters[header_index-6])
                print("schleifen anppasungen:", i)
                print("neuer input für das erste netz:", new_input)
                variables = load_neural_network(new_input, json_file, h5_file, sca-ler_file, 'MeanSquaredError', 'Adam', 'mse')
                drehzahl_value, t_schneke_value = variables
                parameters = load_neural_network(variables, 'Mo-dell_Data_Mining.json', 'Modell_Data_Mining.h5', 'Mo-dell_Data_Mining_scaler.gz', 'MeanSquaredError', 'Adam', 'mse')
                console_message = colored("Einzustellende Stellgrößen", 'red')
                print(console_message)
                print(f"{drehzahl_name}: {drehzahl_value}\n{t_schneke_name}: {t_schneke_value}")
                console_message = colored("Erreichte Messgrößen", 'red')
                print(console_message)
                print(f"{t1_c_name}: {parameters[0]}\n{t2_c_name}: {parame-ters[1]}\n{t3_c_name}: {parameters[2]}\n{t4_c_name}: {parame-ters[3]}\n{t5_c_name}: {parameters[4]}\n{t6_c_name}: {parame-ters[5]}\n{t7_c_name}: {parameters[6]}\n{t8_c_name}: {parame-ters[7]}\n{p_wz_name}: {parameters[8]}\n{p_b_name}: {parame-ters[9]}\n{p_a_name}: {parameters[10]}\n{t_b_name}: {parame-ters[11]}\n{t_a_name}: {parameters[12]}")
                # Ausgabe, dass die Iteration fehlgeschlagen ist, wenn Schleifen-anzahl > max. Iteration
                if i > anad:
                    messagebox.showerror("Error", "Messgröße kann nicht erreicht werden.")             
                    return True
                # Erhöhung des Iterationsindexes 
                i = i+1
          

            # Ergebnisanzeige als Pop-Up: Einzustellende Eingangsgrößen und erreichte Messgrößen
            messagebox.showinfo("Einzustellende Stellgrößen", f"Einzustellende Stellgrößen:\n{drehzahl_name}: {drehzahl_value:.2f}\n{t_schneke_name}: {t_schneke_value:.2f}\nT (Stiftzone) : {t_schneke_value:.2f}\nT (Werkzeug) : {t_schneke_value:.2f}\nT (Speisewalze) : {50}\nT (Einzug) : {70}")
            messagebox.showinfo("Erreichte Messgrößen", f"{t1_c_name}: {pa-rameters[0]:.2f}\n{t2_c_name}: {parameters[1]:.2f}\n{t3_c_name}: {parame-ters[2]:.2f}\n{t4_c_name}: {parameters[3]:.2f}\n{t5_c_name}: {parame-ters[4]:.2f}\n{t6_c_name}: {parameters[5]:.2f}\n{t7_c_name}: {parame-ters[6]:.2f}\n{t8_c_name}: {parameters[7]:.2f}\n{p_wz_name}: {parame-ters[8]:.2f}\n{p_b_name}: {parameters[9]:.2f}\n{p_a_name}: {parame-ters[10]:.2f}\n{t_b_name}: {parameters[11]:.2f}\n{t_a_name}: {parame-ters[12]:.2f}")

            # Überprüfung, ob die Abweichungsanforderung erfüllt wird 
            if abs(vorgabewert - parameters[header_index-6]) > deviation:
                messagebox.showerror("Error", "Messgröße kann nicht erreicht werden.")
                
            console_message = colored("Einzustellende Stellgrößen", 'red')
            print(console_message)
            print(f"{drehzahl_name}: {drehzahl_value}\n{t_schneke_name}: {t_schneke_value}")
            console_message = colored("Erreichte Messgrößen", 'red')
            print(console_message)
            print(f"{t1_c_name}: {parameters[0]}\n{t2_c_name}: {parame-ters[1]}\n{t3_c_name}: {parameters[2]}\n{t4_c_name}: {parame-ters[3]}\n{t5_c_name}: {parameters[4]}\n{t6_c_name}: {parame-ters[5]}\n{t7_c_name}: {parameters[6]}\n{t8_c_name}: {parame-ters[7]}\n{p_wz_name}: {parameters[8]}\n{p_b_name}: {parame-ters[9]}\n{p_a_name}: {parameters[10]}\n{t_b_name}: {parame-ters[11]}\n{t_a_name}: {parameters[12]}")
            
            # Wertübertragung mittels MQTT broker
            mqtt_publish(int(drehzahl_value))
            
            current_time = time.time()
            self.last_intervention_time = current_time

    # Erneutes Ausführen der Steuerung         
    def steuerung_erneut(self, vorgabewert):
        self.steuerung(vorgabewert)

    # Ausgabe zum erneuten Ausführen 
    def enable_control_system(self):
        messagebox.showinfo("Steuerungssystem aktiviert", "Die Steuerung ist nun für die nächste Verwendung aktiviert.")

    # Stoppen der Messwertanzeige 
    def stop_display_rows(self):
        if self.display_task is not None:
            self.root.after_cancel(self.display_task)
            self.display_task = None

# Laden der Modelle 
def load_neural_network(input_data, json, h5, scaler, _loss, _optimizer, _metrics):
    file = open(json, 'r')
    loaded = file.read()
    file.close()

    loaded_model = model_from_json(loaded)
    loaded_model.load_weights(h5)

    loaded_model.compile(loss=_loss, optimizer=_optimizer, met-rics=[_metrics])

    scaler = joblib.load(scaler)
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

    pred = loaded_model.predict(input_data_scaled)

    control_parameters = pred[0]

    return control_parameters

# Main-Funktion
def main():
    
    # Header 
    root = tk.Tk()
    root.title("Zu steuernde Messgröße")
    df = fetch_data_from_influxdb()
    app = DataFrameHeaderSelector(root, df)

    # Stopp-Button 
    stop_button = ttk.Button(root, text="Stop", com-mand=app.stop_display_rows)
    stop_button.grid(row=len(app.tolerance_labels) + 4, column=0, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
