# Import der Bibliotheken
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import paho.mqtt.client as mqtt

# Setzen der Verbindungsparameter
input_url = "http://192.168.51.52:8086" # noch vom Bitmotec-System
input_token = "SGVHL3FgBy0nXfe6uO9kkBN11ViguRRPR0A9ckyHBr0q-sQVDMTOIAXH_7UH_zhZtcR20VX3qqxW3JfdpuYOaA=="
input_org = "bitmoteco"
input_bucket = "data"
client = InfluxDBClient(url=input_url, token=input_token, org=input_org)
query_api = client.query_api()

# Funktion zum Einlesen der Daten aus der InfluxDB
def fetch_data_from_influxdb():
    
    start = pd.Timestamp.utcnow() - pd.Timedelta(seconds=6)
    stop = pd.Timestamp.utcnow() - pd.Timedelta(seconds=5)
    
    # Transformation in das ISO-Zeitformat
    start_iso = start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    stop_iso = stop.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    # Definieren des Messkörpers und der InfluxQuery
    measurement = 'temperaturmessschwert'
    time_interval = "1s"
    flux_query = f'from(bucket: "{input_bucket}")     |> range(start: {start_iso}, stop: {stop_iso})     |> filter(fn: (r) => r["_measurement"] == "{measurement}")     |> aggregateWindow(every: {time_interval}, fn: mean, createEmpty: false)     |> yield(name: "mean")     |> pivot(rowKey:["_time"], columnKey: ["_field"], va-lueColumn: "_value")'
    
    list_df = query_api.query_data_frame(org=input_org, query=flux_query)
    print(list_df[0], list_df[1])
                                         
    return list_df[1].iloc[0:1,:] if list_df else None  
    

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
    
def main():
    
    # Einlesen der Daten aus der InfluxDB
    df = fetch_data_from_influxdb()
    df.to_csv('data_fetch_test.csv', index = False, header= True, sep=';', deci-mal='.')
    
    # Wertübergabe 
    payload = "10"
    print("Test - Payload")
    mqtt_publish(payload)

if __name__ == "__main__":
    main()
