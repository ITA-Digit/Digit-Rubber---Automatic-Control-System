# Machine control using ANN to minimize the temperature deviation of rubber extrusion systems in the screw channel

Zielsetzung dieses Repository ist es, die Codeprojekte aus der Dissertation "Maschinensteuerung durch Künstliche Neuronale Netze zur Minimierung der Temperaturabweichung von Kautschukextrusionsanlagen im Schneckenkanal" zusammenzufassen. Die Codeprojekte umfassen Datenanalysen, Machine-Learning-Modelle, Steuerungssysteme und funktionelle Validierungen. Es folgt eine Referenzierung der Codeprojekte mit einer kurzen Erläuterung. Eine detallierte Erläuterung kann aus der Kommentierung der Codeprojekt-Files entnommen werden. 

## Pearson-Korrelationsmatrix 
- "Pearson-Analyse.py"
- Applikation auf den Extrusions- und Mischerdatensatz, um die Pearson-Koeffizienten für die jeweiligen Ein- und Ausgangsgrößen zu berechnen.
- Ein Pearson-Korrelationswert, der sich in der Nähe von 1 befindet, bildet in Abhängigkeit des Vorzeichens eine starke lineare Korrelation ab, wobei ein Wert um 0 keine oder eine schwache Korrelation aufzeigt.

## Feature Selection
- "Feature-Importance.py"
- Applikation auf den Extrusions- und Mischerdatensatz, um die Feature-Importance-Scores für die jeweiligen Eingangsgrößen zu berechnen.
- Einsatz eines Regressor Random Forest mit einer Estimatoranzahl von 2.000.
- Die Summe aller Feature Importance-Scores entspricht dabei immer einem Wert von 1 bzw. 100 %.

## Initale Prädiktionsgenauigkeit unoptimierter Modellarten: LIN, FNN, RNN, LSTM 
- "Modelcompare.py"
- Applikation auf den Extrusionsdatensatz, um die unoptimierten Modellarten Lineare-Regression, FNN, RNN und LSTM anhand des MSE's zu vergleichen.
- Die Datenvorverarbeitung erfolgt mittels des MinMaxScalers. 
- Zielsetzung dabei ist es, lediglich eine ersten Gegenüberstellung der Modellarten im Bezug auf den bestehenden Extrusionsdatensatz durchzuführen. 

## Versuchsübergreifendes Data-Mining-Modell zur Prädiktion der Extrusionsausgangsgrößen 
- "Data-Mining-Extrusion.py"
- Traning des Data-Mining-Modell auf Grundlage des Extrusionsdatensatzes, um die Extrusionsausgangsgrößen auf Basis der korrespondierenden Eingangsgrößen hervorzusagen.
- Export des Modells als *.h5, *.json und *_scaler.gz zur Implementierung in der Steuerung.
- Darstellung der Verlustverläufe und Residualdiagramme.
- Berechnung der Prädiktionsmetriken MSE, MAPE und MAE.

## Versuchsübergreifendes Data-Mining-Modell zur Prädiktion der Mischerausgangsgrößen
- "Data-Mining-Mischer.py"
- Traning des Data-Mining-Modell auf Grundlage des Mischerdatensatzes, um die Mischerausgangsgrößen auf Basis der korrespondierenden Eingangsgrößen hervorzusagen.
- Export des Modells als *.h5, *.json und *_scaler.gz zur Implementierung in der Steuerung.
- Darstellung der Verlustverläufe und Residualdiagramme.
- Berechnung der Prädiktionsmetriken MSE, MAPE und MAE.

## Inverses-Modell zur Prädiktion der Extrusionseingangsgrößen auf Grundlage aller Ausgangsgrößen
- "Invers-Model-Extrusion.py"
- Traning des Inversen-Modell auf Grundlage des Extrusionsdatensatzes, um die Extrusionseingangsgrößen auf Basis der korrespondierenden Ausgangsgrößen hervorzusagen.
- Export des Modells als *.h5, *.json und *_scaler.gz zur Implementierung in der Steuerung.
- Darstellung der Verlustverläufe und Residualdiagramme.
- Berechnung der Prädiktionsmetriken MSE, MAPE und MAE.

## Inverses-Zusatz-Modell zur Prädiktion der Extrusionsausgangsgrößen auf Grundlage der Ausgangsgröße (am Beipsiel von T4)
- "Invers-Zusatz-Model-Extrusion.py"
- Traning des Inversen-Zusatz-Modells auf Grundlage des Extrusionsdatensatzes, um die Extrusionsausgangsgrößen auf Basis einer korrespondierenden Ausgangsgrößen hervorzusagen.
- Dient zur Berechnung der Eingangsgrößen für das Invers-Model-Extrusion.py-Modell. 
- Export des Modells als *.h5, *.json und *_scaler.gz zur Implementierung in der Steuerung.
- Darstellung der Verlustverläufe und Residualdiagramme.
- Berechnung der Prädiktionsmetriken MSE, MAPE und MAE.

## Inverses-Modell zur Prädiktion der Extrusionseingangsgrößen auf Grundlage einer Ausgangsgröße (am Beipsiel von T4) 
- "Invers-Model-Extrusion-1-Input.py"
- Traning des Inversen-Modell auf Grundlage des Extrusionsdatensatzes, um die Extrusionseingangsgrößen auf Basis einer Ausgangsgröße hervorzusagen.
- Für dieses Modell ist innerhalb der Steuerung kein Invers-Zusatz-Model-Extrusion.py.Modell erforderlich. 
- Export des Modells als *.h5, *.json und *_scaler.gz zur Implementierung in der Steuerung.
- Darstellung der Verlustverläufe und Residualdiagramme.
- Berechnung der Prädiktionsmetriken MSE, MAPE und MAE.

## Gridsearch-Modell 
- "Gridsearch-Model.py"
- Ausführung des Gridsearch-Modells auf Grundlage des Extrusionsdatensatzes, um die Extrusionseingangsgrößen auf Basis einer Ausgangsgröße hervorzusagen.
- Kombination aller Möglichen Parametervarationen in dem definierten Wertebereich.
- Berechnung der Prädiktionsmetriken MSE und Ausgabe der Berechnungszeit.

## Inline-Steuerung 
- "Inline-Steuerung.py"
- Steuerungsalgorithmus, um den Extursionsprozess in user-definierten Toleranzgrenzen einzuhalten.
- GUI zur Eingabe der Toleranzgrenzen, Steuerungsparameter und Ausgabe der aktuell gemessenen Parameter. 
- Aufruf der Data-Mining- und Invers-Modelle zur Berechnung der erforderlichen Eingangsparameter.
- Übergabe der Steuerungsparameter mittels MQTT-Client.
- Für die funktionelle Validierung kann anstatt des LIVE-Einlesens der InlfuxDB der bestehende Extrusionsdatensatz im 1-Sekunden Takt eingelesen werden. 

## Verbindungstest der Inline-Steuerung 
- "Inline-Steuerung-Connectiontest.py"
- Verbindungstest des Steuerungsalgorithmus, um die Steuerung vor dem Einsatz in dem Realbetrieb zu testen. 
- Aufruf des MQTT-Clients und übergabe von Testparametern. 
  
