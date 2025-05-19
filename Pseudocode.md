# Pseudocode
1. **Projektordner öffnen**  
   *Enthält alle WAV‑Stems des Multitrack‑Projekts.*


2. **Alle WAV‑Dateien laden**  
   1. Falls nötig auf eine gemeinsame Samplerate umrechnen  
   2. Auf die kürzeste Spur­länge kürzen (alle gleich lang machen)


3. **Einzelspur‑Analyse** – für jede Spur  
   1. **Energie** berechnen (Summe der Quadrate)  
   2. **RMS**‑Wert bestimmen (Effektivwert)  
   3. *(optional)* **Scheitelfaktor** ermitteln (Peak / RMS)


4. **Korrelation** zweier ausgewählter Spuren bestimmen  
   *Korrelationskoeffizient berechnen.*


5. **Vorläufige Summierung**  
   1. Alle Spuren ohne Pegelung addieren  
   2. RMS der Summe messen


6. **Automatisches Gain‑Staging**  
   1. Ziel‑RMS für den Gesamt‑Mix festlegen  
   2. Globalen Gain‑Faktor errechnen (Leistungserhaltung)  
   3. Gain auf jede Spur anwenden  
   4. Verifizieren, dass die Summen‑RMS dem Ziel entspricht


7. **Export**  
   1. Skalierte Einzelspuren speichern  
   2. Gemischte Summenspur speichern


8. **Report erstellen**  
   *Energie‑, RMS‑, Crest‑ und Korrelationswerte sowie Gains protokollieren.*


9. **Fertigmeldung ausgeben**
