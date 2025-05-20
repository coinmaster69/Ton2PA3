# Ton2PA3 – Praxisaufgabe 3  
Analyse & automatisches Gain-Staging von Multitrack-Projekten

## Ordnerstruktur

```
Ton2PA3/
├─ Stems/           # hier liegen die 6 WAV-Tracks
├─ Output/          # enthält skalierten Mix
├─ Pegeldaten/      # enthält Pegel-Report
├─ main.py          # Analyse- & Pegel-Tool
└─ pyproject.toml   # Abhängigkeiten (poetry)
```

## Installation
Dieses Projekt verwendet [poetry](https://python-poetry.org/) für das Paketmanagement.

1. Dependencies installieren:
   ```bash
   poetry install --no-root
   ```
2. Anwendung starten:
   ```bash
   poetry run python main.py
   ```
3. Neue Dependencies hinzufügen:
   ```bash
   poetry add <library>
   ```

## Verwendung

```bash
python main.py Stems \
              --sr 48000 \
              --target -18.0 \
              --pair 2 5
```

| Option        | Beschreibung                                          | Standard |
|---------------|-------------------------------------------------------|----------|
| `project_dir` | Ordner mit den WAV-Stems                              | Pflicht  |
| `--sr`        | Ziel-Samplerate in Hz (wird bei Bedarf resampelt)     | 48000    |
| `--target`    | Ziel-RMS-Pegel der Summenspur in dBFS                 | -20.0    |
| `--pair i j`  | Index-Paar zweier Stems für Korrelationsanalyse       | 0 1      |

### Ausgabe

| Pfad          | Inhalt                                                          |
|---------------|-----------------------------------------------------------------|
| `Output/`     | `mix_sum.wav` + alle `*_scaled.wav` (skalierten Einzelspuren)   |
| `Pegeldaten/` | `pegel_report.txt` mit Energie-, RMS-, Crest- & Gain-Werten     |




