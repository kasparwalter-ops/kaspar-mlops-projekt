# Projektarbeit MLOps: Regen-Vorhersage Windisch

**Autor:** Kaspar Walter (KW) · CAS AI Operations, FHNW HS 2026 · Modul ML Operations · Abgabe 15.11.2026

*Muss ich einen Schirm einpacken?* Das Projekt sagt voraus, ob es in Windisch in den nächsten zwei Stunden regnet. Ziel ist nicht ein gutes Modell, sondern eine saubere **FTI-Architektur** (Feature, Training, Inference) mit dem **Hopsworks Feature Store**.

> Hinweis zur Umsetzung: Wie vom Dozenten erlaubt, habe ich mir bei der Umsetzung von KI helfen lassen. Konzept, Prüfung und Verantwortung liegen bei mir.

## Überblick

```
 Open-Meteo API ──► Feature-Pipeline ──► Feature Group "wetter_windisch" (Hopsworks)
                                                   │
                                                   ▼
                                         Feature View "regen_vorhersage"
                                            │                     │
                                            ▼                     ▼
                                  Training-Pipeline       Inference-Pipeline ◄── Open-Meteo live
                                            │              (neueste Features      (Bewölkung jetzt)
                                            ▼               + Live-Wert)
                                  Model Registry ─────────────────►│
                                  "regen_modell"                   ▼
                                                     "Regenwahrscheinlichkeit 2h: 73 %"
```

Die drei Pipelines sind unabhängige Skripte. Sie kommunizieren nur über den Feature Store und die Model Registry.

## Daten

- **Quelle:** [Open-Meteo Forecast API](https://open-meteo.com/), frei und ohne API-Key. Standort: Windisch (47.48 N, 8.22 E).
- **Rohdaten:** stündliche relative Luftfeuchtigkeit (%), Bewölkung (%) und Niederschlag (mm) der letzten 92 Tage (Maximum der API, `past_days=92`). Für die ältesten rund 30 Tage liefert die API leere Werte. Diese Stunden werden verworfen, nutzbar sind etwa 60 Tage (ca. 1400 Zeilen, davon rund 10 % mit Regen).
- **Historisch vs. aktuell:** Die vergangenen Stunden sind die historischen Daten fürs Training. Für die Inferenz holt die Inference-Pipeline den aktuellen Bewölkungswert live (`current=cloud_cover`). Wird die Feature-Pipeline regelmässig ausgeführt, kommen neue Stunden dazu.

## Target und Features

| Name | Art | Berechnung |
|---|---|---|
| `regen_in_2h` | **Label** (0/1) | 1, wenn in den zwei folgenden Stunden zusammen ≥ 0.1 mm Niederschlag fällt |
| `feuchte_mittel_24h` | **Aggregiertes Feature** (Batch) | Rolling Mean der Luftfeuchtigkeit über die letzten 24 Stunden (24 Timesteps) |
| `bewoelkung` | **RT-Feature** | Bewölkungsgrad. Im Training der historische Stundenwert, bei der Inferenz der Live-Wert der API |

Die Feature Group `wetter_windisch` (Version 1) hat den Primary Key `ort` + `zeit_id` (Unix-Zeit der Stunde) und die Event Time `zeitpunkt`.

## Modell

Logistische Regression (scikit-learn) mit `class_weight="balanced"`, da Regenstunden deutlich seltener sind als trockene. Split 80/20 lokal mit `train_test_split`, Metriken Accuracy und F1. Die Modellgüte war laut Aufgabe nicht Ziel. Das Modell wird lokal mit `joblib` gespeichert (`modell/modell.pkl`) und in die **Hopsworks Model Registry** hochgeladen (`regen_modell`, mit den Metriken).

## Aufbau der Pipelines

Code in `pipelines/`, gemeinsame Bausteine (Konfiguration, API-Abruf, Feature-Berechnung, Login) in `gemeinsam.py`.

**1. Feature-Pipeline** (`feature_pipeline.py`)
1. Rohdaten der letzten 92 Tage von Open-Meteo abrufen, nur vergangene Stunden behalten
2. Features berechnen (Rolling Window 24 h) und Label aus den zwei Folgestunden bilden
3. Zeilen ohne volles 24-h-Fenster (Anfang) und ohne bekanntes Label (letzte 2 Stunden) weglassen
4. Feature Group mit Primary Key und Event Time erstellen bzw. holen
5. Dataframe einfügen. Gleiche Primary Keys werden aktualisiert, mehrfaches Ausführen erzeugt keine Duplikate

**2. Training-Pipeline** (`training_pipeline.py`)
1. Feature Group laden
2. Feature View `regen_vorhersage` mit Query auf die zwei Features und das Label erstellen
3. Training Dataset aus der Feature View lesen (`training_data`)
4. Train/Test-Split, Modell trainieren, Accuracy und F1 berechnen
5. Modell lokal speichern, dann in die Model Registry hochladen

**3. Inference-Pipeline** (`inference_pipeline.py`)
1. Live-Daten abrufen: aktuelle Bewölkung (RT-Feature)
2. Feature View holen und die neueste Zeile der letzten 2 Tage laden (aggregiertes Feature)
3. Bestes Modell (nach Accuracy) aus der Model Registry herunterladen
4. Prediction aus beiden Features, Ausgabe der Regenwahrscheinlichkeit

## Abhängigkeiten

Getestet am 17.10.2026 unter Windows 11. Linux/macOS bzw. GitHub Codespaces sollten ebenfalls funktionieren, sind aber nicht getestet.

**Werkzeuge und Dienste**

| Werkzeug / Dienst | Verwendete Version | Bemerkung |
|---|---|---|
| Python | 3.12.10 | Hopsworks verlangt < 3.14. Festgelegt in `.python-version` und `pyproject.toml` (`>=3.12,<3.13`) |
| uv | 0.12.18 | Paket- und Umgebungsverwaltung. Installiert Python 3.12 bei Bedarf selbst |
| Git | 2.49 | Zum Klonen |
| Hopsworks Serverless | Server 5.1 (Region eu-west) | Kostenloser Account mit Projekt und API-Key nötig |
| Open-Meteo Forecast API | v1 | Frei, kein Account und kein Key nötig |

**Python-Pakete** (direkt in `pyproject.toml`, exakte Versionen inkl. aller Unterabhängigkeiten in `uv.lock`)

| Paket | Version |
|---|---|
| `hopsworks[python]` | 5.1.2 (fixiert auf 5.1.*) |
| `pandas` | 2.3.3 |
| `scikit-learn` | 1.9.1 |
| `requests` | 2.34.2 |
| `python-dotenv` | 1.2.4 |
| `joblib` | 1.6.0 |

`uv sync` installiert genau die Versionen aus `uv.lock`, die Umgebung ist damit reproduzierbar.

## Installation und Ausführung

**Voraussetzungen:** uv und Git installiert, Hopsworks-Account mit Projekt und API-Key (siehe Abhängigkeiten).

```bash
git clone <dieses Repository>
cd kaspar-mlops-projekt
uv sync
cp .env.example .env      # danach HOPSWORKS_API_KEY und HOPSWORKS_PROJECT eintragen
```

Die Pipelines in dieser Reihenfolge ausführen:

```bash
uv run python pipelines/feature_pipeline.py
uv run python pipelines/training_pipeline.py
uv run python pipelines/inference_pipeline.py
```

Beispielausgabe der Inference-Pipeline (17.10.2026):

```
Luftfeuchtigkeit Mittel 24h (Feature Store, Stand 2026-10-17 08:00:00+00:00): 89.1 %
Bewölkung jetzt (live): 100 %
Modell: regen_modell v2
Regenwahrscheinlichkeit nächste 2 Stunden: 63%
Schirm einpacken!
```


## Reflexion und Limitationen

- **«Messwerte» sind Modellwerte:** Die vergangenen Stunden von Open-Meteo sind Wettermodell-Daten, keine Stationsmessungen.
- **Zufälliger statt zeitlicher Split:** `train_test_split` mischt die Stunden zufällig. Benachbarte Stunden sind sich sehr ähnlich, die Testmetriken sind deshalb zu optimistisch. Ein zeitlicher Split (Backtesting) wäre realistischer.
- **Wenig Daten:** Rund 60 nutzbare Tage aus einer Jahreszeit decken Saisonalität nicht ab. Längere Historie gäbe die Archive-API von Open-Meteo.

- **Keine Orchestrierung:** Die Pipelines werden manuell gestartet. Für den Betrieb fehlen Scheduler, Monitoring (Drift) und das Zurückspielen der Ground Truth in den Feature Store.
- **Modellauswahl:** Die Inferenz nimmt das Modell mit der besten Accuracy.

## Schwierigkeiten und Lösungen

Alle drei Pipelines laufen vollständig (getestet am 17.10.2026 unter Windows 11 gegen Hopsworks Serverless). Unterwegs gab es diese Hürden:

- **Client-Version:** Der aktuelle Client (5.2) warnte, dass der Server auf 5.1 läuft. Darum ist `hopsworks[python]==5.1.*` fixiert.
- **Windows und `/tmp`:** Der Hopsworks-Client legt Zertifikate fest unter `/tmp` ab. Unter Windows entspricht das `C:\tmp`, das standardmässig fehlt. `gemeinsam.py` legt den Ordner deshalb vor dem Login an.
- **Delta-Format unter Windows:** Hopsworks 5.1 legt neue Feature Groups standardmässig im Format DELTA an. Das braucht das Paket `deltalake`, das es nur für Linux und macOS gibt. Die Feature Group verwendet darum explizit `time_travel_format="HUDI"`. Geschrieben wird dann über einen Materialisierungs-Job in Hopsworks (dauert ca. 2 Minuten). Dank an Claude für diese Lösung! :-)
- **Präfix der Event Time:** `get_batch_data(event_time=True)` liefert die Spalte als `<projekt>_wetter_windisch_1_zeitpunkt`. Die Inference-Pipeline sucht sie darum über die Endung `zeitpunkt`.
- **Lokale Modelldatei:** `model.save()` verschiebt die Datei standardmässig in die Registry. Mit `keep_original_files=True` bleibt die lokale Kopie wie verlangt erhalten.
- **Open-Meteo 503:** Die API antwortete einmal  mit «Service Unavailable». Abrufe werden darum bis zu dreimal wiederholt, sollten sie fehlschlagen.
