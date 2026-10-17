# Projektarbeit MLOps: Regen-Vorhersage Windisch

**Autor:** Kaspar Walter (KW) · CAS AI Operations, FHNW HS 2026 · Modul ML Operations · Abgabe 15.11.2026

*Muss ich einen Schirm einpacken?* Das Projekt sagt voraus, ob es in Windisch in den nächsten zwei Stunden regnet. Ziel ist nicht ein gutes Modell, sondern eine saubere **FTI-Architektur** (Feature, Training, Inference) mit dem **Hopsworks Feature Store**.

> Hinweis zur Umsetzung: Wie vom Dozenten erlaubt, habe ich die Arbeit mit Unterstützung von Claude (Anthropic) umgesetzt. Konzept, Prüfung und Verantwortung liegen bei mir.

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

Die drei Pipelines sind unabhängige Skripte. Sie kommunizieren nur über den Feature Store und die Model Registry, nie direkt miteinander.

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

## Installation und Ausführung

**Voraussetzungen:** Python 3.12 (Hopsworks verlangt < 3.14), [uv](https://docs.astral.sh/uv/), ein kostenloser Account auf [hopsworks.ai](https://www.hopsworks.ai/) mit einem Projekt und einem API-Key. Getestet unter Windows 11, läuft auch unter Linux/macOS bzw. in GitHub Codespaces. Alle Abhängigkeiten sind in `pyproject.toml` definiert und in `uv.lock` fixiert.

```bash
git clone <dieses Repository>
cd kaspar-mlops-projekt
uv sync
cp .env.example .env      # danach HOPSWORKS_API_KEY und HOPSWORKS_PROJECT eintragen
```

Die Pipelines in dieser Reihenfolge ausführen:

```bash
uv run python pipelines/feature_pipeline.py     # dauert einige Minuten (Materialisierungs-Job in Hopsworks)
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

Für aktuelle Features vor jeder Inferenz die Feature-Pipeline erneut ausführen, im Betrieb z. B. stündlich per Scheduler. Die Training-Pipeline läuft nur bei Bedarf (Re-Training).

## Reflexion und Limitationen

- **Alter des aggregierten Features:** Weil das Label die zwei Folgestunden braucht, fehlen die letzten zwei Stunden in der Feature Group. Das 24-h-Mittel bei der Inferenz ist darum rund 3 Stunden alt. Bei einem 24-h-Mittel ist der Effekt klein. Sauberer wäre eine Trennung von Features und Labels in zwei Feature Groups (Spine Group, Ausbaustufe).
- **«Messwerte» sind Modellwerte:** Die vergangenen Stunden von Open-Meteo sind Wettermodell-Daten, keine Stationsmessungen. Für den Zweck der Arbeit genügt das.
- **Zufälliger statt zeitlicher Split:** `train_test_split` mischt die Stunden zufällig. Benachbarte Stunden sind sich sehr ähnlich, die Testmetriken sind deshalb zu optimistisch. Ein zeitlicher Split (Backtesting) wäre realistischer.
- **Wenig Daten:** Rund 60 nutzbare Tage aus einer Jahreszeit decken Saisonalität nicht ab. Längere Historie gäbe die Archive-API von Open-Meteo.
- **Schwaches Modell:** Zwei Features erklären Regen nur grob. Luftdruck oder Niederschlag der Vorstunden wären stärker, waren aber bewusst nicht Ziel.
- **Keine Orchestrierung:** Die Pipelines werden manuell gestartet. Für den Betrieb fehlen Scheduler, Monitoring (Drift) und das Zurückspielen der Ground Truth in den Feature Store.
- **Modellauswahl:** Die Inferenz nimmt das Modell mit der besten Accuracy. Im Betrieb würde man eine Version explizit freigeben (z. B. per Tag oder Alias).

## Schwierigkeiten und Lösungen

Alle drei Pipelines laufen vollständig (getestet am 17.10.2026 unter Windows 11 gegen Hopsworks Serverless). Unterwegs gab es diese Hürden:

- **Client-Version:** Der aktuelle Client (5.2) warnte, dass der Server auf 5.1 läuft. Darum ist `hopsworks[python]==5.1.*` fixiert.
- **Windows und `/tmp`:** Der Hopsworks-Client legt Zertifikate fest unter `/tmp` ab. Unter Windows entspricht das `C:\tmp`, das standardmässig fehlt. `gemeinsam.py` legt den Ordner deshalb vor dem Login an (unter Linux/macOS ohne Wirkung).
- **Delta-Format unter Windows:** Hopsworks 5.1 legt neue Feature Groups standardmässig im Format DELTA an. Das braucht das Paket `deltalake`, das es nur für Linux und macOS gibt. Die Feature Group verwendet darum explizit `time_travel_format="HUDI"`. Geschrieben wird dann über einen Materialisierungs-Job in Hopsworks (dauert ca. 2 Minuten).
- **Präfix der Event Time:** `get_batch_data(event_time=True)` liefert die Spalte als `<projekt>_wetter_windisch_1_zeitpunkt`. Die Inference-Pipeline sucht sie darum über die Endung `zeitpunkt`.
- **Lokale Modelldatei:** `model.save()` verschiebt die Datei standardmässig in die Registry. Mit `keep_original_files=True` bleibt die lokale Kopie wie verlangt erhalten.
- **Open-Meteo 503:** Die API antwortete einmal kurz mit «Service Unavailable». Abrufe werden darum bis zu dreimal wiederholt.
