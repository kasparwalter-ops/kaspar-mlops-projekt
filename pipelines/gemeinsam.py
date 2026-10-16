# 16.10.26 KW: Gemeinsame Bausteine der drei Pipelines (Konfiguration, Datenabruf, Hopsworks-Login)
import os

import pandas as pd
import requests
from dotenv import load_dotenv

# 16.10.26 KW: Standort Windisch (FHNW Campus)
ORT = "windisch"
BREITE = 47.4787
LAENGE = 8.2173

FEATURE_GROUP = "wetter_windisch"
FEATURE_VIEW = "regen_vorhersage"
MODELL_NAME = "regen_modell"
VERSION = 1

# 16.10.26 KW: Reihenfolge der Features muss in Training und Inferenz identisch sein
FEATURES = ["feuchte_mittel_24h", "bewoelkung"]
LABEL = "regen_in_2h"

API_URL = "https://api.open-meteo.com/v1/forecast"


def hopsworks_login():
    """Meldet sich mit API-Key und Projektname aus der Datei .env bei Hopsworks an."""
    import hopsworks

    load_dotenv()
    return hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT") or None,
    )


def lade_stundenwerte(tage: int = 92) -> pd.DataFrame:
    """Holt stündliche Wetterwerte der letzten Tage von Open-Meteo (kein API-Key nötig)."""
    antwort = requests.get(
        API_URL,
        params={
            "latitude": BREITE,
            "longitude": LAENGE,
            "hourly": "relative_humidity_2m,cloud_cover,precipitation",
            "past_days": tage,
            "forecast_days": 1,
            "timezone": "UTC",
        },
        timeout=30,
    )
    antwort.raise_for_status()
    df = pd.DataFrame(antwort.json()["hourly"])
    df["zeitpunkt"] = pd.to_datetime(df["time"], utc=True)
    # 16.10.26 KW: Die API liefert auch Prognosestunden, für die Features zählen nur vergangene Stunden
    jetzt = pd.Timestamp.now(tz="UTC").floor("h")
    # 16.10.26 KW: Für die ältesten Tage liefert die API teils leere Werte, diese Stunden verwerfen
    return df[df["zeitpunkt"] <= jetzt].drop(columns="time").dropna().reset_index(drop=True)


def lade_aktuelle_bewoelkung() -> float:
    """RT-Feature: aktueller Bewölkungsgrad in Prozent, erst zum Zeitpunkt der Inferenz bekannt."""
    antwort = requests.get(
        API_URL,
        params={"latitude": BREITE, "longitude": LAENGE, "current": "cloud_cover", "timezone": "UTC"},
        timeout=30,
    )
    antwort.raise_for_status()
    return float(antwort.json()["current"]["cloud_cover"])


def berechne_features(roh: pd.DataFrame) -> pd.DataFrame:
    """Aggregiertes Feature, RT-Feature (historischer Wert fürs Training) und Label pro Stunde."""
    df = roh.sort_values("zeitpunkt").reset_index(drop=True)
    # 16.10.26 KW: Aggregiertes Feature über 24 Timesteps (Rolling Window)
    df["feuchte_mittel_24h"] = df["relative_humidity_2m"].rolling(24, min_periods=24).mean()
    df["bewoelkung"] = df["cloud_cover"].astype(float)
    # 16.10.26 KW: Label = Regen (>= 0.1 mm) in einer der zwei folgenden Stunden
    regen_folgestunden = df["precipitation"].shift(-1) + df["precipitation"].shift(-2)
    df[LABEL] = (regen_folgestunden >= 0.1).astype(int)
    # 16.10.26 KW: Erste 23 Stunden ohne volles Fenster und letzte 2 Stunden ohne bekanntes Label weglassen
    df = df.iloc[:-2].dropna(subset=["feuchte_mittel_24h"])

    df["ort"] = ORT
    df["zeit_id"] = df["zeitpunkt"].astype("int64") // 10**9
    return df[["ort", "zeit_id", "zeitpunkt", *FEATURES, LABEL]].reset_index(drop=True)
