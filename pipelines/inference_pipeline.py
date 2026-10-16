# 16.10.26 KW: Inference-Pipeline. Neueste Features aus dem Feature Store + Live-Wert, Modell aus der Registry, Prediction.
# Aufruf: uv run python pipelines/inference_pipeline.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from gemeinsam import FEATURE_VIEW, FEATURES, MODELL_NAME, VERSION, hopsworks_login, lade_aktuelle_bewoelkung


def main():
    projekt = hopsworks_login()
    fs = projekt.get_feature_store()

    # 1. Live-Daten abrufen: RT-Feature, erst jetzt bekannt
    bewoelkung = lade_aktuelle_bewoelkung()

    # 2. Feature View holen und die neuesten Features laden (aggregiertes Feature)
    fv = fs.get_feature_view(FEATURE_VIEW, version=VERSION)
    seit = datetime.now(timezone.utc) - timedelta(days=2)
    batch = fv.get_batch_data(start_time=seit, event_time=True)
    neueste = batch.sort_values("zeitpunkt").iloc[-1]

    # 3. Modell aus der Model Registry herunterladen. 16.10.26 KW: bestes Modell nach Accuracy
    registry_modell = projekt.get_model_registry().get_best_model(MODELL_NAME, "accuracy", "max")
    modell = joblib.load(Path(registry_modell.download()) / "modell.pkl")

    # 4. Prediction mit gespeichertem aggregiertem Feature + Live-RT-Feature
    eingabe = pd.DataFrame([{"feuchte_mittel_24h": neueste["feuchte_mittel_24h"], "bewoelkung": bewoelkung}])[FEATURES]
    wahrscheinlichkeit = modell.predict_proba(eingabe)[0][1]

    print(f"Luftfeuchtigkeit Mittel 24h (Feature Store, Stand {neueste['zeitpunkt']}): {neueste['feuchte_mittel_24h']:.1f} %")
    print(f"Bewölkung jetzt (live): {bewoelkung:.0f} %")
    print(f"Modell: {MODELL_NAME} v{registry_modell.version}")
    print(f"Regenwahrscheinlichkeit nächste 2 Stunden: {wahrscheinlichkeit:.0%}")
    print("Schirm einpacken!" if wahrscheinlichkeit >= 0.5 else "Kein Schirm nötig.")


if __name__ == "__main__":
    main()
