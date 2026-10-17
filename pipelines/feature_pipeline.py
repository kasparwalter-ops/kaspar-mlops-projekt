# 16.10.26 KW: Feature-Pipeline. Rohdaten holen, Features berechnen, in die Hopsworks Feature Group schreiben.
# Aufruf: uv run python pipelines/feature_pipeline.py [--tage 92]
import argparse

from gemeinsam import FEATURE_GROUP, VERSION, berechne_features, hopsworks_login, lade_stundenwerte


def main():
    parser = argparse.ArgumentParser(description="Feature-Pipeline Regen-Vorhersage")
    parser.add_argument("--tage", type=int, default=92, help="Anzahl vergangener Tage (max. 92)")
    args = parser.parse_args()

    # 1. Rohdaten via API abrufen
    roh = lade_stundenwerte(args.tage)
    # 2./3. Features engineeren und Dataframe mit Features und Label erstellen
    features = berechne_features(roh)
    print(f"{len(features)} Zeilen berechnet, von {features['zeitpunkt'].min()} bis {features['zeitpunkt'].max()}")

    # 4. Feature Group mit Primary Key und Event Time erstellen (oder bestehende holen)
    fs = hopsworks_login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP,
        version=VERSION,
        description="Stündliche Wetter-Features Windisch für die Regen-Vorhersage (Kaspar Walter)",
        primary_key=["ort", "zeit_id"],
        event_time="zeitpunkt",
        # 17.10.26 KW: Standard DELTA braucht das Paket deltalake, das es für Windows nicht gibt.
        # HUDI schreibt über einen Materialisierungs-Job in Hopsworks und läuft auf jedem OS.
        time_travel_format="HUDI",
    )
    # 5. Dataframe einfügen. 16.10.26 KW: Standard ist Upsert, gleicher Primary Key = Update, darum keine Duplikate.
    # wait=True wartet auf den Materialisierungs-Job, damit die Training-Pipeline danach die Daten sieht.
    fg.insert(features, wait=True)
    print(f"Feature Group '{FEATURE_GROUP}' v{VERSION} aktualisiert.")


if __name__ == "__main__":
    main()
