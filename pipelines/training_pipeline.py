# 16.10.26 KW: Training-Pipeline. Features aus dem Feature Store laden, Modell trainieren, in die Model Registry laden.
# Aufruf: uv run python pipelines/training_pipeline.py
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from gemeinsam import FEATURE_GROUP, FEATURE_VIEW, FEATURES, LABEL, MODELL_NAME, VERSION, hopsworks_login

MODELL_ORDNER = Path(__file__).resolve().parent.parent / "modell"


def main():
    projekt = hopsworks_login()
    fs = projekt.get_feature_store()

    # 1. Feature Group laden
    fg = fs.get_feature_group(FEATURE_GROUP, version=VERSION)
    # 2. Feature View mit Query auf Features + Label
    fv = fs.get_or_create_feature_view(
        name=FEATURE_VIEW,
        version=VERSION,
        description="Features und Label für die Regen-Vorhersage (Kaspar Walter)",
        query=fg.select([*FEATURES, LABEL]),
        labels=[LABEL],
    )
    # 3. Training Dataset aus der Feature View erstellen
    X, y = fv.training_data(description="Trainingsdaten Regen-Vorhersage")
    X = X[FEATURES]
    y = y[LABEL]
    print(f"{len(X)} Trainingszeilen, Anteil Regen: {y.mean():.1%}")

    # 4. Modell trainieren und evaluieren. 16.10.26 KW: class_weight, weil Regenstunden deutlich seltener sind
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    modell = LogisticRegression(class_weight="balanced")
    modell.fit(X_train, y_train)
    vorhersage = modell.predict(X_test)
    metriken = {
        "accuracy": float(accuracy_score(y_test, vorhersage)),
        "f1": float(f1_score(y_test, vorhersage)),
    }
    print(f"Metriken Testset: {metriken}")

    # 5. Modell lokal speichern, dann in die Hopsworks Model Registry hochladen
    MODELL_ORDNER.mkdir(exist_ok=True)
    joblib.dump(modell, MODELL_ORDNER / "modell.pkl")
    registry_modell = projekt.get_model_registry().sklearn.create_model(
        name=MODELL_NAME,
        metrics=metriken,
        description="Logistische Regression: Regen in den nächsten 2 Stunden in Windisch (Kaspar Walter)",
        input_example=X_test.head(1),
    )
    registry_modell.save(str(MODELL_ORDNER))
    print(f"Modell '{MODELL_NAME}' v{registry_modell.version} in der Model Registry gespeichert.")


if __name__ == "__main__":
    main()
