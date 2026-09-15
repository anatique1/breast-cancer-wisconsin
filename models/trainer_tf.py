import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .tensorflow_arch import build_tabular_model


def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    # Escalado de características para Wisconsin
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_tabular_model(input_dim=X_train.shape[1])

    model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)
    )

    # Evaluar para sacar métricas finales
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Guardar modelo y métricas
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    save_dir = Path("models/saved/tf_tabular.keras")
    model.save(save_dir)

    metrics = {"accuracy": float(accuracy), "loss": float(loss)}
    with open("models/saved/tf_metrics.json", "w") as f:
        json.dump(metrics, f)

    return save_dir, metrics