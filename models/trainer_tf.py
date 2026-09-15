import json
from pathlib import Path

import tensorflow as tf

from .tensorflow_arch import build_tabular_model
from .tensorflow_models import invalidate_cache


def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    # NOTA: X_train / X_test ya vienen escalados desde utils/data.py
    # (prepare_tabular_data), que además guarda el scaler en
    # models/saved/scaler.pkl para usarlo en predicción. No volvemos a
    # escalar aquí para no desalinear entrenamiento y predicción.
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

    # Invalidar el modelo cacheado para que el próximo /predict/ recargue
    # este modelo recién entrenado en vez del anterior en memoria.
    invalidate_cache("tabular")

    return save_dir, metrics