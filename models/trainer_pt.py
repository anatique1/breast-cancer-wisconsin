import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .pytorch_arch import TabularNet
from .pytorch_models import invalidate_cache


def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    # NOTA: X_train / X_test ya vienen escalados desde utils/data.py
    # (prepare_tabular_data), que además guarda el scaler en
    # models/saved/scaler.pkl para usarlo en predicción. No volvemos a
    # escalar aquí para no desalinear entrenamiento y predicción.
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TabularNet(input_dim=X_train.shape[1], n_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    average_loss = 0
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        print(f"Época {ep + 1}/{epochs} - Loss: {average_loss:.4f}")

    # Evaluación
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        predictions = model(X_test_tensor)
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == y_test_tensor).float().mean().item()

    print(f"Precisión en prueba: {accuracy * 100:.2f}%")

    # Guardar modelo y métricas
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    save_path = Path("models/saved/pt_tabular.pt")
    torch.save(model.state_dict(), save_path)

    metrics = {"accuracy": float(accuracy), "loss": float(average_loss)}
    with open("models/saved/pt_metrics.json", "w") as f:
        json.dump(metrics, f)

    # Invalidar el modelo cacheado para que el próximo /predict/ recargue
    # este modelo recién entrenado en vez del anterior en memoria.
    invalidate_cache("tabular")

    return save_path, metrics