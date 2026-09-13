import pandas as pd
import torch

from utils.preprocess import preprocess_tabular
from models.pytorch_models import get_model


# Cargar datos de Wisconsin para predecir
df = pd.read_csv(
    "datasets/wisconsin/breast_cancer_pred.csv"
)

# Preprocesar usando el scaler guardado
X, _ = preprocess_tabular(df)

# Cargar el modelo entrenado
model = get_model("tabular")

# Convertir los datos a tensor
x = torch.tensor(X, dtype=torch.float32)

# Modo evaluación
model.eval()

# Hacer predicciones
with torch.no_grad():
    preds = model(x)
    classes = torch.argmax(preds, dim=1)

print("Predicciones:")
print(classes.tolist())