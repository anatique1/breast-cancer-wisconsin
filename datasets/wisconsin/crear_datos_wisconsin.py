from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Carpeta donde está este archivo
BASE_DIR = Path(__file__).resolve().parent

# 1. Cargar dataset Wisconsin
data = load_breast_cancer()

# 2. Crear DataFrame con las características
df_data = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# 3. Crear DataFrame con el target
df_target = pd.DataFrame(
    data.target,
    columns=["target"]
)

# 4. Separar entrenamiento y predicción
X_train, X_pred, y_train, y_pred = train_test_split(
    df_data,
    df_target,
    test_size=0.05,
    random_state=42
)

# 5. Crear archivo de entrenamiento
df_train = pd.concat([X_train, y_train], axis=1)
df_train.to_csv(BASE_DIR / "breast_cancer_train.csv", index=False)

# 6. Crear archivo de predicción
df_pred = pd.DataFrame(
    X_pred,
    columns=df_data.columns
)
df_pred.to_csv(BASE_DIR / "breast_cancer_pred.csv", index=False)

print("Archivos creados correctamente.")
print("Entrenamiento:", df_train.shape)
print("Predicción:", df_pred.shape)