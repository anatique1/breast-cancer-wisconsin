import streamlit as st
import requests
import pandas as pd

FASTAPI_URL = "http://localhost:8000"

st.title("Clasificación de Cáncer de Mama - Wisconsin")
st.write("Microservicio de clasificación utilizando redes neuronales.")

framework = st.selectbox(
    "Framework",
    ["tensorflow", "pytorch"]
)

mode = st.radio(
    "Modo de operación",
    ["Entrenar", "Predecir"]
)

if mode == "Entrenar":
    st.subheader("Entrenamiento del modelo")

    csv_file = st.file_uploader(
        "Sube el CSV de entrenamiento",
        type="csv"
    )

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        st.write("Datos del archivo:")
        st.dataframe(df.head())

        target_column = st.selectbox(
            "Selecciona la columna objetivo",
            options=df.columns.tolist()
        )

        st.write(f"Cantidad de registros: {df.shape[0]}")
        st.write(f"Cantidad de características: {df.shape[1] - 1}")

    else:
        target_column = None

    epochs = st.slider(
        "Número de épocas",
        1,
        100,
        20
    )

    if csv_file is not None and target_column is not None:
        if st.button("Entrenar modelo"):
            files = {
                "csv_file": (
                    csv_file.name,
                    csv_file.getvalue(),
                    "text/csv"
                )
            }

            data = {
                "framework": framework,
                "epochs": str(epochs),
                "target_column": target_column
            }

            with st.spinner("Entrenando el modelo..."):
                res = requests.post(
                    f"{FASTAPI_URL}/train",
                    files=files,
                    data=data
                )

            if res.ok:
                resultado = res.json()
                st.success("Modelo entrenado correctamente.")
                st.write("Modelo guardado en:")
                st.code(resultado["saved_model"])
            else:
                st.error(f"Error {res.status_code}: {res.text}")

elif mode == "Predecir":
    st.subheader("Predicción")

    st.write(
        "Sube un CSV que contenga las características "
        "de los pacientes que se desean clasificar."
    )

    csv_file = st.file_uploader(
        "Sube el CSV a predecir",
        type="csv"
    )

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        st.write("Datos a predecir:")
        st.dataframe(df.head())

        st.write(f"Cantidad de registros: {df.shape[0]}")
        st.write(f"Cantidad de características: {df.shape[1]}")

        if st.button("Realizar predicción"):
            files = {
                "csv_file": (
                    csv_file.name,
                    csv_file.getvalue(),
                    "text/csv"
                )
            }

            data = {
                "data_type": "tabular",
                "framework": framework
            }

            with st.spinner("Realizando predicciones..."):
                res = requests.post(
                    f"{FASTAPI_URL}/predict/",
                    data=data,
                    files=files
                )

            if res.ok:
                resultado = res.json()
                predicciones = resultado["prediction"]

                st.success("Predicción realizada correctamente.")

                resultado_df = df.copy()
                resultado_df["Predicción"] = predicciones

                st.dataframe(resultado_df)

                st.write("0 = Maligno | 1 = Benigno")
            else:
                st.error(f"Error {res.status_code}: {res.text}")