import streamlit as st
import requests
import pandas as pd

FASTAPI_URL = "http://localhost:8000"

st.title("Demo Microservicio / PyTorch")

data_type = st.selectbox("Tipo de dato", ["tabular", "image", "audio"])
framework = st.selectbox("Framework", ["tensorflow", "pytorch"])
mode = st.radio("Modo de operación", ["Entrenar", "Predecir"])   

if mode == "Entrenar" and data_type == "tabular":
    st.subheader("Entrenamiento")
    csv_file = st.file_uploader("Sube CSV de entrenamiento", type="csv")
    if csv_file is not None:
        # Read the uploaded CSV into a DataFrame
        df = pd.read_csv(csv_file)
        st.subheader("First 5 Rows")
        st.dataframe(df.head())   # shows only the first 5 rows
        # Dropdown con las columnas disponibles para seleccionar el target
        target_column = st.selectbox("Selecciona la columna target", options=df.columns.tolist())
    else:
        target_column = None

    epochs = st.slider("Épocas", 1, 100, 20)
    if csv_file and target_column and st.button("Entrenar"):
        files = {"csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        data = {"framework": framework, 
                "data_type": data_type, 
                "epochs": str(epochs), 
                "target_column": target_column}
        with st.spinner("Entrenando el modelo..."):
            res = requests.post(f"{FASTAPI_URL}/train/", files=files, data=data)
        st.write("Status:", res.status_code)
        st.write("Raw response:", res.text)

        if res.ok:
            st.write(res.json())
        else:
            st.error(f"Error {res.status_code}: {res.text}")

elif mode == "Predecir":
    if data_type == "tabular":
        st.subheader("Predicción")
        csv_file = st.file_uploader("Sube CSV a predecir", type="csv")
        if csv_file is not None:
            # Read the uploaded CSV into a DataFrame
            df = pd.read_csv(csv_file)

            st.subheader("First 5 Rows")
            st.dataframe(df.head())   # shows only the first 5 rows
        
        if csv_file and st.button("Predecir"):
            # OJO: para FastAPI mandamos multipart/form-data con 'files'
            files = {
                "csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")
            }
            data = {
                "data_type": "tabular",
                "framework": framework,  # el que hayas seleccionado en tu UI
            }

            # Ahora sí: enviar archivo + form-data a FastAPI
            res = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)

            st.write("Status:", res.status_code)
            st.write("Raw response:", res.text)

            if res.ok:
                st.success("Predicción:")
                st.json(res.json())
            else:
                st.error("Error en la predicción")
    else:
        file = st.file_uploader("Sube archivo", type=["png","jpg","wav","mp3"])
        if file and st.button("Predecir"):
            data = {"data_type": data_type, "framework": framework}
            files = {"file": (file.name, file.getvalue())}
            r = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)
            st.write(r.json())
