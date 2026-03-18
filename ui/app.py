import streamlit as st
import requests

st.title("Clasificación de Cáncer de Mama")

# Entrenar modelo
if st.button("Entrenar modelo"):
    res = requests.get("http://localhost:8000/train").json()
    st.success(f"Modelo entrenado - Accuracy: {res['accuracy']:.4f}")

st.write("Ingrese los 30 valores:")

inputs = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predecir"):
    res = requests.post(
        "http://localhost:8000/predict",
        json=inputs
    ).json()

    st.write(f"Probabilidad: {res['probability']:.4f}")

    if res["class"] == "Benign":
        st.success("Resultado: Benigno")
    else:
        st.error("Resultado: Maligno")
