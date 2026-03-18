# Neural Network – Breast Cancer Classification

Este proyecto implementa una red neuronal para clasificar tumores como benignos o malignos usando el dataset Breast Cancer Wisconsin.

## Dataset

* 569 muestras
* 30 características
* Clases:

  * 0 → Malignant
  * 1 → Benign

## Arquitectura

* Entrada: 30 neuronas
* Capas ocultas: 16 y 8 neuronas (ReLU)
* Salida: 1 neurona (Sigmoid)
* Loss: Binary Crossentropy
* Optimizer: Adam

## Instalación

```bash
python -m venv .venv
```

### Activar entorno

Windows:

```bash
.\.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar backend

```bash
uvicorn app.main:app --reload --port 8000
```

## Ejecutar frontend

```bash
streamlit run ui/app.py
```

## Integrantes

* Juan Esteban Sosa
