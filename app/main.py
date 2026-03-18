from fastapi import FastAPI
from utils.data import prepare_tabular_data
from trainer_tf import train_model

app = FastAPI()

model = None
scaler = None
accuracy = None

@app.get("/")
def root():
    return {"message": "API Breast Cancer funcionando"}


@app.get("/train")
def train():
    global model, scaler, accuracy

    X_train, X_test, y_train, y_test, scaler = prepare_tabular_data()

    model, _ = train_model(X_train, y_train)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        "message": "Modelo entrenado",
        "accuracy": float(accuracy)
    }


@app.post("/predict")
def predict(features: list):
    global model, scaler

    import numpy as np

    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)

    pred = model.predict(data)[0][0]

    return {
        "probability": float(pred),
        "class": "Benign" if pred > 0.5 else "Malignant"
    }