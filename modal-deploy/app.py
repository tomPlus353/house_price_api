import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")  # optional but helpful
    .pip_install(
        "fastapi[standard]",
        "tensorflow",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    )
)

app = modal.App(name="ml-model-api", image=image)

### Model prediction endpoint

import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Set up persistent volume for model storage(files need to be uploaded manually beforehand)
model_volume = modal.Volume.from_name("model-storage", create_if_missing=True)

# directory inside the volume where models are stored
MODEL_DIR = "/models/model"

class InputData(BaseModel):
    data: dict

def load_artifacts():
    print("Loading model into memory...")
    model = tf.keras.models.load_model(f"{MODEL_DIR}/real_estate_model.keras")
    encoder = joblib.load(f"{MODEL_DIR}/one_hot_encoder.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/robust_scaler.joblib")
    num_cols = joblib.load(f"{MODEL_DIR}/num_cols.joblib")
    cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.joblib")

    return model, encoder, scaler, num_cols, cat_cols


@app.function(
    image=image,
    volumes={"/models": model_volume},
)
@modal.fastapi_endpoint(method="POST")
def predict(request: InputData):
        model, encoder, scaler, num_cols, cat_cols = load_artifacts()

        raw = request.data

        # Ensure each value is a list (FastAPI won't enforce it)
        for k, v in raw.items():
            if not isinstance(v, list):
                raw[k] = [v]

        df = pd.DataFrame(raw)

        # Feature engineering
        df["DistrictName2"] = df["DistrictName"].astype(str) + df["Municipality"].astype(str)
        df["AgeAtSale"] = df["Year"] - df["BuildingYear"]

        # Transform
        cat_encoded = encoder.transform(df[cat_cols]).toarray()
        num_scaled = scaler.transform(df[num_cols])
        X = np.hstack([cat_encoded, num_scaled])

        pred = model.predict(X)

        return {"prediction": float(pred[0][0])}

# def predict():
#     return {"message": "Predict endpoint exists!"}


    



