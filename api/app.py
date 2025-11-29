from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

from utils.validator import validate_categorical_inputs, load_allowed_categories

# Load artifacts
MODEL_DIR = "./model_renovation"
model = tf.keras.models.load_model(f"{MODEL_DIR}/real_estate_model.keras")
encoder = joblib.load(f"{MODEL_DIR}/one_hot_encoder.joblib")
scaler = joblib.load(f"{MODEL_DIR}/robust_scaler.joblib")
num_cols = joblib.load(f"{MODEL_DIR}/num_cols.joblib")
cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.joblib")

app = FastAPI()


class InputData(BaseModel):
    """Input data model for prediction."""

    data: dict


@app.post("/predict")
def predict(request: InputData):
    """
    Predict real estate price based on input data.

    Parameters:
    request (InputData): Data to be used for prediction.

    Returns:
    dict: Prediction result.
    """

    mapping = load_allowed_categories(MODEL_DIR)
    print("Allowed categories mapping computed.")

    validated_data, validation_issues = validate_categorical_inputs(
        request.data, mapping
    )
    if validation_issues:
        print("Validation issues found:")
        print(validation_issues)
        print("Validated data:", validated_data)
        return {"error": "Input validation failed", "details": validation_issues}

    print("request.data", request.data)

    # Extract raw data
    raw = request.data

    # Ensure all values are lists
    for k, v in raw.items():
        if not isinstance(v, list):
            raw[k] = [v]

    # Create DataFrame
    df = pd.DataFrame(raw)
    print("Input DataFrame:")
    print(df)
    # Engineer new feature
    df["DistrictName2"] = df["DistrictName"].astype("str") + df["Municipality"].astype(
        "str"
    )
    print("DataFrame with DistrictName2:")
    print(df)

    # Engineer new feature
    df["AgeAtSale"] = df["Year"] - df["BuildingYear"]

    # Extract num/cat columns and encode them in same way as training
    print("cat_cols:", cat_cols)
    print("num_cols:", num_cols)
    cat_encoded = encoder.transform(df[cat_cols]).toarray()
    print("Encoded categorical features shape:", cat_encoded.shape)
    print(
        "Encoded categorical features (first row):",
        cat_encoded[0] if cat_encoded.shape[0] > 0 else cat_encoded,
    )
    # Standardize numerical features using the *already fitted* scaler
    num_scaled = scaler.transform(df[num_cols])
    print("Scaled numerical features shape:", num_scaled.shape)
    print(
        "Scaled numerical features (first row):",
        num_scaled[0] if num_scaled.shape[0] > 0 else num_scaled,
    )

    # Combine numerical and categorical features
    X = np.hstack([cat_encoded, num_scaled])
    print("Final model input shape:", X.shape)
    print("Final model input (first row):", X[0] if X.shape[0] > 0 else X)

    # Predict
    prediction = model.predict(X)
    print("Raw model prediction:", prediction)

    return {"prediction": f"{prediction[0][0]:,.2f}円"}
