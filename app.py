from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

# Load artifacts
model = tf.keras.models.load_model("./model/real_estate_model_1.keras")
encoder = joblib.load("./model/one_hot_encoder.joblib")
scaler = joblib.load("./model/robust_scaler.joblib")
num_cols = joblib.load("./model/num_cols.joblib")
cat_cols = joblib.load("./model/cat_cols.joblib")

app = FastAPI()

# test_input: dict = {
#     "Municipality": "Izumi Ward,Sendai City",
#     "DistrictName": "Minaminakayama",
#     "Area": 225.09,
#     "TotalFloorArea": 129.89,
#     "BuildingYear": 1996.0,
#     "CoverageRatio": 40.0,
#     "FloorAreaRatio": 60.0,
#     "MaxTimeToNearestStation": 21.0,
#     "MinTimeToNearestStation": 21.0,
# }
## real asking price: 1,990万円

test_input: dict = {
    "Municipality": ["Izumi Ward,Sendai"],
    "DistrictName": ["Asahigaoka"],
    "Area": [100.19],
    "TotalFloorArea": [34.83],
    "BuildingYear": [1968.0],
    "CoverageRatio": [50.0],
    "FloorAreaRatio": [80.0],
    "MaxTimeToNearestStation": [5.0],
    "MinTimeToNearestStation": [5.0],
}
## real asking price: 1,300万円

# test_input = {
#     "Municipality": ["Izumi Ward,Sendai"],
#     "DistrictName": ["Yamanotera"],
#     "Area": [380.18],
#     "TotalFloorArea": [180.94],
#     "BuildingYear": [1980.66],
#     "CoverageRatio": [50.0],
#     "FloorAreaRatio": [80.0],
#     "MaxTimeToNearestStation": [13.0],
#     "MinTimeToNearestStation": [48.0],
# }
## real asking price: 1,700万円


class InputData(BaseModel):
    """Input data model for prediction."""

    data: dict


@app.post("/predict")
def predict(request: InputData = InputData(data=test_input)):
    """
    Predict real estate price based on input data.

    Parameters:
    request (InputData): Data to be used for prediction.

    Returns:
    dict: Prediction result.
    """

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
    df["DistrictName2"] = df["DistrictName"].astype("str") + df["Municipality"].astype("str")
    print("DataFrame with DistrictName2:")
    print(df)

    # Extract num/cat columns and encode them in same way as training
    print("cat_cols:", cat_cols)
    print("num_cols:", num_cols)
    cat_encoded = encoder.transform(df[cat_cols]).toarray()
    print("Encoded categorical features shape:", cat_encoded.shape)
    print("Encoded categorical features (first row):", cat_encoded[0] if cat_encoded.shape[0] > 0 else cat_encoded)
    # Standardize numerical features using the *already fitted* scaler
    num_scaled = scaler.transform(df[num_cols])
    print("Scaled numerical features shape:", num_scaled.shape)
    print("Scaled numerical features (first row):", num_scaled[0] if num_scaled.shape[0] > 0 else num_scaled)

    # Combine numerical and categorical features
    X = np.hstack([cat_encoded, num_scaled])
    print("Final model input shape:", X.shape)
    print("Final model input (first row):", X[0] if X.shape[0] > 0 else X)

    # Predict
    prediction = model.predict(X)
    print("Raw model prediction:", prediction)

    return {"prediction": float(prediction[0][0])}
