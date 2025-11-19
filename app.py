from fastapi import FastAPI
import joblib
import numpy as np
import tensorflow as tf
from pydantic import BaseModel

# Load artifacts
model = tf.keras.models.load_model("./model/real_estate_model.keras")
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

test_input: dict = {
    "Municipality": "Izumi Ward,Sendai",
    "DistrictName": "Asahigaoka",
    "Area": 100.19,
    "TotalFloorArea": 34.83,
    "BuildingYear": 1968.0,
    "CoverageRatio": 50.0,
    "FloorAreaRatio": 80.0,
    "MaxTimeToNearestStation": 5.0,
    "MinTimeToNearestStation": 5.0,
}


class InputData(BaseModel):
    data: dict


@app.post("/predict")
def predict(request: InputData = InputData(data=test_input)):
    raw = request.data

    raw["DistrictName2"] = raw["DistrictName"] + raw["Municipality"]

    # Split columns
    num_data = np.array([[raw[col] for col in num_cols]], dtype=float)
    cat_data = np.array([[raw[col] for col in cat_cols]], dtype=object)

    # Transform
    num_scaled = scaler.transform(num_data)
    cat_encoded = encoder.transform(cat_data).toarray()

    # Combine
    X = np.concatenate([cat_encoded, num_scaled], axis=1)

    print(X.shape)

    # Predict
    prediction = model.predict(X)[0][0]

    return {"prediction": float(prediction)}
