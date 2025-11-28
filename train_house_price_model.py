import os
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import r2_score, median_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

DATA_DIR = "/Users/tsullivan/Downloads/kaggle_data"
MODEL_DIR = "model" + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")


def run_pipeline(path):
    """
    Train a real estate price prediction model using the given data.

    Parameters:
    path (str): Path to the data directory.

    Returns:
    None
    """
    USE_LIMITED_FEATURES = True
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data (Miyagi)
    with open(f"{path}/trade_prices/04.csv") as f:
        data = pd.read_csv(f, low_memory=False)
    print("Loaded data:", data.shape)

    # Feature selection
    if USE_LIMITED_FEATURES:
        all_cols = [
            "Municipality",
            "DistrictName",
            "NearestStation",
            "CityPlanning",
            "Structure",
            "Area",
            "TotalFloorArea",
            "BuildingYear",
            "CoverageRatio",
            "FloorAreaRatio",
            "MaxTimeToNearestStation",
            "MinTimeToNearestStation",
            "TradePrice",
        ]
    else:
        all_cols = [
            "Region",
            "MinTimeToNearestStation",
            "MaxTimeToNearestStation",
            "TradePrice",
            "Area",
            "AreaIsGreaterFlag",
            "Frontage",
            "FrontageIsGreaterFlag",
            "TotalFloorArea",
            "TotalFloorAreaIsGreaterFlag",
            "BuildingYear",
            "Structure",
            "Purpose",
            "Direction",
            "Classification",
            "Breadth",
            "CityPlanning",
            "CoverageRatio",
            "FloorAreaRatio",
            "Year",
            "Quarter",
        ]

    # Select categorical and numerical features
    num_cols = []
    target_exposed_cols = ["TradePrice"]
    for col in all_cols:
        if (
            col in data.columns
            and pd.api.types.is_numeric_dtype(data[col])
            and col not in target_exposed_cols
        ):
            num_cols.append(col)
    cat_cols = []
    for col in all_cols:
        if col in data.columns and pd.api.types.is_object_dtype(data[col]):
            cat_cols.append(col)

    # Data cleaning
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mean())
    for col in cat_cols:
        if data[col].isnull().any():
            mode_val = data[col].mode()
            if not mode_val.empty:
                data[col] = data[col].fillna(mode_val[0])
            else:
                data[col] = data[col].fillna("Unknown")
    for col in data.select_dtypes(include="object").columns:
        if col not in cat_cols and data[col].isnull().any():
            data[col] = data[col].fillna("Unknown")
    for col in data.select_dtypes(include=np.number).columns:
        if col not in num_cols and data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mean())

    # Feature engineering
    data["DistrictName2"] = data["DistrictName"].astype("str") + data[
        "Municipality"
    ].astype("str")
    cat_cols.append("DistrictName2")

    # Preprocessing
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(data[cat_cols])
    encoded_cat = encoder.transform(data[cat_cols]).toarray()
    scaler = RobustScaler()
    scaled_num = scaler.fit_transform(data[num_cols])
    X = np.hstack((encoded_cat, scaled_num))
    y = data["TradePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train shape:", X_train.shape)

    # Model
    checkpoint_path = os.path.join(MODEL_DIR, "cp.weights.h5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )
    usePreviousModel = False
    usePreviousWeight = True
    interrupted_epoch = 15
    if usePreviousModel:
        model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "real_estate_model.keras")
        )
    else:
        model = Sequential()
        model.add(
            Dense(
                units=int(X_train.shape[1] / 2),
                activation="relu",
                input_shape=(X_train.shape[1],),
            )
        )
        model.add(Dropout(0.2))
        model.add(Dense(units=int(X_train.shape[1] / 4), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=int(X_train.shape[1] / 8), activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        # Load previous weights if available
        if usePreviousWeight and os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)

        model.compile(loss="mae", optimizer="adam")
        early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
        model.fit(
            X_train,
            y_train,
            epochs=100,
            initial_epoch=interrupted_epoch,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, cp_callback],
        )

    # Evaluation
    mae = model.evaluate(X_test, y_test)
    print(f"Mean absolute error: {mae:.2f}")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    print(f"R-square (R2): {r2}")
    print(f"Median absolute error: {medae}")
    avg = y.median()
    print(f"Mean absolute error as percentage of median price: {(medae/avg)*100}%")

    # Save artifacts
    model.save(os.path.join(MODEL_DIR, "real_estate_model.keras"))
    print("Keras model saved successfully.")
    joblib.dump(encoder, os.path.join(MODEL_DIR, "one_hot_encoder.joblib"))
    print("OneHotEncoder saved successfully.")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "robust_scaler.joblib"))
    print("RobustScaler saved successfully.")
    joblib.dump(num_cols, os.path.join(MODEL_DIR, "num_cols.joblib"))
    print("Numerical columns list saved successfully.")
    joblib.dump(cat_cols, os.path.join(MODEL_DIR, "cat_cols.joblib"))
    print("Categorical columns list saved successfully.")


def main():
    run_pipeline(DATA_DIR)


if __name__ == "__main__":
    main()
