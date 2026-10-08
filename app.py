import os
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Load model and pipeline
model = None
pipeline = None
model_loaded = False
pipeline_loaded = False
load_error = None

try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
    pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PIPELINE_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing file: {MODEL_FILE}")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Missing file: {PIPELINE_FILE}")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    model_loaded = True
    pipeline_loaded = True
except Exception as e:
    load_error = str(e)
    model = None
    pipeline = None
    model_loaded = False
    pipeline_loaded = False

# Expected features from pipeline
EXPECTED_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = ["ocean_proximity"]
OCEAN_PROXIMITY_OPTIONS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if load_error is not None:
        error = f"Failed to load model/pipeline: {load_error}. Ensure {MODEL_FILE} and {PIPELINE_FILE} exist in project root."
        return render_template("index.html", prediction=prediction, error=error)

    if request.method == "POST":
        try:
            form_data = request.form

            # Extract and validate inputs
            longitude = form_data.get("longitude", "").strip()
            latitude = form_data.get("latitude", "").strip()
            housing_median_age = form_data.get("housing_median_age", "").strip()
            total_rooms = form_data.get("total_rooms", "").strip()
            total_bedrooms = form_data.get("total_bedrooms", "").strip()
            population = form_data.get("population", "").strip()
            households = form_data.get("households", "").strip()
            median_income = form_data.get("median_income", "").strip()
            ocean_proximity = form_data.get("ocean_proximity", "").strip()

            missing = []
            for k, v in {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }.items():
                if v == "":
                    missing.append(k.replace("_", " ").title())

            if missing:
                raise ValueError(f"Missing required inputs: {', '.join(missing)}")

            # Convert types
            try:
                longitude = float(longitude)
                latitude = float(latitude)
                housing_median_age = float(housing_median_age)
                total_rooms = float(total_rooms)
                total_bedrooms = float(total_bedrooms)
                population = float(population)
                households = float(households)
                median_income = float(median_income)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid numeric input: {e}")

            if ocean_proximity not in OCEAN_PROXIMITY_OPTIONS:
                raise ValueError(
                    f"Invalid ocean proximity. Must be one of: {', '.join(OCEAN_PROXIMITY_OPTIONS)}"
                )

            # Build dataframe
            input_df = pd.DataFrame(
                [
                    {
                        "longitude": longitude,
                        "latitude": latitude,
                        "housing_median_age": housing_median_age,
                        "total_rooms": total_rooms,
                        "total_bedrooms": total_bedrooms,
                        "population": population,
                        "households": households,
                        "median_income": median_income,
                        "ocean_proximity": ocean_proximity,
                    }
                ]
            )

            # Ensure column order matches expected
            input_df = input_df[EXPECTED_FEATURES]

            # Transform and predict
            transformed = pipeline.transform(input_df)
            pred = model.predict(transformed)
            prediction = float(pred[0]) if hasattr(pred, "__getitem__") else float(pred)

        except Exception as e:
            error = f"Prediction error: {e}"
            traceback.print_exc()
            return render_template("index.html", prediction=prediction, error=error)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        ocean_proximity_options=OCEAN_PROXIMITY_OPTIONS,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
