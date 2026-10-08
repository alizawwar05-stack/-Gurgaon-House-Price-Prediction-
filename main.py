import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attrib, cat_attrib):
    
    # Let make the pipline of numerical categories

    num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy=('median'))),
    ('scaler', StandardScaler())
    ])

    #  Let make the pipeline of categorical columns

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    # 8. Let combine the pipeline of numerical and categorical with the help of column transformer

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attrib),
        ('cat', cat_pipeline, cat_attrib)
    ])


    return full_pipeline

if not os.path.exists(MODEL_FILE):

    # Let trains the model

    housing = pd.read_csv("housing.csv")

    # Create a stratified test set
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat",axis=1)


    # Separate labels and features
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value",axis=1)

    # Separate numerical and categorical columns
    num_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    # Build pipeline
    pipeline = build_pipeline(num_attribs,cat_attribs)

    # Transform training data
    housing_prepared = pipeline.fit_transform(housing_features)


    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(
        housing_prepared,
        housing_labels
    )

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model is trained. Congrats!")


else:

    # Let do the inference

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Load the new input data

    input_data = pd.read_csv("input.csv")

    # Transformed the using the training buiid pipeline

    transformed_input = pipeline.transform(input_data)

    # Predict the model

    predictions = model.predict(transformed_input)

    # Add predictions to input data

    input_data['median_house_value'] = predictions

    # Save the results

    input_data.to_csv("output.csv", index=False)

    print("Inference is complete, results saved to output.csv. Enjoy it!")


