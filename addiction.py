from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "teen_phone_addiction_dataset.csv"
MODEL_PATH = BASE_DIR / "phone_addiction_models.joblib"
ADDICTION_CUTOFF = 8


def make_pipeline(features, estimator):
    """Preprocess features, then fit the supplied model."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_preprocessing, numeric_columns),
        ("categorical", categorical_preprocessing, categorical_columns),
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])


# Load and lightly clean the data.
df = pd.read_csv(DATA_PATH)
df = df.replace([" ", "NA", "N/A", ""], np.nan)
df["Addiction_Level"] = pd.to_numeric(df["Addiction_Level"], errors="coerce")
df = df.dropna(subset=["Addiction_Level"]).copy()

# Create the binary classification label: 0 = Not addicted, 1 = Addicted.
df["Addiction_Status"] = (
    df["Addiction_Level"] >= ADDICTION_CUTOFF
).astype(int)

# Do not use identifiers or either target as model inputs.
columns_to_exclude = [
    "ID",
    "Name",
    "Location",
    "Addiction_Level",
    "Addiction_Status",
]

X = df.drop(columns=columns_to_exclude, errors="ignore")
y_classification = df["Addiction_Status"]
y_regression = df["Addiction_Level"]

# Build separate pipelines so each model has its own fitted preprocessor.
classifier = make_pipeline(
    X,
    LogisticRegression(max_iter=2000, random_state=42),
)
regressor = make_pipeline(
    X,
    LinearRegression(),
)

# Fit both pipelines on all available rows for deployment.
classifier.fit(X, y_classification)
regressor.fit(X, y_regression)

# Save the pipelines and the feature order expected by them.
joblib.dump(
    {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": X.columns.tolist(),
        "addiction_cutoff": ADDICTION_CUTOFF,
    },
    MODEL_PATH,
)

print(f"Saved trained pipelines to: {MODEL_PATH}")
print(f"Input features: {X.columns.tolist()}")