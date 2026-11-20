"""California housing (1990 census) — data loading and the models listed in the README.

The notebook loads the data with ``sklearn.datasets.fetch_california_housing``; the same
data is saved as ``california_housing.csv`` so nothing has to be downloaded at runtime.
Target ``MedHouseVal`` is the district median house value in units of $100,000 (capped at 5.0).

Run ``python california_housing.py`` to compare the README's models (Linear Regression,
Decision Tree, Random Forest, Gradient Boosting) with 5-fold CV and write models/metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "california_housing.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TARGET = "MedHouseVal"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
RANDOM_STATE = 42
MODELS = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting"]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    if Path(path).exists():
        df = pd.read_csv(path)
    else:  # same source as the notebook
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def build_model(name: str):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5,
                                                      subsample=0.8, random_state=RANDOM_STATE),
    }[name]


def compare_models(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import KFold, cross_validate

    X, y = df[FEATURES], df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name in MODELS:
        r = cross_validate(build_model(name), X, y, cv=cv,
                           scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
        out[name] = {"r2": float(r["test_r2"].mean()), "r2_std": float(r["test_r2"].std()),
                     "rmse": float(-r["test_neg_root_mean_squared_error"].mean()),
                     "mae": float(-r["test_neg_mean_absolute_error"].mean())}
        print(f"{name:17} R² {out[name]['r2']:.3f} ± {out[name]['r2_std']:.3f}  RMSE {out[name]['rmse']:.3f}  MAE {out[name]['mae']:.3f}")
    return out


def best_model(results: dict) -> str:
    return max(results, key=lambda k: results[k]["r2"])


def train_final(df: pd.DataFrame, name: str):
    return build_model(name).fit(df[FEATURES], df[TARGET])


def load_or_train(df: pd.DataFrame, name: str):
    """Load the saved model; retrain if it is missing or was saved by an incompatible scikit-learn."""
    import warnings
    import joblib

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat sklearn version-mismatch warnings as failures
            model = joblib.load(MODEL_PATH)
        model.predict(df[FEATURES].head(1))
        return model, "loaded"
    except Exception:
        return train_final(df, name), "retrained"


def predict(model, rows: pd.DataFrame | dict) -> np.ndarray:
    frame = pd.DataFrame([rows]) if isinstance(rows, dict) else rows
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # the target is capped at 5.0 in the data, so predictions are clipped to the observed range
    return np.clip(model.predict(frame[FEATURES].astype(float)), 0.0, 5.0)


if __name__ == "__main__":
    data = load_data()
    res = compare_models(data)
    best = best_model(res)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"cv": "5-fold KFold(shuffle=True, random_state=42)", "results": res,
                                        "best": best}, indent=2))
    import joblib
    joblib.dump(train_final(data, best), MODEL_PATH, compress=3)
    print("best:", best, "-> saved", MODEL_PATH)
