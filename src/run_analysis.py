from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def classification_workflow() -> dict[str, float]:
    data = load_breast_cancer(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        stratify=data.target,
        random_state=RANDOM_STATE,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Breast-cancer classification confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT / "classification_confusion_matrix.png", dpi=160)
    plt.close()

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
    }


def regression_workflow() -> dict[str, float]:
    data = load_diabetes(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    model = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.75)
    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(bounds, bounds, linestyle="--")
    plt.xlabel("Actual progression")
    plt.ylabel("Predicted progression")
    plt.title("Regression: actual vs predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT / "regression_actual_vs_predicted.png", dpi=160)
    plt.close()

    importance = pd.DataFrame(
        {"feature": data.feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT / "regression_feature_importance.csv", index=False)

    return {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
    }


def build_time_series(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    date = pd.date_range("2024-01-01", periods=rows, freq="D")
    trend = np.linspace(90, 145, rows)
    season = 8 * np.sin(np.arange(rows) * 2 * np.pi / 30)
    noise = rng.normal(0, 2.5, rows)
    value = trend + season + noise
    return pd.DataFrame({"date": date, "value": value})


def forecasting_workflow() -> dict[str, float]:
    frame = build_time_series()
    for lag in range(1, 8):
        frame[f"lag_{lag}"] = frame["value"].shift(lag)
    frame = frame.dropna().reset_index(drop=True)

    split = int(len(frame) * 0.8)
    train = frame.iloc[:split]
    test = frame.iloc[split:]
    features = [f"lag_{lag}" for lag in range(1, 8)]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[features], train["value"])
    predictions = model.predict(test[features])

    plt.figure(figsize=(9, 5))
    plt.plot(test["date"], test["value"], label="Actual")
    plt.plot(test["date"], predictions, label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Synthetic index")
    plt.title("Time-series holdout forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "forecast_actual_vs_predicted.png", dpi=160)
    plt.close()

    return {
        "rmse": float(mean_squared_error(test["value"], predictions) ** 0.5),
        "mae": float(mean_absolute_error(test["value"], predictions)),
        "test_rows": int(len(test)),
    }


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    metrics = {
        "classification": classification_workflow(),
        "regression": regression_workflow(),
        "forecasting": forecasting_workflow(),
        "configuration": {"random_state": RANDOM_STATE},
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for task, values in metrics.items():
        if task == "configuration":
            continue
        rows.append({"workflow": task, **values})
    pd.DataFrame(rows).to_csv(OUTPUT / "model_summary.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs written to: {OUTPUT}")


if __name__ == "__main__":
    main()
