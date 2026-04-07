from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cluster_telemetry.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run 'python src/generate_data.py' first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def regression_task(df: pd.DataFrame) -> None:
    features = [
        "memory_usage",
        "io_wait",
        "network_in",
        "network_out",
        "active_jobs",
        "queue_depth",
        "temperature",
        "power_draw",
        "hour",
        "dayofweek",
        "node_id",
    ]
    target = "cpu_usage"

    X = df[features]
    y = df[target]
    numeric_features = [c for c in features if c != "node_id"]
    categorical_features = ["node_id"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }
    (OUTPUT_DIR / "regression_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, OUTPUT_DIR / "cpu_usage_regressor.joblib")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual CPU Usage")
    plt.ylabel("Predicted CPU Usage")
    plt.title("CPU Usage Prediction")
    low = min(y_test.min(), preds.min())
    high = max(y_test.max(), preds.max())
    plt.plot([low, high], [low, high])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cpu_prediction_scatter.png", dpi=180)
    plt.close()

    rf = model.named_steps["regressor"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top Features for CPU Usage Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close()


def classification_task(df: pd.DataFrame) -> None:
    features = [
        "cpu_usage",
        "memory_usage",
        "io_wait",
        "network_in",
        "network_out",
        "active_jobs",
        "queue_depth",
        "temperature",
        "power_draw",
        "hour",
        "dayofweek",
        "node_id",
    ]
    target = "failure_risk"
    X = df[features]
    y = df[target]

    numeric_features = [c for c in features if c != "node_id"]
    categorical_features = ["node_id"]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=220, random_state=42, n_jobs=-1, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    (OUTPUT_DIR / "classification_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, OUTPUT_DIR / "failure_risk_classifier.joblib")


def anomaly_task(df: pd.DataFrame) -> None:
    feature_cols = [
        "cpu_usage",
        "memory_usage",
        "io_wait",
        "network_in",
        "network_out",
        "active_jobs",
        "queue_depth",
        "temperature",
        "power_draw",
    ]
    X = df[feature_cols]
    model = IsolationForest(contamination=0.03, random_state=42)
    anomaly_flags = model.fit_predict(X)
    scores = model.decision_function(X)

    summary = {
        "anomaly_count": int((anomaly_flags == -1).sum()),
        "total_records": int(len(df)),
        "anomaly_ratio": float((anomaly_flags == -1).mean()),
    }
    (OUTPUT_DIR / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(model, OUTPUT_DIR / "anomaly_detector.joblib")

    plot_df = df.copy()
    plot_df["anomaly_score"] = scores
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df.index, plot_df["anomaly_score"], alpha=0.4)
    plt.xlabel("Record Index")
    plt.ylabel("Anomaly Score")
    plt.title("Isolation Forest Anomaly Scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_scores.png", dpi=180)
    plt.close()


def main() -> None:
    df = load_data()
    regression_task(df)
    classification_task(df)
    anomaly_task(df)
    print(f"Training complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
