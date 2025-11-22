# scripts/train.py

import pandas as pd
from pathlib import Path
import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("PredictiveMaintenance")



def train_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train a model with MLflow tracking.
    Returns model object and its score.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)

        # Train
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[INFO] {model_name}: accuracy={acc:.4f} | auc={auc:.4f}")

        return model, acc, auc


def train_pipeline(processed_csv, target_col="Engine Condition"):
    """
    Full training pipeline:
    - Load processed dataset
    - Train 2 models (RF + XGB)
    - Log results in MLflow
    - Save best model to models/best_model.joblib
    """

    print(f"[INFO] Loading processed data: {processed_csv}")
    df = pd.read_csv(processed_csv)

    # Encode target if categorical
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype("category").cat.codes

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Models to test
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=150,
            random_state=42
        )
    }

    best_model = None
    best_auc = -1
    best_name = None

    # Train each model
    for name, model in models.items():
        trained_model, acc, auc = train_model(
            model, name, X_train, X_test, y_train, y_test
        )

        # Select best model by AUC
        if auc > best_auc:
            best_auc = auc
            best_model = trained_model
            best_name = name

    # Save best model locally
    Path("models").mkdir(exist_ok=True)
    best_path = Path("models/best_model.joblib")
    joblib.dump(best_model, best_path)

    print(f"\n[INFO] Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"[INFO] Saved → {best_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--target", default="Engine Condition", help="Target column")
    args = parser.parse_args()

    train_pipeline(args.input, args.target)
