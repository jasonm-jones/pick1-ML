import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

CLEAN_DIR = "data-cleaned"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def prepare_features(df):
    # Drop rows with missing target and work on a copy to avoid SettingWithCopyWarning
    df = df.dropna(subset=["win"]).copy()

    desired = ["win_probability", "spread", "pick_percentage"]
    features = [c for c in desired if c in df.columns]

    # Ensure numeric; coerce bad values to NaN (handled by imputer)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    # Guard against inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["win"].astype(int)
    return X, y, features

def evaluate_baseline(X, y):
    """
    Evaluate raw Vegas win_probability as a predictor.
    """
    # Use only rows with valid win_probability for baseline
    mask = X["win_probability"].notna()
    probs = (X.loc[mask, "win_probability"].astype(float) / 100.0).clip(0, 1)
    y_masked = y.loc[mask]

    auc = roc_auc_score(y_masked, probs)
    brier = brier_score_loss(y_masked, probs)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_masked, preds)

    print("\n📊 Baseline (Vegas win_probability):")
    print(f"AUC:   {auc:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"Acc:   {acc:.4f}")

def train_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Pipeline: impute NaNs -> logistic regression; wrap with calibration
    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("lr", LogisticRegression(max_iter=1000))
    ])
    model = CalibratedClassifierCV(base, cv=5)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    print("\n🤖 ML Model (Calibrated Logistic Regression):")
    print(f"AUC:   {auc:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"Acc:   {acc:.4f}")

    return model

if __name__ == "__main__":
    df = load_data()
    X, y, feature_names = prepare_features(df)

    evaluate_baseline(X, y)

    model = train_model(X, y, feature_names)

    # Save model
    path = os.path.join(MODEL_DIR, "win_predictor.pkl")
    joblib.dump(model, path)
    print(f"\n💾 Model saved to {path}")