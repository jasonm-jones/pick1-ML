import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
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
    # Drop rows with missing target
    df = df.dropna(subset=["win"])

    features = [
        "win_probability",
        "pick_percentage",
        "spread",
        "ev",
        "future_val",
        "opp_win_probability",
        "opp_pick_percentage",
        "opp_spread",
        "opp_ev",
        "opp_future_val"
    ]

    X = df[features].copy()
    y = df["win"].astype(int)

    return X, y

def evaluate_baseline(X, y):
    """
    Evaluate raw Vegas win_probability as a predictor.
    """
    # Normalize to [0,1]
    probs = X["win_probability"].astype(float) / 100.0
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)
    acc = accuracy_score(y, preds)

    print("\n📊 Baseline (Vegas win_probability):")
    print(f"AUC:   {auc:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"Acc:   {acc:.4f}")

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    print("\n🤖 ML Model (RandomForest):")
    print(f"AUC:   {auc:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"Acc:   {acc:.4f}")

    return model

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    # 1) Evaluate baseline Vegas odds
    evaluate_baseline(X, y)

    # 2) Train ML model
    model = train_model(X, y)

    # Save model
    path = os.path.join(MODEL_DIR, "win_predictor.pkl")
    joblib.dump(model, path)
    print(f"\n💾 Model saved to {path}")
