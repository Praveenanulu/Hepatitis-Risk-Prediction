"""
=========================================================
Explainable AI-Based Hepatitis Risk Prediction
Model Training Script
=========================================================
This script loads the dataset, trains a Random Forest
Classifier and Logistic Regression model, evaluates them,
and saves the trained models to model.pkl.

Run:  python model.py
=========================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_model():
    # ── 1. Load Dataset ──────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset.csv")

    print(f"📂 Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    feature_cols = ["age", "alcohol", "smoking", "fatigue", "jaundice"]
    X = df[feature_cols]
    y = df["risk"]

    print(f"\nClass distribution:\n{y.value_counts()}\n")

    # ── 2. Train / Test Split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ── 3. Random Forest ─────────────────────────────────
    print("=" * 50)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 50)

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
    print(classification_report(y_test, rf_pred, target_names=["Low Risk", "High Risk"]))

    # ── 4. Logistic Regression (Explainable) ─────────────
    print("=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
    print(classification_report(y_test, lr_pred, target_names=["Low Risk", "High Risk"]))

    # ── 5. Save Model ────────────────────────────────────
    model_path = os.path.join(base_dir, "model.pkl")

    joblib.dump(
        {
            "rf_model": rf_model,
            "lr_model": lr_model,
            "feature_importances": dict(zip(feature_cols, rf_model.feature_importances_)),
            "lr_coefficients": dict(zip(feature_cols, lr_model.coef_[0])),
        },
        model_path
    )

    print(f"\n✅ Model saved at: {model_path}")
    print("🚀 Now run: python app.py\n")


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()


