import glob
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_registry import update_model_entry, save_detection_settings
from features import FEATURE_NAMES, extract_features
from heuristics import compute_heuristic_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKLEARN_PATH = os.path.join(BASE_DIR, "model_sklearn.joblib")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.json")


def collect_dataset():
    real_paths = sorted(
        glob.glob(os.path.join(BASE_DIR, "real", "*.jpg"))
        + glob.glob(os.path.join(BASE_DIR, "real", "*.png"))
        + glob.glob(os.path.join(BASE_DIR, "real", "*.jpeg"))
    )
    screen_paths = sorted(
        glob.glob(os.path.join(BASE_DIR, "screen", "*.jpg"))
        + glob.glob(os.path.join(BASE_DIR, "screen", "*.png"))
        + glob.glob(os.path.join(BASE_DIR, "screen", "*.jpeg"))
    )
    return real_paths, screen_paths


def main():
    real_paths, screen_paths = collect_dataset()
    print(f"Found {len(real_paths)} real images and {len(screen_paths)} screen images.")
    if not real_paths or not screen_paths:
        print("Error: Dataset directories are empty. Run download scripts first.")
        return

    X_data, y_data = [], []
    print("Extracting features from all images...")
    start_time = time.time()

    for path in real_paths:
        try:
            X_data.append(extract_features(path))
            y_data.append(0)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    for path in screen_paths:
        try:
            X_data.append(extract_features(path))
            y_data.append(1)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    X = np.array(X_data, dtype=np.float64)
    y = np.array(y_data)

    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        print(f"  Replaced {nan_mask.sum()} NaN/Inf values with column means.")

    print(f"Feature extraction done in {time.time() - start_time:.2f}s. Shape: {X.shape}")

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                         class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample", random_state=42
        ),
    }

    print("\n--- 5-Fold Cross-Validation ---")
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_scores[name] = scores.mean()
        print(f"{name}: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    val_scores = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        val_scores[name] = accuracy_score(y_val, preds)
        fitted[name] = model

    print("\n--- Holdout Validation ---")
    for name, acc in sorted(val_scores.items(), key=lambda x: -x[1]):
        print(f"{name}: {acc * 100:.2f}%")

    best_name = max(val_scores, key=val_scores.get)
    best_model = fitted[best_name]
    best_acc = val_scores[best_name]
    y_pred = best_model.predict(X_val)

    print(f"\n--- Best Model: {best_name} ({best_acc * 100:.2f}%) ---")
    print(classification_report(y_val, y_pred, target_names=["Real", "Screen"]))

    if hasattr(best_model, "predict_proba"):
        val_probs = best_model.predict_proba(X_val)[:, 1]
    else:
        val_probs = best_model.predict(X_val).astype(float)

    fraud_threshold = 0.38
    for t in np.linspace(0.25, 0.50, 26):
        preds = (val_probs >= t).astype(int)
        fake_recall = recall_score(y_val, preds, pos_label=1)
        real_recall = recall_score(1 - y_val, 1 - preds, pos_label=1)
        if fake_recall >= 0.96 and real_recall >= 0.93:
            fraud_threshold = float(t)
            break

    heuristic_stats = compute_heuristic_stats(X[y == 0], X[y == 1])
    save_detection_settings(fraud_threshold, heuristic_stats)
    print(f"Fraud detection threshold: {fraud_threshold:.2f} (lower = stricter fake catch)")

    scaler = best_model.named_steps["scaler"] if hasattr(best_model, "named_steps") else None
    if scaler is not None:
        means = scaler.mean_
        stds = scaler.scale_
        stds = np.where(stds == 0, 1.0, stds)
        inner_clf = best_model.named_steps["clf"]
    else:
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        stds[stds == 0] = 1.0
        inner_clf = best_model

    bundle = {
        "model_type": best_name,
        "feature_names": FEATURE_NAMES,
        "means": means,
        "stds": stds,
        "model": best_model,
        "cv_accuracy": cv_scores[best_name],
        "val_accuracy": best_acc,
    }
    joblib.dump(bundle, SKLEARN_PATH)
    update_model_entry("sklearn", best_acc, {"model_type": best_name})
    print(f"Saved sklearn model to {SKLEARN_PATH}")

    if best_name == "LogisticRegression":
        lr = inner_clf
        model_data = {
            "model_type": "LogisticRegression",
            "feature_names": FEATURE_NAMES,
            "means": means.tolist(),
            "stds": stds.tolist(),
            "weights": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }
        with open(WEIGHTS_PATH, "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Saved JSON weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
