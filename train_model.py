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
from features import FEATURE_NAMES, extract_features_from_bgr
from heuristics import compute_heuristic_stats
from preprocess import collect_image_paths, load_image_bgr, prepare_image
from augmentation import augment_variants

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKLEARN_PATH = os.path.join(BASE_DIR, "model_sklearn.joblib")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.json")
AUG_PER_IMAGE = 2


def collect_dataset():
    real_paths = collect_image_paths(os.path.join(BASE_DIR, "real"))
    screen_paths = collect_image_paths(os.path.join(BASE_DIR, "screen"))
    return real_paths, screen_paths


def features_from_path(path, augment=False):
    img = load_image_bgr(path)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    prepared = prepare_image(img, use_face_crop=True, output_size=512)
    rows = [extract_features_from_bgr(img, prepared=prepared)]
    if augment:
        for aug in augment_variants(prepared, count=AUG_PER_IMAGE):
            rows.append(extract_features_from_bgr(aug, prepared=aug))
    return rows


def main():
    real_paths, screen_paths = collect_dataset()
    print(f"Found {len(real_paths)} real images and {len(screen_paths)} screen images.")
    if not real_paths or not screen_paths:
        print("Error: Dataset directories are empty. Run download scripts first.")
        return

    all_paths = real_paths + screen_paths
    all_labels = [0] * len(real_paths) + [1] * len(screen_paths)
    train_paths, val_paths, y_train_paths, y_val_paths = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    print("Extracting features (MediaPipe face crop + train-only augmentation)...")
    start_time = time.time()

    for path, label in zip(train_paths, y_train_paths):
        try:
            for feats in features_from_path(path, augment=True):
                X_train_list.append(feats)
                y_train_list.append(label)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    for path, label in zip(val_paths, y_val_paths):
        try:
            for feats in features_from_path(path, augment=False):
                X_val_list.append(feats)
                y_val_list.append(label)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    X_train = np.array(X_train_list, dtype=np.float64)
    y_train = np.array(y_train_list)
    X_val = np.array(X_val_list, dtype=np.float64)
    y_val = np.array(y_val_list)
    X = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y = np.concatenate([y_train, y_val]) if len(y_val) else y_train

    for arr in (X_train, X_val):
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    print(f"Feature extraction done in {time.time() - start_time:.2f}s.")
    print(f"  Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

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

    print("\n--- 5-Fold Cross-Validation (train set) ---")
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores[name] = scores.mean()
        print(f"{name}: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

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
