"""Evaluate predict.py accuracy on real/ and screen/ folders."""

import os
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model_loader import is_screen_fraud, predict_from_path
from model_registry import get_fraud_threshold
from preprocess import collect_image_paths

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    real_paths = collect_image_paths(os.path.join(BASE, "real"))
    screen_paths = collect_image_paths(os.path.join(BASE, "screen"))
    paths = real_paths + screen_paths
    labels = [0] * len(real_paths) + [1] * len(screen_paths)

    if not paths:
        print("No images found.")
        sys.exit(1)

    preds, scores = [], []
    errors = []
    for p, y in zip(paths, labels):
        try:
            s = predict_from_path(p)
            pred = 1 if is_screen_fraud(s) else 0
            scores.append(s)
            preds.append(pred)
            if pred != y:
                errors.append((p, y, s))
        except Exception as e:
            print(f"Error {p}: {e}")

    acc = accuracy_score(labels, preds)
    print(f"Images: {len(paths)} ({len(real_paths)} real, {len(screen_paths)} screen)")
    print(f"Threshold: {get_fraud_threshold()}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(labels, preds, target_names=["Real", "Screen"]))
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(labels, preds))

    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for p, y, s in errors[:15]:
            tag = "real" if y == 0 else "screen"
            print(f"  {p} | true={tag} | score={s:.4f}")


if __name__ == "__main__":
    main()
