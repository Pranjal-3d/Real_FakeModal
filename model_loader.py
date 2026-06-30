"""Unified model loading and prediction for liveness / screen-recapture detection."""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOBILENET_PATH = os.path.join(BASE_DIR, "mobilenet_liveness.pt")
SKLEARN_PATH = os.path.join(BASE_DIR, "model_sklearn.joblib")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.json")

_mobilenet_model = None
_mobilenet_device = None
_sklearn_bundle = None
_lr_model = None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _load_lr_model():
    global _lr_model
    if _lr_model is not None:
        return _lr_model
    if not os.path.exists(WEIGHTS_PATH):
        return None
    with open(WEIGHTS_PATH, "r") as f:
        _lr_model = json.load(f)
    return _lr_model


def _load_sklearn_bundle():
    global _sklearn_bundle
    if _sklearn_bundle is not None:
        return _sklearn_bundle
    if not os.path.exists(SKLEARN_PATH):
        return None
    import joblib
    _sklearn_bundle = joblib.load(SKLEARN_PATH)
    return _sklearn_bundle


def _load_mobilenet():
    global _mobilenet_model, _mobilenet_device
    if _mobilenet_model is not None:
        return _mobilenet_model, _mobilenet_device
    if not os.path.exists(MOBILENET_PATH):
        return None, None

    import torch
    from torchvision import models

    _mobilenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MOBILENET_PATH, map_location=_mobilenet_device, weights_only=False)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_mobilenet_device)
    model.eval()
    _mobilenet_model = {
        "model": model,
        "img_size": checkpoint.get("img_size", 224),
        "mean": checkpoint.get("mean", [0.485, 0.456, 0.406]),
        "std": checkpoint.get("std", [0.229, 0.224, 0.225]),
    }
    return _mobilenet_model, _mobilenet_device


def active_model_name():
    if os.path.exists(MOBILENET_PATH) and os.path.exists(SKLEARN_PATH):
        return "Ensemble(MobileNetV3+sklearn)"
    if os.path.exists(SKLEARN_PATH):
        bundle = _load_sklearn_bundle()
        return bundle.get("model_type", "sklearn") if bundle else "sklearn"
    if os.path.exists(WEIGHTS_PATH):
        return "LogisticRegression"
    return "none"


def _predict_mobilenet(img_bgr):
    mobilenet, device = _load_mobilenet()
    if mobilenet is None:
        return None
    import torch
    import cv2
    from torchvision import transforms

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    size = mobilenet["img_size"]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mobilenet["mean"], std=mobilenet["std"]),
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = mobilenet["model"](tensor).squeeze()
        return float(torch.sigmoid(logit).item())


def _predict_sklearn(img_bgr):
    bundle = _load_sklearn_bundle()
    if bundle is None:
        return None
    from features import extract_features_from_bgr
    features = np.array(extract_features_from_bgr(img_bgr), dtype=np.float64).reshape(1, -1)
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    return float(model.predict(features)[0])


def predict_from_bgr(img_bgr):
    """Return fraud probability in [0, 1] where 1 = screen recapture."""
    from heuristics import heuristic_screen_score

    mobilenet_prob = _predict_mobilenet(img_bgr)
    sklearn_prob = _predict_sklearn(img_bgr)
    heuristic_prob = heuristic_screen_score(img_bgr)

    ml_probs = [p for p in (mobilenet_prob, sklearn_prob) if p is not None]
    if ml_probs:
        ml_score = sum(ml_probs) / len(ml_probs)
        # Heuristics only boost when ML is already somewhat suspicious
        if heuristic_prob >= 0.6 and ml_score >= 0.25:
            prob = max(ml_score, 0.6 * ml_score + 0.4 * heuristic_prob)
        else:
            prob = ml_score
        if mobilenet_prob is not None and sklearn_prob is not None:
            name = "Ensemble(MobileNetV3+sklearn+heuristics)"
        elif mobilenet_prob is not None:
            name = "MobileNetV3+heuristics"
        else:
            bundle = _load_sklearn_bundle()
            name = f"{bundle.get('model_type', 'sklearn')}+heuristics"
        return float(prob), name

    if heuristic_prob >= 0.45:
        return float(heuristic_prob), "heuristics-only"

    lr = _load_lr_model()
    if lr is not None:
        from features import extract_features_from_bgr
        features = np.array(extract_features_from_bgr(img_bgr), dtype=np.float64)
        means = np.array(lr["means"])
        stds = np.array(lr["stds"])
        weights = np.array(lr["weights"])
        intercept = lr["intercept"]
        # Legacy 14-feature models: pad or truncate gracefully
        if len(features) != len(means):
            if len(means) == 14 and len(features) == 18:
                features = np.array([
                    features[4], features[5], features[6], features[7],
                    features[8], features[9], features[10], features[11],
                    features[12], features[13], features[14],
                    features[15], features[16], features[17],
                ])
            elif len(features) != len(means):
                raise ValueError("Feature dimension mismatch with saved model.")
        features_scaled = (features - means) / stds
        score = np.dot(features_scaled, weights) + intercept
        return float(_sigmoid(score)), "LogisticRegression"

    raise RuntimeError("No trained model found. Run train_mobilenet.py or train_model.py first.")


def predict_from_path(image_path):
    import cv2
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    prob, _ = predict_from_bgr(img_bgr)
    return prob


def is_screen_fraud(score):
    from model_registry import get_fraud_threshold
    return float(score) >= get_fraud_threshold()
