"""Pick the best available model using saved validation metrics."""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")


def read_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def write_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def update_model_entry(name, val_accuracy, extra=None):
    config = read_config()
    entry = {"val_accuracy": float(val_accuracy)}
    if extra:
        entry.update(extra)
    config[name] = entry
    best = max(
        ((k, v) for k, v in config.items()
         if k not in ("primary_model", "fraud_threshold", "heuristic_stats") and isinstance(v, dict)),
        key=lambda kv: kv[1].get("val_accuracy", 0),
    )
    config["primary_model"] = best[0]
    write_config(config)
    return config


def save_detection_settings(fraud_threshold, heuristic_stats=None):
    config = read_config()
    config["fraud_threshold"] = float(fraud_threshold)
    if heuristic_stats is not None:
        config["heuristic_stats"] = heuristic_stats
    write_config(config)


def get_fraud_threshold():
    config = read_config()
    return float(config.get("fraud_threshold", 0.35))
