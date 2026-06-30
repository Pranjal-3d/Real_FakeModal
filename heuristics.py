"""Rule-based screen-recapture signals that help on unseen fake photos."""

import json
import os

import numpy as np

from features import extract_features_from_bgr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

# Fallback stats from training set (real vs screen percentiles)
DEFAULT_STATS = {
    "real_peak512_p75": 800.0,
    "screen_peak512_p25": 1200.0,
    "real_lbp_entropy_p25": 5.0,
    "screen_lbp_entropy_p75": 6.2,
    "real_lap_var_p75": 400.0,
    "screen_lap_var_p25": 150.0,
}


def _sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_reference_stats():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_STATS.copy()
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config.get("heuristic_stats", DEFAULT_STATS.copy())


def heuristic_screen_score(img_bgr, prepared=None):
    """
    Score in [0, 1] from physical screen artifacts (Moiré peaks, texture, sharpness).
    Works without ML and helps catch fakes the neural net misses.
    """
    if prepared is None:
        from preprocess import prepare_image
        prepared = prepare_image(img_bgr, use_face_crop=True, output_size=512)
    feats = extract_features_from_bgr(img_bgr, prepared=prepared)

    stats = load_reference_stats()
    peak256 = feats[3]
    peak512 = feats[7]
    lap_var = feats[8]
    lbp_entropy = feats[11]
    s_mean = feats[14]
    v_mean = feats[16]

    score_parts = []

    real_peak_hi = stats.get("real_peak512_p75", 800.0)
    screen_peak_lo = stats.get("screen_peak512_p25", 1200.0)

    # Moiré peaks: only flag when clearly above BOTH class distributions
    peak_cutoff = max(real_peak_hi, screen_peak_lo) * 0.85
    if peak512 >= peak_cutoff and peak512 > 900:
        peak_strength = (peak512 - peak_cutoff) / max(peak_cutoff, 1.0)
        score_parts.append(_sigmoid(1.8 * peak_strength))

    # Very high peak count at both scales (classic LCD grid)
    if peak512 > 1500 and peak256 > 800:
        score_parts.append(0.7)

    # Structured micro-texture + display brightness together
    ent_ref = stats.get("real_lbp_entropy_p25", 5.0)
    if lbp_entropy < ent_ref and s_mean > 0.30 and v_mean > 0.50:
        score_parts.append(0.55)

    # Very flat capture + moiré hints
    if lap_var < stats.get("screen_lap_var_p25", 150.0) and peak512 > 1000:
        score_parts.append(0.5)

    if len(score_parts) < 2:
        return 0.0
    return float(min(max(score_parts), 1.0))


def compute_heuristic_stats(real_features, screen_features):
    """Build percentile thresholds from training features for heuristic scoring."""
    real = np.asarray(real_features, dtype=np.float64)
    screen = np.asarray(screen_features, dtype=np.float64)
    return {
        "real_peak512_p75": float(np.percentile(real[:, 7], 75)),
        "screen_peak512_p25": float(np.percentile(screen[:, 7], 25)),
        "real_lbp_entropy_p25": float(np.percentile(real[:, 11], 25)),
        "screen_lbp_entropy_p75": float(np.percentile(screen[:, 11], 75)),
        "real_lap_var_p75": float(np.percentile(real[:, 8], 75)),
        "screen_lap_var_p25": float(np.percentile(screen[:, 8], 25)),
    }
