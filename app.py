import base64
import os
import time

import cv2
import numpy as np
import scipy.stats
from flask import Flask, jsonify, render_template, request

from model_loader import active_model_name, is_screen_fraud, predict_from_bgr
from model_registry import get_fraud_threshold

app = Flask(__name__)


def compute_lbp(gray):
    h, w = gray.shape
    inner = gray[1:h - 1, 1:w - 1]
    lbp = np.zeros(inner.shape, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        lbp += ((inner >= neighbor).astype(np.uint8)) << i
    return lbp


def build_feature_report(img):
    """Build UI diagnostics from handcrafted features (independent of classifier)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crop_size = 512
    if h >= crop_size and w >= crop_size:
        cy, cx = h // 2, w // 2
        crop = gray[cy - crop_size // 2:cy + crop_size // 2,
                    cx - crop_size // 2:cx + crop_size // 2]
    else:
        crop = cv2.resize(gray, (crop_size, crop_size))

    hann_2d = np.outer(np.hanning(crop_size), np.hanning(crop_size))
    windowed_crop = crop.astype(float) * hann_2d
    f = np.fft.fft2(windowed_crop)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_norm = mag / (np.sum(mag) + 1e-8)

    my, mx = crop_size // 2, crop_size // 2
    y_grid, x_grid = np.ogrid[:crop_size, :crop_size]
    dist_from_center = np.sqrt((x_grid - mx) ** 2 + (y_grid - my) ** 2)
    high_freq_mask = (dist_from_center >= 30) & (dist_from_center <= 240)
    high_freq_vals = mag_norm[high_freq_mask]

    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(mag_norm, kernel) == mag_norm
    median_val = np.median(mag_norm)
    mad_val = np.median(np.abs(mag_norm - median_val))
    peak_threshold = max(median_val + 8 * mad_val, 2e-5)
    peaks = local_max & high_freq_mask & (mag_norm > peak_threshold)
    fft_peak_count = int(np.sum(peaks))
    fft_high_max = float(np.max(high_freq_vals))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_kurt = float(scipy.stats.kurtosis(laplacian.ravel()))

    lbp = compute_lbp(crop)
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
    lbp_entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-12)))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s_ch, _ = cv2.split(hsv)
    s_mean = float(np.mean(s_ch) / 255.0)

    return {
        "fft_peak_count": {
            "value": fft_peak_count,
            "description": "Grid frequency peaks (Moiré pattern indicator)",
        },
        "fft_high_max": {
            "value": round(fft_high_max, 5),
            "description": "Highest power grid carriers",
        },
        "lap_kurt": {
            "value": round(lap_kurt, 2),
            "description": "Gradient sharpness outliers",
        },
        "lbp_entropy": {
            "value": round(lbp_entropy, 3),
            "description": "Micro-texture randomness",
        },
        "s_mean": {
            "value": round(s_mean, 3),
            "description": "RGB gamut light emission",
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    t_start = time.perf_counter()
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image payload found"}), 400

        base64_data = data["image"]
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        image_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image bytes"}), 400

        score, model_name = predict_from_bgr(img)
        latency = (time.perf_counter() - t_start) * 1000.0
        threshold = get_fraud_threshold()
        status = "FRAUD (Screen Photo)" if is_screen_fraud(score) else "GENUINE (Real Photo)"

        return jsonify({
            "status": "success",
            "score": score,
            "threshold": threshold,
            "prediction": status,
            "model": model_name,
            "latency_ms": round(latency, 2),
            "features": build_feature_report(img),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Active model: {active_model_name()}")
    app.run(host="127.0.0.1", port=5000, debug=True)
