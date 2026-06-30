"""Shared handcrafted feature extraction for screen-recapture detection."""

import numpy as np
import cv2
import scipy.stats


def compute_lbp(gray):
    h, w = gray.shape
    inner = gray[1:h - 1, 1:w - 1]
    lbp = np.zeros(inner.shape, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        lbp += ((inner >= neighbor).astype(np.uint8)) << i
    return lbp


def _fft_features(crop, min_r=30, max_r=240, peak_floor=2e-5):
    crop_size = crop.shape[0]
    hann_2d = np.outer(np.hanning(crop_size), np.hanning(crop_size))
    windowed_crop = crop.astype(float) * hann_2d

    f = np.fft.fft2(windowed_crop)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    total_energy = np.sum(mag)
    mag_norm = mag / (total_energy + 1e-8)

    my, mx = crop_size // 2, crop_size // 2
    y_grid, x_grid = np.ogrid[:crop_size, :crop_size]
    dist_from_center = np.sqrt((x_grid - mx) ** 2 + (y_grid - my) ** 2)
    high_freq_mask = (dist_from_center >= min_r) & (dist_from_center <= max_r)
    high_freq_vals = mag_norm[high_freq_mask]

    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(mag_norm, kernel) == mag_norm
    median_val = np.median(mag_norm)
    mad_val = np.median(np.abs(mag_norm - median_val))
    peak_threshold = max(median_val + 8 * mad_val, peak_floor)
    peaks = local_max & high_freq_mask & (mag_norm > peak_threshold)

    return [
        float(np.mean(high_freq_vals)),
        float(np.std(high_freq_vals)),
        float(np.max(high_freq_vals)),
        int(np.sum(peaks)),
    ]


def extract_features_from_bgr(img_bgr):
    """Extract feature vector from a BGR OpenCV image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    feature_list = []

    for crop_size in (256, 512):
        if h >= crop_size and w >= crop_size:
            cy, cx = h // 2, w // 2
            crop = gray[cy - crop_size // 2:cy + crop_size // 2,
                        cx - crop_size // 2:cx + crop_size // 2]
        else:
            crop = cv2.resize(gray, (crop_size, crop_size))
        feature_list.extend(_fft_features(crop))

    crop_512 = crop if crop.shape[0] == 512 else cv2.resize(gray, (512, 512))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian))
    lap_flat = laplacian.ravel()
    if np.std(lap_flat) < 1e-10:
        lap_skew, lap_kurt = 0.0, 0.0
    else:
        lap_skew = float(scipy.stats.skew(lap_flat))
        lap_kurt = float(scipy.stats.kurtosis(lap_flat))
    feature_list.extend([lap_var, lap_skew, lap_kurt])

    lbp = compute_lbp(crop_512)
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
    feature_list.extend([
        float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-12))),
        float(np.max(lbp_hist)),
        float(np.std(lbp_hist)),
    ])

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    feature_list.extend([
        float(np.mean(s_ch) / 255.0),
        float(np.std(s_ch) / 255.0),
        float(np.mean(v_ch) / 255.0),
        float(np.std(v_ch) / 255.0),
    ])

    return [0.0 if not np.isfinite(v) else v for v in feature_list]


def extract_features(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return extract_features_from_bgr(img_bgr)


FEATURE_NAMES = [
    "fft256_high_mean", "fft256_high_std", "fft256_high_max", "fft256_peak_count",
    "fft512_high_mean", "fft512_high_std", "fft512_high_max", "fft512_peak_count",
    "lap_var", "lap_skew", "lap_kurt",
    "lbp_entropy", "lbp_max", "lbp_std",
    "s_mean", "s_std", "v_mean", "v_std",
]
