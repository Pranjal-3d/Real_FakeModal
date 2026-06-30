"""Load images and crop faces with MediaPipe for consistent real vs screen detection."""

import glob
import os

import cv2
import numpy as np

_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4,
        )
    return _face_detector


def load_image_bgr(path):
    """Load image as BGR; supports jpg/png/jpeg via OpenCV or PIL."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def crop_face_or_center(img_bgr, padding=0.35):
    """MediaPipe face crop with padding; falls back to center square crop."""
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        return img_bgr

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detector = _get_face_detector()
    results = detector.process(rgb)

    if results.detections:
        det = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        cx = x1 + bw // 2
        cy = y1 + bh // 2
        side = int(max(bw, bh) * (1.0 + padding))
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_bgr[y1:y1 + side, x1:x1 + side]


def prepare_image(img_bgr, use_face_crop=True, output_size=512):
    """Standard pipeline: optional face crop then resize."""
    out = crop_face_or_center(img_bgr) if use_face_crop else img_bgr
    if output_size and out.shape[0] > 0 and out.shape[1] > 0:
        out = cv2.resize(out, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return out


def collect_image_paths(folder):
    """Collect image paths; skip HEIC when a JPG/JPEG of same name exists."""
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG",
                "*.heic", "*.HEIC", "*.heif", "*.HEIF")
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    paths = sorted(set(paths))

    stems_with_raster = {
        os.path.splitext(os.path.basename(p))[0].lower()
        for p in paths
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    }
    filtered = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        if ext in (".heic", ".heif") and stem in stems_with_raster:
            continue
        filtered.append(p)
    return sorted(filtered)
