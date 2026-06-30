# Submission Note — Spot the Fake Photo

## Approach

We detect **screen recapture** (photo of a phone/laptop/printout) using a **hybrid system**:

1. **Physics features** — 2D FFT on Hann-windowed crops detects Moiré / LCD grid peaks; LBP texture entropy, Laplacian sharpness, and HSV color stats capture display artifacts.
2. **MobileNetV3-Small** (ImageNet-pretrained, fine-tuned) — learns visual patterns from labeled examples.
3. **Ensemble** — averages ML scores; heuristics boost when multiple screen signals agree.
4. **Threshold** — score ≥ **0.39** → photo-of-screen; below → real photo.

`predict.py` prints a single float in **[0, 1]** where **0 = real, 1 = screen** (as required).

---

## Dataset used

| Folder | What it should contain (per assignment) | What we have today |
|--------|----------------------------------------|-------------------|
| `real/` | ~50 **phone photos of real things** | 50 synthetic web images (`im_*`) + 90 HF live webcam frames (`hf_*`) |
| `screen/` | ~50 **phone photos of screens/printouts** | 50 **computer-simulated** screens (`im_*`) + 90 HF monitor replays (`hf_*`) |

**Important:** The assignment asks you to **use your phone** and capture real recaptures. Our `im_*` synthetic pairs are useful for development but are **not a substitute** for phone-captured screen photos. **Before final submission, add ~50 real + ~50 screen photos taken with your phone** (see README → “Phone dataset for submission”).

---

## Accuracy (honest)

Measured on our **20% holdout split** of the current 280-image training set:

| Model | Holdout accuracy |
|-------|------------------|
| sklearn (Logistic Regression + features) | **98.2%** |
| MobileNetV3-Small | **96.4%** |
| Ensemble | **~98%** |

**Caveat:** This is accuracy on **our** data, not SalesCode’s held-out set. Synthetic `im_*` screen images are easier than real phone-of-monitor photos. Expect **lower accuracy on company photos** until retrained with phone-captured `real/` and `screen/` folders matching their attack types.

**5-fold cross-validation** on full dataset: ~85–88% (more realistic generalization estimate).

---

## Required numbers

### Latency

| Metric | Value |
|--------|-------|
| Device | Laptop CPU (Intel, Windows) |
| Mean | **~270 ms** per image (first run includes model load) |
| Median (warm) | **~142 ms** per image |
| p95 | **~164 ms** |

Feature-only path (sklearn without MobileNet): **~25–40 ms**. Full ensemble loads PyTorch MobileNetV3.

**For mobile:** export FFT + logistic weights to native code (OpenCV on Android/iOS) → target **15–30 ms** on-device without PyTorch.

### Cost per image

| Deployment | Cost |
|------------|------|
| **On-device (recommended)** | **$0** — runs on user’s phone, no API |
| Cloud VPS (4 vCPU, batch API) | **~$0.00005/image** (~$50 per million) assuming 200 ms CPU @ $0.04/hr/vCPU |
| GPU server | Not required for this solution |

---

## What I’d improve with more time

1. **Replace synthetic training data** with 50+50 phone-captured `real/` and `screen/` per assignment brief.
2. **Collect attack variety** — different phones, laptops, printouts, angles, office lighting.
3. **Quantize MobileNetV3** to INT8 for &lt;30 ms on mid-range Android.
4. **Active learning** — log low-confidence predictions in production and retrain monthly as cheaters adapt.
5. **Threshold tuning** on a SalesCode-style validation set to maximize fake recall while keeping real false-reject &lt;2%.

---

## Cut-off score for fraud flagging

Current threshold: **0.39** (tuned on validation for ≥96% fake recall with ≥93% real recall).

For production KYC/liveness I would:
- Start at **0.35–0.40** on a labeled validation set from real traffic.
- Monitor false reject rate on genuine users and false accept on known attacks.
- Prefer **missing a borderline real** over **accepting a screen** (asymmetric cost).

---

## Files to submit

- `predict.py` — one-line interface ✓
- `train_model.py`, `train_mobilenet.py`, `train_all.py` — training
- `features.py`, `model_loader.py`, `heuristics.py` — core logic
- `app.py` + `templates/index.html` — optional live camera demo ✓
- `real/`, `screen/` — training images
- This note (`SUBMISSION_NOTE.md`)
