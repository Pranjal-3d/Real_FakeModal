"""
Add your own photos to the dataset and retrain.

Examples:
    python add_sample.py fake C:\\Photos\\phone_screen.jpg
    python add_sample.py real C:\\Photos\\my_face.jpg
    python add_sample.py fake screen1.jpg screen2.jpg
"""

import glob
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE_DIR, "real")
SCREEN_DIR = os.path.join(BASE_DIR, "screen")


def next_name(folder, prefix="custom"):
    existing = glob.glob(os.path.join(folder, f"{prefix}_*.jpg"))
    idx = len(existing)
    return os.path.join(folder, f"{prefix}_{idx:03d}.jpg")


def main():
    if len(sys.argv) < 3:
        print("Usage: python add_sample.py <real|fake> <image1> [image2 ...]")
        sys.exit(1)

    label = sys.argv[1].lower()
    if label not in ("real", "fake"):
        print("Label must be 'real' or 'fake'")
        sys.exit(1)

    dest_dir = REAL_DIR if label == "real" else SCREEN_DIR
    os.makedirs(dest_dir, exist_ok=True)
    prefix = "custom_real" if label == "real" else "custom_fake"

    copied = 0
    for src in sys.argv[2:]:
        if not os.path.isfile(src):
            print(f"Skip (not found): {src}")
            continue
        dst = next_name(dest_dir, prefix)
        shutil.copy2(src, dst)
        print(f"Added {label}: {dst}")
        copied += 1

    if copied == 0:
        print("No images copied.")
        sys.exit(1)

    print(f"\nRetraining on updated dataset ({copied} new image(s))...")
    subprocess.run([sys.executable, "train_all.py"], cwd=BASE_DIR, check=True)
    print("\nDone. Restart app.py if it is running.")


if __name__ == "__main__":
    main()
