"""Train both the feature-based sklearn model and MobileNetV3, then update model registry."""

import subprocess
import sys


def run(script):
    print(f"\n{'=' * 60}\nRunning {script}\n{'=' * 60}")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with code {result.returncode}")


if __name__ == "__main__":
    run("train_model.py")
    run("train_mobilenet.py")
    print("\nAll models trained. Prediction uses an ensemble when both are available.")
