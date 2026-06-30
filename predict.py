"""
Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys

from model_loader import predict_from_path


def predict(image_path: str) -> float:
    return predict_from_path(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"{predict(sys.argv[1]):.4f}")
    except Exception as e:
        print(f"Error predicting image: {e}", file=sys.stderr)
        sys.exit(1)
