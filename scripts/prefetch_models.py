"""Download every model's pretrained weights into the local torch hub cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdream.models import AVAILABLE_MODELS, get_model


def main() -> None:
    for name in AVAILABLE_MODELS:
        print(f"Fetching {name}...")
        get_model(name)
    print("All model weights cached.")


if __name__ == "__main__":
    main()
