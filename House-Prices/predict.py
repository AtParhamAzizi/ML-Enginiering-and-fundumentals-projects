"""Load a saved house-price checkpoint and predict prices for a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Run `python train.py` first.")
    return joblib.load(path)


def predict_prices(bundle: dict, data: pd.DataFrame) -> np.ndarray:
    features = bundle["features"]
    missing_features = sorted(set(features) - set(data.columns))
    if missing_features:
        raise ValueError(f"Input is missing required features: {', '.join(missing_features)}")
    predictions = np.maximum(0, np.expm1(bundle["model"].predict(data[features])))
    if not np.isfinite(predictions).all():
        raise ValueError("The model could not produce a finite price for these inputs.")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/house_price_model.joblib"))
    parser.add_argument("--input", type=Path, default=Path("datasets/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_predictions.csv"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Run `python train.py` first."
        )
    if not args.input.exists():
        raise FileNotFoundError(f"Input data not found at {args.input}")

    bundle = load_checkpoint(args.checkpoint)
    data = pd.read_csv(args.input)
    predictions = predict_prices(bundle, data)
    result = pd.DataFrame({"Id": data["Id"], "SalePrice": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
