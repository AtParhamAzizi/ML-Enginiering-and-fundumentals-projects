"""Train an Ames house-price model and write a Kaggle-style submission.

Usage:
    python train.py
    python train.py --data-dir datasets --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/house_price_model.joblib"))
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {args.data_dir.resolve()}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "SalePrice"
    features = [column for column in train.columns if column != target]
    numeric_columns = train[features].select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]

    x_train, x_valid, y_train, y_valid = train_test_split(
        train[features], train[target], test_size=0.2, random_state=42
    )
    model = build_model(numeric_columns, categorical_columns)
    model.fit(x_train, np.log1p(y_train))
    validation_predictions = np.expm1(model.predict(x_valid))
    rmsle = mean_squared_error(np.log1p(y_valid), np.log1p(np.maximum(validation_predictions, 0))) ** 0.5
    print(f"Validation RMSLE: {rmsle:.4f}")

    model.fit(train[features], np.log1p(train[target]))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    predictions = np.maximum(0, np.expm1(model.predict(test[features])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"Id": test["Id"], target: predictions})
    output_path = args.output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
