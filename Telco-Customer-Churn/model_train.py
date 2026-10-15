

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET_DIR = Path("./dataset")
DEFAULT_MODEL_DIR = Path("./models")
DEFAULT_TARGET = "MonthlyCharges"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42



def load_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    """Find the first CSV in the folder and load it."""
    dataset_dir = Path(dataset_dir)
    csv_files = sorted(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Hmm, no CSV file in {dataset_dir.resolve()}"
        )

    # Just grab the first CSV we find.
    # If the folder has more than one, point at the exact filename instead.
    dataset_path = csv_files[0]

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Original shape: {df.shape}")

    return df


def clean_dataframe(df, target=DEFAULT_TARGET):
    """Do the same light cleaning we did before modeling."""
    # customerID is unique per row — no use for modeling.
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # TotalCharges comes in as a string, so convert it to numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(),
            errors="coerce",
        )
        # Missing TotalCharges usually means a brand-new customer,
        # so filling with 0 makes sense.
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop any duplicate rows.
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Make sure the target we want is actually there.
    if target not in df.columns:
        raise ValueError(
            f"Can't find target '{target}'. "
            f"Columns available: {df.columns.tolist()}"
        )

    # Coerce the target to numeric and drop rows where it didn't work.
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df.dropna(subset=[target], inplace=True)

    print(f"Shape after cleaning: {df.shape}")
    return df


def split_features_and_target(df, target=DEFAULT_TARGET):
    """Pull X and y apart."""
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\nTarget column: {target}")
    print(f"Input features before encoding: {X.shape[1]}")

    return X, y


def build_preprocessor(numeric_features, categorical_features):
    """Numeric -> median + scale. Categorical -> mode + one-hot."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipeline(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def evaluate_model(model, X_data, y_data, name):
    """Predict and print the usual regression metrics."""
    predictions = model.predict(X_data)

    mse = mean_squared_error(y_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_data, predictions)
    r2 = r2_score(y_data, predictions)

    print(f"\n--- {name} Results ---")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    return predictions


def inspect_coefficients(model_pipeline, top_n=15):
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    regressor = model_pipeline.named_steps["regressor"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": regressor.coef_,
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values(
        "absolute_coefficient", ascending=False, inplace=True
    )

    print(f"\nTop {top_n} features by absolute coefficient:")
    print(coefficients.head(top_n).to_string(index=False))
    print(f"\nIntercept: {regressor.intercept_:.4f}")

    return coefficients




def run_regression_step(
    dataset_dir=DEFAULT_DATASET_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    target=DEFAULT_TARGET,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Run the whole Step 3 flow:
      load -> clean -> split -> preprocess -> train -> evaluate -> save.

    Returns a dict with the fitted pipeline, metrics, and file paths
    so you can reuse them downstream.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean
    df = load_dataset(dataset_dir)
    df = clean_dataframe(df, target=target)

    # 2. Split features and target
    X, y = split_features_and_target(df, target=target)

    # 3. Train/test split — do this BEFORE fitting any preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    # 4. Figure out which columns are numeric vs. categorical.
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 5. Build preprocessor + full pipeline.
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model_pipeline = build_model_pipeline(preprocessor)

    # 6. Train.
    model_pipeline.fit(X_train, y_train)
    print("\nTraining finished.")

    # 7. Evaluate.
    train_predictions = evaluate_model(
        model_pipeline, X_train, y_train, "Training"
    )
    test_predictions = evaluate_model(
        model_pipeline, X_test, y_test, "Testing"
    )

    # 8. Save the trained pipeline.
    model_path = model_dir / "multiple_linear_regression.joblib"
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel saved to: {model_path.resolve()}")

    # 9. Save test predictions so we can eyeball them later.
    results = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": test_predictions,
            "Error": y_test.to_numpy() - test_predictions,
        },
        index=y_test.index,
    ).sort_index()

    results_path = model_dir / "test_predictions.csv"
    results.to_csv(results_path, index=False)
    print(f"Test predictions saved to: {results_path.resolve()}")

    # 10. Quick peek at feature importance.
    coefficients = inspect_coefficients(model_pipeline)

    print("\nStep 3 done.")

    return {
        "model": model_pipeline,
        "coefficients": coefficients,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_path": model_path,
        "results_path": results_path,
    }


if __name__ == "__main__":
    run_regression_step()