import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datacleaning import TelcoDataCleaner
from loading_model import load_checkpoint
from model_train import build_model_pipeline, build_preprocessor

DEFAULT_DATASET = Path("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_TARGET = "MonthlyCharges"

def clean_data(dataset_path, cleaned_path):
    cleaner = TelcoDataCleaner(missing_strategy="zero")
    cleaner.load_data(dataset_path)
    cleaner.drop_useless_columns()
    cleaner.fix_data_types()
    cleaner.handle_missing_values()
    cleaner.remove_duplicates()
    cleaner.df.to_csv(cleaned_path, index=False)
    return cleaner.df

def evaluate(model, features, target):
    predictions = model.predict(features)
    mse = mean_squared_error(target, predictions)
    return {"r2": float(r2_score(target, predictions)), "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(target, predictions))}, predictions

def run_pipeline(dataset_path=DEFAULT_DATASET, model_dir=DEFAULT_MODEL_DIR, target=DEFAULT_TARGET, train_fraction=0.8):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    dataset_path, model_dir = Path(dataset_path), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = clean_data(dataset_path, model_dir / "cleaned_data.csv")
    if target not in frame.columns:
        raise ValueError(f"Target '{target}' is not present in the dataset")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    split = int(len(frame) * train_fraction)
    train, test = frame.iloc[:split], frame.iloc[split:]
    x_train, y_train = train.drop(columns=[target]), train[target]
    x_test, y_test = test.drop(columns=[target]), test[target]
    numeric = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = x_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    model = build_model_pipeline(build_preprocessor(numeric, categorical))
    model.fit(x_train, y_train)
    checkpoint = model_dir / "multiple_linear_regression.joblib"
    joblib.dump(model, checkpoint)
    loaded = load_checkpoint(checkpoint)
    metrics, predictions = evaluate(loaded, x_test, y_test)
    pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions, "Error": y_test.to_numpy() - predictions}).to_csv(model_dir / "evaluation_results.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Cleaned data: {model_dir / 'cleaned_data.csv'}")
    print(f"Training rows: {len(train)} (first {train_fraction:.0%})")
    print(f"Evaluation rows: {len(test)} (last {1-train_fraction:.0%})")
    print(f"Checkpoint: {checkpoint}")
    print(f"Evaluation: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    return {"model_path": checkpoint, "metrics": metrics}

def main():
    parser = argparse.ArgumentParser(description="Run the Telco regression pipeline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    run_pipeline(args.dataset, args.model_dir, args.target, args.train_fraction)

if __name__ == "__main__":
    main()
