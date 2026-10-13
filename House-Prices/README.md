# House Price Prediction, Without the Mystery

This project turns the Ames housing data into a small, understandable prediction workflow. Instead of treating the CSV as a black box, it asks the same questions a person might ask about a home: how much space does it have, what neighborhood is it in, how well is it built, and how old is it?

The model uses those details to estimate a home's sale price, while the visualization helps us see the patterns behind the predictions.

## AI assistance

The website interface (`web/`) and its Python backend (`app.py`) were written with AI assistance. AI assistance was also used to create the training, prediction, and visualization scripts and this README, so it would not be accurate to describe only the website as AI-written.

## What is included

```text
.
├── datasets/
│   ├── train.csv                  labeled homes and sale prices
│   ├── test.csv                   homes waiting for predictions
│   ├── sample_submission.csv      expected submission format
│   └── data_description.txt       column definitions
├── train.py                       train, evaluate, and save the model
├── predict.py                     load a checkpoint and predict new rows
├── visualize.py                   create the visual data story
├── app.py                         serve the local website and prediction API
├── web/                           website HTML, CSS, and JavaScript
├── requirements.txt               Python dependencies
└── outputs/                       generated model files and results
```

## Start here

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell activation is blocked, use the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## The complete workflow

### Try the website

Train the model first if `outputs/house_price_model.joblib` does not exist, then start the local website:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8000` in your browser. Fill in your home's size, neighborhood, quality, age, bathrooms, and garage details, then click **Estimate my home's price**. Expand the optional advanced section to edit the remaining dataset features.

Keep the backend terminal running. Do not open `web/index.html` directly or use Live Server: those do not provide the Python prediction API. If the page reports a connection error, start `app.py` and open the address printed in its terminal.

The backend calls the shared prediction function in `predict.py` and reuses the saved preprocessing pipeline. Unchanged fields use training-data medians or most common categories; the record ID is fixed to the training median. Numeric inputs are limited to observed training ranges. The page shows a historical Ames estimate, not a current market appraisal. Requests are not saved, and the server binds only to your computer. Stop it with `Ctrl+C`.

Use `python app.py --port 8001` to choose another port. Only load checkpoints you trust: Joblib files can execute code when loaded. This lightweight server is for local use, not public deployment.

Website files: `app.py` (backend) and `web/` (HTML, CSS, JavaScript). No extra web framework or frontend build step is required.

### 1. Create the visual story

```powershell
python visualize.py
```

This creates `outputs/house_price_story.png`, a presentation-ready summary of the data:

![House price data story](outputs/house_price_story.png)

The image highlights the price distribution, neighborhood differences, quality scores, living area, construction year, and plain-English takeaways.

### 2. Train and save the model

```powershell
python train.py
```

Training fills missing values, encodes text columns, trains a `HistGradientBoostingRegressor` on `log1p(SalePrice)`, prints a validation RMSLE score, saves `outputs/house_price_model.joblib`, and writes `outputs/submission.csv`.

### 3. Predict from the saved checkpoint

Once the checkpoint exists, predict from `./datasets/test.csv` without retraining:

```powershell
python predict.py
```

This writes `outputs/test_predictions.csv`. To choose another input or output file:

```powershell
python predict.py --input datasets/test.csv --checkpoint outputs/house_price_model.joblib --output outputs/test_predictions.csv
```

## Reading the outputs

| File | What it means |
| --- | --- |
| `outputs/house_price_story.png` | Visual explanation of the training data |
| `outputs/house_price_model.joblib` | Saved preprocessing steps and fitted model |
| `outputs/submission.csv` | Predictions formatted with `Id` and `SalePrice` |
| `outputs/test_predictions.csv` | Predictions from the standalone checkpoint runner |

The prediction CSVs contain one row per test home, preserve the original `Id`, and contain a non-negative `SalePrice` estimate.

## A human note about the model

Homes are not tidy rows in a spreadsheet. A blank garage field may mean “this home has no garage,” not “someone forgot to answer.” This baseline handles missing values consistently, but stays readable so it is easy to improve with feature engineering, cross-validation, outlier review, or a model designed specifically for mixed housing data.

## Good next experiments

- Add total finished area and total bathroom features.
- Compare several models with cross-validation.
- Investigate unusually large homes with surprisingly low prices.
- Track each experiment's score and settings.

## Dataset

The files use the familiar Kaggle **House Prices: Advanced Regression Techniques** format. Keeping the data under `./datasets` makes the project reproducible and runnable offline once dependencies are installed.
