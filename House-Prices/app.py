"""Serve the local house-price website: python app.py."""

from __future__ import annotations

import argparse
import json
import logging
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from predict import load_checkpoint, predict_prices


ROOT = Path(__file__).resolve().parent
LABELS = {
    "Neighborhood": "Neighborhood", "GrLivArea": "Living area (sq ft)",
    "OverallQual": "Overall quality (1–10)", "OverallCond": "Overall condition (1–10)",
    "YearBuilt": "Year built", "YearRemodAdd": "Last remodel year",
    "LotArea": "Lot area (sq ft)", "BedroomAbvGr": "Bedrooms above ground",
    "FullBath": "Full bathrooms", "HalfBath": "Half bathrooms",
    "GarageCars": "Garage capacity (cars)", "GarageArea": "Garage area (sq ft)",
    "TotalBsmtSF": "Basement area (sq ft)", "HouseStyle": "House style",
    "BldgType": "Building type", "CentralAir": "Central air conditioning",
    "YrSold": "Sale year", "MoSold": "Sale month",
}
CORE = list(LABELS)


class HouseEstimator:
    def __init__(self, checkpoint: Path) -> None:
        self.bundle = load_checkpoint(checkpoint)
        training = pd.read_csv(ROOT / "datasets/train.csv")
        self.fields = []
        self.defaults = {}
        for name in self.bundle["features"]:
            column = training[name]
            numeric = pd.api.types.is_numeric_dtype(column)
            default = float(column.median()) if numeric else str(column.mode().iloc[0])
            self.defaults[name] = default
            if name == "Id":
                continue
            field = {
                "name": name, "label": LABELS.get(name, name),
                "type": "number" if numeric else "select", "default": default,
                "advanced": name not in CORE,
            }
            if numeric:
                field.update(min=float(column.min()), max=float(column.max()))
            else:
                field["options"] = sorted(column.dropna().unique().tolist())
            self.fields.append(field)
        self.fields.sort(key=lambda field: CORE.index(field["name"]) if field["name"] in CORE else len(CORE))

    def estimate(self, values: dict) -> float:
        if not isinstance(values, dict):
            raise ValueError("Please send a JSON object with house parameters.")
        schema = {field["name"]: field for field in self.fields}
        unknown = set(values) - set(schema)
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
        row = self.defaults.copy()
        for name, value in values.items():
            field = schema[name]
            if value is None:
                row[name] = np.nan
            elif field["type"] == "number":
                if isinstance(value, bool):
                    raise ValueError(f"{field['label']} must be a number.")
                try:
                    number = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"{field['label']} must be a number.") from None
                if not math.isfinite(number) or not field["min"] <= number <= field["max"]:
                    raise ValueError(f"{field['label']} must be between {field['min']:g} and {field['max']:g}.")
                row[name] = number
            else:
                if value not in field["options"]:
                    raise ValueError(f"Choose a listed option for {field['label']}.")
                row[name] = value
        return float(predict_prices(self.bundle, pd.DataFrame([row]))[0])


def make_handler(estimator: HouseEstimator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_content(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, payload: dict) -> None:
            self.send_content(status, json.dumps(payload, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/api/fields":
                self.send_json(200, {"fields": estimator.fields})
                return
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            if route not in files:
                self.send_json(404, {"error": "Page not found."})
                return
            filename, content_type = files[route]
            self.send_content(200, (ROOT / "web" / filename).read_bytes(), content_type)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/predict":
                self.send_json(404, {"error": "Endpoint not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("Request must contain at most 32 KB of JSON.")
                values = json.loads(self.rfile.read(length))
                self.send_json(200, {"price": estimator.estimate(values)})
            except (ValueError, TypeError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                logging.exception("Prediction failed")
                self.send_json(500, {"error": "Prediction failed. Check the server terminal for details."})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/house_price_model.joblib")
    args = parser.parse_args()
    estimator = HouseEstimator(args.checkpoint)
    with HTTPServer(("127.0.0.1", args.port), make_handler(estimator)) as server:
        print(f"House price website: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
