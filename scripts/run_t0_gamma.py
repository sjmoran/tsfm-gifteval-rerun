import json
import os
import warnings

import torch
from dotenv import load_dotenv

# GIFT-Eval's GluonTS + pandas stack emits deprecation / "IProgress not found"
# noise that doesn't affect the forecasts or metrics — silence it for a clean run.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")

load_dotenv()
os.environ.setdefault("GIFT_EVAL", "data/gifteval")  # reads GIFT_EVAL from a local .env, if present

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Weights download and cache from the Hugging Face Hub on first use, so re-runs are offline.
MODEL_PATH = "theforecastingcompany/t0-gamma"
MODEL_NAME = "t0-gamma"
CONTEXT_LENGTH = 8192  # most recent observations kept per series; longer series are trimmed to this window
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH_SIZE = 64  # series per forward; halved automatically on OOM
CROSS_INFERENCE = 64  # series of one window forecast together; 0 forecasts each on its own

# %% --- next cell ---
# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
# Two quick-demo defaults: us_births/D (clean univariate daily, ~5x better than
# Seasonal-Naive, also the sanity-plot series below) and bizitobs_l2c/5T
# (multivariate, exercises the medium/long terms). Uncomment the full lists for all 97 configs.
short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

dataset_properties_map = json.load(open("dataset_properties.json"))
if os.environ.get("ONLY_DS"):  # smoke-test filter (not in official notebook)
    all_datasets = [d for d in all_datasets if d in os.environ["ONLY_DS"].split()]

# %% --- next cell ---
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
]

# %% --- next cell ---
from t0 import T0Forecaster
from t0.evaluation import T0Predictor

# %% --- next cell ---
import logging


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

# %% --- next cell ---
import csv
import time

from gift_eval.data import Dataset
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

model = T0Forecaster.from_pretrained(MODEL_PATH).to(DEVICE).eval()
print(f"Loaded t0-gamma from {MODEL_PATH} ({sum(p.numel() for p in model.parameters()):,} parameters)")

output_dir = f"../results/{MODEL_NAME}"
os.makedirs(output_dir, exist_ok=True)
csv_file_path = os.path.join(output_dir, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "dataset",
            "model",
            "eval_metrics/MSE[mean]",
            "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
        ]
    )

for ds_num, ds_name in enumerate(all_datasets):
    print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
    for term in ["short", "medium", "long"]:
        if term in ("medium", "long") and ds_name not in med_long_datasets.split():
            continue

        if "/" in ds_name:
            ds_key, ds_freq = ds_name.split("/")
            ds_key = pretty_names.get(ds_key.lower(), ds_key.lower())
        else:
            ds_key = pretty_names.get(ds_name.lower(), ds_name.lower())
            ds_freq = dataset_properties_map[ds_key]["frequency"]
        ds_config = f"{ds_key}/{ds_freq}/{term}"

        # Initialize the dataset; multivariate datasets are split into
        # univariate series (each variate is forecast independently).
        to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        season_length = get_seasonality(dataset.freq)
        print(f"Dataset size: {len(dataset.test_data)}")

        predictor = T0Predictor(
            model,
            prediction_length=dataset.prediction_length,
            quantile_levels=QUANTILE_LEVELS,
            context_length=CONTEXT_LENGTH,
            batch_size=BATCH_SIZE,
            show_progress=True,
            trim_leading_missing=True,
            symmetric_averaging=True,
            clip_nonnegative=True,
            snap_integers=True,
            cross_inference=CROSS_INFERENCE,
        )

        t0 = time.monotonic()
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        elapsed = time.monotonic() - t0

        # evaluate_model returns a one-row DataFrame; .iloc[0] reads the row
        # by position (plain [0] is deprecated for label/position ambiguity).
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    ds_config,
                    MODEL_NAME,
                    res["MSE[mean]"].iloc[0],
                    res["MSE[0.5]"].iloc[0],
                    res["MAE[0.5]"].iloc[0],
                    res["MASE[0.5]"].iloc[0],
                    res["MAPE[0.5]"].iloc[0],
                    res["sMAPE[0.5]"].iloc[0],
                    res["MSIS"].iloc[0],
                    res["RMSE[mean]"].iloc[0],
                    res["NRMSE[mean]"].iloc[0],
                    res["ND[0.5]"].iloc[0],
                    res["mean_weighted_sum_quantile_loss"].iloc[0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                ]
            )

        print(
            f"{ds_config}: MASE={res['MASE[0.5]'].iloc[0]:.4f} "
            f"CRPS={res['mean_weighted_sum_quantile_loss'].iloc[0]:.4f} ({elapsed:.1f}s)"
        )

# %% --- next cell ---
import pandas as pd

df = pd.read_csv(csv_file_path)
df
