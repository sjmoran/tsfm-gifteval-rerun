import json
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset
from timestral import TimestralForecaster

os.environ.setdefault("GIFT_EVAL", "data/gifteval")

logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)  # pandas 'H'/'T' freq aliases used by gift-eval

# %% --- next cell ---
SIZE = os.environ.get("SIZE", "209M")  # 5M | 28M | 123M | 209M
MODEL_NAME = f"Timestral-{SIZE}"
CHECKPOINT = f"mistralai/{MODEL_NAME}"  # downloaded from the Hugging Face Hub
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs") / MODEL_NAME  # per-config metrics; finished configs are skipped on restart

# Batch budget: max rows (series x (variates + covariates)) and max rows x (context + horizon) steps.
MAX_ROWS = 1024
MAX_TOKENS = 2**24

# %% --- next cell ---
SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

PRETTY_DATASET_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = [
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
METRIC_COLUMNS = [
    "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]",
    "sMAPE[0.5]", "MSIS", "RMSE[mean]", "NRMSE[mean]", "ND[0.5]",
    "mean_weighted_sum_quantile_loss",
]


def all_configs(properties: dict) -> list[tuple[str, str, str]]:
    """(dataset_name, term, config key) for the 97 configs."""
    short_datasets = SHORT_DATASETS.split()
    med_long_datasets = MED_LONG_DATASETS.split()
    configs = []
    for dataset_name in dict.fromkeys(short_datasets + med_long_datasets):
        for term in ["short", "medium", "long"]:
            if term in ("medium", "long") and dataset_name not in med_long_datasets:
                continue
            if "/" in dataset_name:
                ds_key, ds_freq = dataset_name.split("/")
                ds_key = PRETTY_DATASET_NAMES.get(ds_key.lower(), ds_key.lower())
            else:
                ds_key = PRETTY_DATASET_NAMES.get(dataset_name.lower(), dataset_name.lower())
                ds_freq = properties[ds_key]["frequency"]
            configs.append((dataset_name, term, f"{ds_key}/{ds_freq}/{term}"))
    assert len(configs) == 97, len(configs)
    return configs

# %% --- next cell ---
class TimestralPredictor:
    """GluonTS-style predictor: predict(test_data.input) -> list[QuantileForecast].

    Series are sorted by length and batched within the MAX_ROWS / MAX_TOKENS budget; a batch that
    runs out of GPU memory is split in two.
    """

    def __init__(self, forecaster: TimestralForecaster, prediction_length: int, use_past_covariates: bool):
        self.fc = forecaster
        self.prediction_length = prediction_length
        self.use_past_covariates = use_past_covariates
        levels = list(forecaster.quantile_levels)
        self.q_index = [next(i for i, lv in enumerate(levels) if abs(lv - q) < 1e-9) for q in QUANTILE_LEVELS]
        self.forecast_keys = [str(q) for q in QUANTILE_LEVELS] + ["mean"]

    def predict(self, test_data_input) -> list[QuantileForecast]:
        entries = list(test_data_input)
        max_ctx = self.fc.max_context
        targets, covs = [], []
        for e in entries:
            t = np.asarray(e["target"], dtype=np.float32)
            targets.append((t[None] if t.ndim == 1 else t)[:, -max_ctx:])  # (V, T)
            c = e.get("past_feat_dynamic_real") if self.use_past_covariates else None
            if c is not None:
                c = np.asarray(c, dtype=np.float32)
                c = (c[None] if c.ndim == 1 else c)[:, -max_ctx:]
            covs.append(c)
        n_rows = targets[0].shape[0] + (covs[0].shape[0] if covs[0] is not None else 0)

        outputs = [None] * len(entries)
        order = sorted(range(len(entries)), key=lambda i: targets[i].shape[-1])
        start = 0
        while start < len(order):
            # Grow the batch from the shortest remaining series while the padded size fits the budget.
            end = start + 1
            while end < len(order):
                length = targets[order[end]].shape[-1] + self.prediction_length
                if (end + 1 - start) * n_rows > MAX_ROWS or (end + 1 - start) * n_rows * length > MAX_TOKENS:
                    break
                end += 1
            self._predict_batch(order[start:end], targets, covs, outputs)
            start = end

        forecasts = []
        for e, q in zip(entries, outputs):
            # q: (V, H, 9) -> (10, H, V), or (10, H) for a univariate target; "mean" is the median.
            arr = np.concatenate([q, q[..., QUANTILE_LEVELS.index(0.5), None]], axis=-1)
            arr = arr.transpose(2, 1, 0)
            if np.asarray(e["target"]).ndim == 1:
                arr = arr[..., 0]
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=arr,
                    forecast_keys=self.forecast_keys,
                    start_date=e["start"] + np.asarray(e["target"]).shape[-1],
                    item_id=e.get("item_id"),
                )
            )
        return forecasts

    def _predict_batch(self, idx, targets, covs, outputs) -> None:
        try:
            q = self._run(idx, targets, covs)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            q = None  # recurse outside the except block, so the failed attempt's tensors are freed
        if q is None:
            torch.cuda.empty_cache()
            half = len(idx) // 2
            self._predict_batch(idx[:half], targets, covs, outputs)
            self._predict_batch(idx[half:], targets, covs, outputs)
            return
        for j, i in enumerate(idx):
            outputs[i] = q[j]

    def _run(self, idx, targets, covs) -> np.ndarray:
        # Left-pad with NaN to a common length (leading unobserved steps are ignored by the model).
        length = max(targets[i].shape[-1] for i in idx)
        ctx = np.stack([left_pad(targets[i], length) for i in idx])
        past = None
        if covs[idx[0]] is not None:
            past = np.stack([left_pad(covs[i], length) for i in idx])
        f = self.fc.forecast(ctx, self.prediction_length, past_covariates=past)
        return f.quantiles[..., self.q_index].float().cpu().numpy()  # (B, V, H, 9)


def left_pad(x: np.ndarray, length: int) -> np.ndarray:
    pad = length - x.shape[-1]
    if pad == 0:
        return x
    return np.concatenate([np.full((x.shape[0], pad), np.nan, dtype=np.float32), x], axis=-1)

# %% --- next cell ---
def evaluate_config(fc: TimestralForecaster, dataset_name: str, term: str) -> dict:
    ds = Dataset(name=dataset_name, term=term, to_univariate=False)
    predictor = TimestralPredictor(
        fc, ds.prediction_length, use_past_covariates=bool(ds.past_feat_dynamic_real_dim)
    )
    res = evaluate_model(
        predictor,
        test_data=ds.test_data,
        metrics=METRICS,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(ds.freq),
    )
    return {k: float(res[k].iloc[0]) for k in res.columns}

# %% --- next cell ---
# Runs the 97 configs. Each config's metrics go to OUTPUT_DIR/per_config/ as soon as it is done.
properties = json.load(open("dataset_properties.json"))
fc = TimestralForecaster.from_pretrained(CHECKPOINT, device=DEVICE)

(OUTPUT_DIR / "per_config").mkdir(parents=True, exist_ok=True)
for dataset_name, term, key in [c for c in all_configs(properties) if not os.environ.get('ONLY') or c[2] in os.environ['ONLY'].split()]:
    path = OUTPUT_DIR / "per_config" / (key.replace("/", "__") + ".json")
    if path.exists():
        continue
    metrics = evaluate_config(fc, dataset_name, term)
    path.write_text(json.dumps({"dataset": key, "metrics": metrics}, indent=1))
    print(f"{key}: MASE={metrics['MASE[0.5]']:.4f} CRPS={metrics['mean_weighted_sum_quantile_loss']:.4f}")

# %% --- next cell ---
# Writes results/<MODEL_NAME>/all_results.csv in the leaderboard format.
rows = []
for p in sorted((OUTPUT_DIR / "per_config").glob("*.json")):
    r = json.loads(p.read_text())
    ds_key = r["dataset"].split("/")[0]
    rows.append(
        {
            "dataset": r["dataset"],
            "model": MODEL_NAME,
            **{f"eval_metrics/{k}": r["metrics"][k] for k in METRIC_COLUMNS},
            "domain": properties[ds_key]["domain"],
            "num_variates": properties[ds_key]["num_variates"],
        }
    )
results = pd.DataFrame(rows).sort_values("dataset")

results_dir = Path("../results") / MODEL_NAME
results_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(results_dir / "all_results.csv", index=False)
print(f"Saved {len(results)} configs to {results_dir / 'all_results.csv'}")

# %% --- next cell ---
# Leaderboard aggregates: MASE and CRPS of each config divided by Seasonal Naive, then the
# geometric mean over the 97 configs.
seasonal_naive = pd.read_csv("../results/Seasonal_Naive/all_results.csv").set_index("dataset")
res = results.set_index("dataset").loc[seasonal_naive.index]


def gmean(x: pd.Series) -> float:
    return float(math.exp(x.map(math.log).mean()))


for name, col in [("MASE", "eval_metrics/MASE[0.5]"), ("CRPS", "eval_metrics/mean_weighted_sum_quantile_loss")]:
    print(f"{name}: {gmean(res[col] / seasonal_naive[col]):.4f}")
