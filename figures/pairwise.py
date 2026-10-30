"""Pairwise GIFT-Eval comparison from official per-task CSVs: gmean ratio of
seasonal-naive-normalised CRPS/MASE, task-bootstrap 95% CI, and per-task win counts."""
import os, sys, numpy as np, pandas as pd
R = os.environ.get("GIFT_EVAL_RESULTS", "gift-eval/results")
C, M = "eval_metrics/mean_weighted_sum_quantile_loss", "eval_metrics/MASE[0.5]"
def load(m): return pd.read_csv(f"{R}/{m}/all_results.csv").set_index("dataset")
sn = load("seasonal_naive")
models = ["Timestral-209M", "t0-gamma", "Morpheus-2.0", "TimesFM-3", "Chronos-2"]
norm = {m: load(m)[[C, M]].div(sn[[C, M]]) for m in models}
rng = np.random.default_rng(0)
for a, b in [("Timestral-209M", "t0-gamma"), ("Timestral-209M", "Morpheus-2.0"), ("t0-gamma", "Morpheus-2.0")]:
    for col, name in [(C, "CRPS"), (M, "MASE")]:
        lr = np.log(norm[a][col] / norm[b][col]).dropna().values  # <0 means a better
        boots = [np.exp(rng.choice(lr, len(lr)).mean()) for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{a:>15} vs {b:<13} {name}: ratio {np.exp(lr.mean()):.3f} [{lo:.3f},{hi:.3f}]  {a} wins {(lr<0).sum()}/{len(lr)}")
# per-domain gmean for the three
dom = load("t0-gamma")["domain"]
tab = pd.DataFrame({m: np.exp(np.log(norm[m][C]).groupby(dom).mean()) for m in models[:3]})
tab["n"] = dom.value_counts()
print("\nCRPS by domain (gmean, normalised):\n", tab.round(3).to_string())
