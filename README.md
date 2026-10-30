# Rerunning three time-series foundation models on GIFT-Eval

Code and per-task results behind my post comparing Timestral, t0, Tabby and Morpheus-2.0. I reran the three open models on all 97 GIFT-Eval tasks to check that their leaderboard entries reproduce.

| Model | My CRPS / MASE | Official CRPS / MASE | Notes |
|---|---|---|---|
| t0-gamma | 0.4520 / 0.6609 | 0.4520 / 0.6608 | every task within 0.25% |
| t0-gamma, inference adjustments off | 0.4535 / 0.6640 | | adjustments are worth 0.0015 CRPS |
| Tabby-Pretrain, `--min_forecast_span 128` | 0.4827 / 0.7040 | 0.4827 / 0.7040 | every task within 0.4% |
| Tabby-Pretrain, script default (span 0) | 0.4923 / 0.7192 | | 15 tasks more than 5% worse |
| Timestral-209M | in progress | 0.4460 / 0.6575 | 79/97 tasks so far, each within 1e-6 of official |

Scores are geometric means over the 97 tasks of CRPS and MASE, each normalised by seasonal naive, as on the leaderboard.

## Layout

- `scripts/run_t0_gamma.py`, `scripts/run_t0_gamma_noadj.py`: t0-gamma with and without the five inference adjustments from the t0 notebook.
- `scripts/run_timestral.py`: Timestral through the GIFT-Eval predictor interface. On CUDA out-of-memory it halves the batch and retries.
- `scripts/aws_box.sh`: builds the three environments and runs all jobs in parallel on one GPU.
- `results/`: per-task metrics for each run.
- `figures/`: the scripts that draw the post's charts. `make_charts.py` and `pairwise.py` read the leaderboard results from a GIFT-Eval checkout (`GIFT_EVAL_RESULTS`, default `gift-eval/results`).

## Reproducing

The runs used one NVIDIA A10G (24 GB) on an AWS g5.8xlarge. Clone the upstream code at the commits I used, next to this repo's `scripts/`:

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval && git -C gift-eval checkout 3f77b8b
git clone https://github.com/theforecastingcompany/tfc-t0 && git -C tfc-t0 checkout 5896313
git clone https://github.com/mistralai/timestral && git -C timestral checkout a534bdb
git clone https://github.com/huawei-noah/trustworthyAI && git -C trustworthyAI checkout 8675af0
```

Download the data with `hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gifteval` and set `GIFT_EVAL` to that directory. `scripts/aws_box.sh` shows the environment set-up and the exact commands. For Timestral, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Tabby's evaluation script defaults to `--min_forecast_span 0`. Its model card uses 128, and only 128 reproduces the leaderboard entry:

```bash
python benchmarks/forecasting/gift_eval/evaluate.py --mode zs --pretrain_ckpt paris-noah/Tabby-Pretrain \
  --gift_eval_repo ../gift-eval --dataset_properties benchmarks/forecasting/gift_eval/dataset_properties.json \
  --context_length 8096 --device cuda --datasets all --min_forecast_span 128 --out_csv tabby_pretrain_span128
```

## Figures

```bash
uv run --no-project --with matplotlib python figures/make_arch.py
uv run --no-project --with pandas --with numpy --with matplotlib python figures/make_charts.py
```
