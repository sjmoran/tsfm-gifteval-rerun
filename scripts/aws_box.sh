#!/usr/bin/env bash
# Runs on the AWS box: build three venvs, fetch GIFT-Eval data, run the three jobs in parallel on one GPU.
set -euo pipefail
cd ~/tb
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH
export GIFT_EVAL=$HOME/gifteval HF_HUB_DISABLE_PROGRESS_BARS=1
uvx --from huggingface_hub hf download Salesforce/GiftEval --repo-type dataset --local-dir $GIFT_EVAL >/dev/null
cd rerun
uv venv -q -p 3.11 .venv-t0 && uv pip install -q -p .venv-t0 -e ../gift-eval -e ./tfc-t0 "pyarrow<18" python-dotenv
uv venv -q -p 3.11 .venv-ts && uv pip install -q -p .venv-ts -e ../gift-eval -e ./timestral "pyarrow<18" python-dotenv
uv venv -q -p 3.11 .venv-tb && uv pip install -q -p .venv-tb -e "./trustworthyAI/TabbyTSFM[forecast]" "pyarrow<18"
echo VENVS_DONE
(cd t0 && ../.venv-t0/bin/python run_t0_gamma_noadj.py > ../log_t0_gamma_noadj.txt 2>&1; echo "exit $?" >> ../log_t0_gamma_noadj.txt) &
(cd ts && SIZE=209M ../.venv-ts/bin/python run_timestral.py > ../log_timestral.txt 2>&1; echo "exit $?" >> ../log_timestral.txt) &
(cd trustworthyAI/TabbyTSFM && ../../.venv-tb/bin/python benchmarks/forecasting/gift_eval/evaluate.py --mode zs \
  --pretrain_ckpt paris-noah/Tabby-Pretrain --gift_eval_repo $HOME/tb/gift-eval \
  --dataset_properties benchmarks/forecasting/gift_eval/dataset_properties.json --context_length 8096 \
  --device cuda --datasets all --out_csv tabby_pretrain_full > ../../log_tabby.txt 2>&1; echo "exit $?" >> ../../log_tabby.txt) &
wait; echo ALL_DONE
