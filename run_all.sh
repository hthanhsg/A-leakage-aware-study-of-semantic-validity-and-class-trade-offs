#!/usr/bin/env bash
# Run all experiments PHASE by PHASE. Each phase can be run separately (recommended).
# Every phase is resumable: re-running a command skips cells that already have results.
set -euo pipefail
cd "$(dirname "$0")"
REAL="nf_unsw_v3 nf_ton_v3 nf_bot_v3 nf_cic2018_v3 cic2017_fixed"
PHASE=${1:-all}

if [[ $PHASE == A || $PHASE == all ]]; then   # PHASE A: data preparation + audit
  python scripts/prepare_data.py --datasets $REAL
fi
if [[ $PHASE == B || $PHASE == all ]]; then   # PHASE B: protocol-induced inflation (cheap)
  python scripts/run_grid.py --exp configs/phaseB_protocol.yaml
  python scripts/analyze.py --results runs/phaseB_protocol/results.jsonl --out runs/phaseB_protocol/analysis --control smote
fi
if [[ $PHASE == C || $PHASE == all ]]; then   # PHASE C: tune guidance on VAL (per dataset)
  for ds in $REAL; do
    python scripts/tune_guidance.py --exp configs/phaseD_main.yaml --dataset $ds --trials 20 --out runs \
      --set aug.validaug.pool_cap=10000 aug.validaug.oversample=1.5
  done
  python scripts/collect_tuning.py --tune-dir runs --apply configs/phaseD_main.yaml configs/phaseE_ablation.yaml configs/phaseF_generalization.yaml
  echo ">> per-dataset guidance parameters written to aug_per_dataset of the phase D/E/F configs"
fi
if [[ $PHASE == D || $PHASE == all ]]; then   # PHASE D: main comparison (heaviest; split GPUs by dataset)
  # Example with 2 GPUs:  CUDA_VISIBLE_DEVICES=0 ... --only-datasets nf_unsw_v3 nf_ton_v3 &
  python scripts/run_grid.py --exp configs/phaseD_main.yaml
  python scripts/analyze.py --results runs/phaseD_main/results.jsonl --out runs/phaseD_main/analysis
fi
if [[ $PHASE == E || $PHASE == all ]]; then   # PHASE E: ablation + sensitivity
  python scripts/run_grid.py --exp configs/phaseE_ablation.yaml
  python scripts/analyze.py --results runs/phaseE_ablation/results.jsonl --out runs/phaseE_ablation/analysis
  for lc in 0.25 0.5 2.0; do
    python scripts/run_grid.py --exp configs/phaseE_ablation.yaml --only-methods validaug --seeds 0 1 2 \
      --set aug.validaug.lambda_c=$lc --out runs/phaseE_sens/lambda_c_$lc
  done
  for lb in 0.05 0.5 1.0; do
    python scripts/run_grid.py --exp configs/phaseE_ablation.yaml --only-methods validaug --seeds 0 1 2 \
      --set aug.validaug.lambda_b=$lb --out runs/phaseE_sens/lambda_b_$lb
  done
  for beta in 2 5 20; do
    python scripts/run_grid.py --exp configs/phaseE_ablation.yaml --only-methods validaug diffusion --seeds 0 1 2 \
      --set aug.budget.beta=$beta --out runs/phaseE_sens/beta_$beta
  done
fi
if [[ $PHASE == F || $PHASE == all ]]; then   # PHASE F: holdout + cross-dataset
  python scripts/run_grid.py --exp configs/phaseF_generalization.yaml
  python scripts/analyze.py --results runs/phaseF_generalization/results.jsonl --out runs/phaseF_generalization/analysis
fi
