set -euo pipefail
SPLITS="${SPLITS:-splits/controlled_v2.json}"
COMMON=(--cache_dir cache_256 --splits_path "$SPLITS" --rho 0.95)
for s in 0 1 2; do
  python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method erm
  python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method rwg
  python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method rwg  --weight_decay 1.0
  python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method gdro --eta_q 0.01
  python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method gdro --eta_q 0.01 --weight_decay 1.0
  for T in 25 50 100; do
    python scripts/train.py "${COMMON[@]}" --data_seed "$s" --method jtt --jtt_T "$T" --jtt_lambda 20
  done
done
python scripts/dfr.py --runs "runs/controlled_rho0.95_ds[0-2]_ts[0-2]_erm_resnet50_*"
mkdir -p results/pilot
python scripts/summarize_validation.py --runs "runs/controlled_rho0.95_*" --out_csv results/pilot/methods_validation.csv