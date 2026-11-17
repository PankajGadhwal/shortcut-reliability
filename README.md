# Reliability of Shortcut Mitigation under Spurious Correlations

ES 678 course project. This repository contains the data pipeline and training
code for a controlled study of whether shortcut-mitigation methods reliably
improve worst-group accuracy on Waterbirds as the strength of the spurious
correlation (rho) varies.

**Status:** data pipeline, image cache, EDA, training for ERM, RWG, Group DRO
and JTT with logit logging, DFR on saved embeddings, model selection (S1 to S4)
and sealed final evaluation.
No experimental results are included anywhere in this repository yet.

## Layout

```
srl/                    the package
  constants.py          group encoding, official counts, normalization
  data.py               pool loading, count verification, controlled nested splits
  cache.py              uint8 image cache, GPU image bank, GPU augmentation
  samplers.py           resumable uniform and group-balanced batch streams
  models.py             torchvision ResNets with a fresh head; logits + features
  objectives.py         ERM, RWG, Group DRO, JTT (per-method loss and sampler)
  dfr.py                DFR last-layer retraining on saved embeddings
  metrics.py            group accuracies, WGA, Acc(rho'), shortcut gap, ECE/NLL/Brier
  train.py              training loop, logging, checkpoints, CLI config
  selection.py          selection protocols S1 to S4 from validation logits
  evaluate.py           final evaluation; the only code that reads sealed test logits
  testing.py            synthetic data for tests only
scripts/                check_env, download, make_splits, build_cache, eda, train,
                        inspect_run, dfr, summarize_validation, pilot_methods.sh,
                        final_eval
tests/                  pytest suite (CPU, under a minute)
notebooks/              Colab driver notebook
splits/                 controlled_v1.json goes here; commit it
```

## Local setup (VS Code, own GPU)

Run every command from the repository root, inside a virtual environment.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
```

Install PyTorch with CUDA using the command from the selector at
pytorch.org ("Get Started", choose Pip, your OS and a CUDA version your
driver supports), then the rest:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py     # must report cuda_available True
pytest -q                       # 23+ tests, about a minute
```

## Pipeline

```bash
python scripts/download_waterbirds.py --dest data
python scripts/make_splits.py --data_root data/waterbird_complete95_forest2water2
python scripts/build_cache.py --data_root data/waterbird_complete95_forest2water2 --out_dir cache_256
python scripts/eda.py --cache_dir cache_256 --splits splits/controlled_v1.json --out_dir reports/eda
```

`make_splits.py` verifies the official per-group counts, builds the controlled
splits, validates every invariant and writes `splits/controlled_v1.json`.
**Commit that file**: splits are part of the experimental record. `data/`,
`cache_256/`, `runs/` and `reports/` are git-ignored.

Timing pilot, then the full ERM pilot at rho = 0.95:

```bash
python scripts/train.py --cache_dir cache_256 --splits_path splits/controlled_v1.json \
    --rho 0.95 --data_seed 0 --steps 300 --eval_every 100 --dense_eval_every 0 --tag timing
python scripts/inspect_run.py runs/<timing run folder>
python scripts/train.py --cache_dir cache_256 --splits_path splits/controlled_v1.json --rho 0.95 --data_seed 0
python scripts/inspect_run.py runs/<pilot run folder>
```

Every field of `TrainConfig` in `srl/train.py` is a flag. Rerunning an
interrupted command resumes from its checkpoint; rerunning a finished one is a
no-op. `notebooks/00_setup_eda_pilot.ipynb` is an equivalent Colab driver if
you ever need to run on Colab instead.

## Methods

| Method | How to run | Group labels used | Notes |
|---|---|---|---|
| ERM | `--method erm` | none | uniform batches |
| RWG | `--method rwg` | training | group-balanced batches, equal group weights |
| Group DRO | `--method gdro --eta_q 0.01` | training | online group weights; `eta_q 0` is exactly RWG |
| JTT | `--method jtt --jtt_T 50 --jtt_lambda 20` | validation (tuning) | error set read from the matching ERM run at step T; train that ERM run first; `jtt_lambda 1` is exactly ERM |
| DFR | `python scripts/dfr.py --runs "runs/..._erm_..."` | held-out set (val_bal) | L1 logistic head on saved ERM embeddings; selection protocol S1 only |

`bash scripts/pilot_methods.sh` runs the method pilot at rho = 0.95 (seeds 0 to 2) and
`scripts/summarize_validation.py` compares runs on validation data only.

## Controlled splits (defaults)

| Set | Size | Composition | Varies with |
|---|---|---|---|
| test | 1,200 | 300 per group | fixed |
| val_bal | 400 | 100 per group | fixed |
| train | 1,200 | 600 per class at correlation rho | rho, data seed |
| val_id | 400 | 200 per class at correlation rho | rho, data seed |

rho in {0.5, 0.8, 0.9, 0.95, 0.99} gives 300, 120, 60, 30 and 6 minority training
examples per class. For each seed, each group's remaining pool is shuffled once;
training sets read from the front and val_id from the back, so training sets
are nested across rho and never overlap val_id.

## Design decisions

| Decision | Reason |
|---|---|
| Fixed number of gradient steps for every method | Reweighting and upsampling change what an epoch means |
| Constant learning rate | An ERM run at step t equals an ERM run of t steps, so logged train logits give JTT's identification stage for free |
| Training seed defaults to data seed | Paired design: same images, head init and (same sampler) batches across methods |
| Logits logged on train/val/test at every evaluation step | All selection protocols, calibration and shift curves computed later without retraining |
| Test outputs written to `sealed/`, never scored during training | Prevents selection on test; only `scripts/final_eval.py --unseal` reads them |
| Dense evaluation early (every 25 steps to step 200), then every 100 | JTT and early-stopping analyses need early checkpoints |
| Images resized once to 256x256, cropped on GPU | Removes Colab's CPU JPEG-decoding bottleneck |
| GPU RandomResizedCrop(224, scale 0.7 to 1, ratio 3/4 to 4/3) + flip | Matches the augmentation of the Group DRO / DFR codebases |
| Defaults lr 1e-3, weight decay 1e-3, batch 32, SGD momentum 0.9 | ERM base-model settings of the DFR codebase; confirm in the pilot and preregister |
| ResNet-50 with torchvision IMAGENET1K_V1 weights | The weights used by prior Waterbirds work; V2 differs |

## Run directory

`runs/<split>_rho<rho>_ds<data seed>_ts<train seed>_<method>_<arch>_<config hash>/`
holds `config.json`, `env.json` (software, GPU, git commit per session),
`sets.npz` (ids, labels, groups), `train_log.csv`, `logits/*.npy`
(float16, `[n_evals, n, 2]`), `embeddings/*.npy` (final features for DFR),
`sealed/test.npy`, `sealed/test_embeddings.npy`, `summary.json` and `DONE`.

## Reproducibility

- Splits are committed JSON; `make_splits.py` prints their SHA-256.
- The cache manifest stores SHA-256 hashes of the image array and index; both
  fingerprints enter every run's config hash, so changed data means a new run name.
- `env.json` records library versions, GPU model and git commit (with a dirty flag).
- cuDNN is set deterministic by default. Bitwise GPU reproducibility is still not
  guaranteed, which is why variability is measured across seeds.

## Tests

```bash
pip install -e . && pytest -q
```

Runs on CPU with synthetic images: split invariants, metric identities,
augmentation geometry, sampler resumption, and an end-to-end training run that
must produce identical outputs whether or not it was interrupted and resumed.

## Data and licences

Cite all three sources: CUB-200-2011 (Wah et al., 2011), Places (Zhou et al.,
2017) and Waterbirds (Sagawa et al., 2020). CUB images are restricted to
non-commercial research and educational use, and some overlap with ImageNet,
which matters for ImageNet-pretrained models (note it as a limitation).

## AI assistance

The initial version of this code was drafted with an AI assistant (Claude,
Anthropic). Record what you adopted and how you verified it in `AI_USE.md`.
