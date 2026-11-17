"""Train one model. Every option of srl.train.TrainConfig is a flag.

Pilot (ERM, rho = 0.95, seed 0):
    python scripts/train.py --cache_dir /content/cache_256 \
        --splits_path splits/controlled_v1.json --rho 0.95 --data_seed 0 \
        --out_dir /content/drive/MyDrive/es678/runs

Official-split anchor:
    python scripts/train.py --cache_dir /content/cache_256 --split official --data_seed 0 \
        --out_dir /content/drive/MyDrive/es678/runs

Rerunning an interrupted command resumes it; rerunning a finished one is a no-op.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srl.train import config_from_cli, train  # noqa: E402

if __name__ == "__main__":
    train(config_from_cli())
