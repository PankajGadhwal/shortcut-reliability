"""End-to-end test on tiny synthetic images, CPU, untrained ResNet-18.
Checks plumbing, file layout, sealing, resume equivalence and selection;
it says nothing about real performance."""

import numpy as np
import pytest

from srl.cache import ImageCache, build_cache
from srl.data import SplitSpec, load_pool, make_controlled_splits, save_splits
from srl.evaluate import evaluate_run
from srl.selection import available_protocols, select
from srl.testing import write_fake_dataset
from srl.train import TrainConfig, train

SMALL = SplitSpec(rhos=(0.5, 0.75), data_seeds=(0, 1), test_per_group=6, val_bal_per_group=4,
                  train_per_class=16, val_id_per_class=8, name="tiny")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    counts = {"train": (14, 6, 6, 14), "val": (8, 8, 8, 8), "test": (8, 16, 16, 8)}
    root = write_fake_dataset(tmp / "wb", counts)
    pool = load_pool(root)
    build_cache(pool, root, tmp / "cache", size=36, workers=1)
    splits = make_controlled_splits(pool, SMALL)
    splits_path = tmp / "splits.json"
    save_splits(splits_path, splits)
    return tmp, splits_path


def _cfg(tmp, splits_path, out, **kw):
    base = dict(cache_dir=str(tmp / "cache"), splits_path=str(splits_path), split="controlled",
                rho=0.75, data_seed=1, arch="resnet18", pretrained=False, steps=12,
                batch_size=8, crop=32, eval_every=4, dense_eval_every=2, dense_eval_until=4,
                ckpt_every=4, eval_batch_size=16, out_dir=str(tmp / out), amp=False)
    base.update(kw)
    return TrainConfig(**base)


def test_cache_roundtrip(setup):
    tmp, _ = setup
    cache = ImageCache(tmp / "cache", verify=True)
    ids = cache.index["img_id"].to_numpy()[[5, 0, 17]]
    imgs = cache.take(ids)
    assert imgs.shape == (3, 36, 36, 3) and imgs.dtype == np.uint8
    assert np.array_equal(imgs[1], cache.images[0])


def test_run_layout_sealing_and_resume(setup):
    tmp, splits_path = setup
    run_a = train(_cfg(tmp, splits_path, "runs_a"))
    for rel in ("DONE", "config.json", "env.json", "sets.npz", "train_log.csv", "summary.json",
                "logits/eval_steps.npy", "logits/train.npy", "logits/val_bal.npy",
                "logits/val_id.npy", "sealed/test.npy", "sealed/test_embeddings.npy",
                "embeddings/train.npy"):
        assert (run_a / rel).exists(), rel
    assert not (run_a / "logits" / "test.npy").exists()
    assert not (run_a / "ckpt.pt").exists()
    steps = np.load(run_a / "logits" / "eval_steps.npy")
    assert steps.tolist() == [0, 2, 4, 8, 12]
    val = np.load(run_a / "logits" / "val_bal.npy")
    assert val.shape == (5, 16, 2) and np.isfinite(val).all()
    log = open(run_a / "train_log.csv").read()
    assert "test_" not in log                     # no test metric ever computed in training

    # Same config, interrupted after step 6 and resumed, must match exactly.
    paused = train(_cfg(tmp, splits_path, "runs_b", stop_after_step=6))
    assert (paused / "ckpt.pt").exists() and not (paused / "DONE").exists()
    run_b = train(_cfg(tmp, splits_path, "runs_b"))
    assert run_b.name == run_a.name
    for name in ("logits/train.npy", "logits/val_bal.npy", "sealed/test.npy"):
        assert np.array_equal(np.load(run_a / name), np.load(run_b / name)), name

    # A completed run is skipped, not retrained.
    assert train(_cfg(tmp, splits_path, "runs_a")) == run_a


def test_selection_and_final_eval(setup):
    tmp, splits_path = setup
    run = train(_cfg(tmp, splits_path, "runs_a"))
    assert available_protocols(run) == ["S1", "S2", "S3", "S4"]
    sel = select(run, "S1")
    assert sel.step > 0
    with pytest.raises(KeyError):
        evaluate_run(run)[0]["test_wga"]          # sealed unless unseal=True
    rows = evaluate_run(run, unseal=True)
    assert {"test_wga", "test_id_acc", "test_acc_at_rho_0"} <= set(rows[0])


def test_official_split_runs(setup):
    tmp, splits_path = setup
    run = train(_cfg(tmp, splits_path, "runs_off", split="official", steps=4, eval_every=2,
                     dense_eval_every=0))
    assert available_protocols(run) == ["S1", "S2"]
    assert (run / "logits" / "val.npy").exists() and (run / "sealed" / "test.npy").exists()
