import numpy as np
import torch

from srl.cache import DeviceImageBank
from srl.constants import IMAGENET_MEAN, IMAGENET_STD
from srl.metrics import acc_at_rho, expected_calibration_error, group_accuracies, summarize
from srl.samplers import GroupBalancedStream, UniformStream


# ---- metrics -------------------------------------------------------------------

def test_acc_at_rho_matches_a_sample_actually_drawn_at_rho():
    group_acc = np.array([0.98, 0.70, 0.55, 0.97])
    rho = 0.8
    counts = [400, 100, 100, 400]                    # class-balanced, rho = 0.8
    correct, g = [], []
    for k, n in enumerate(counts):
        hits = int(round(group_acc[k] * n))
        correct += [1] * hits + [0] * (n - hits)
        g += [k] * n
    correct, g = np.array(correct), np.array(g)
    assert np.isclose(correct.mean(), acc_at_rho(group_accuracies(correct, g), rho))


def test_summary_basics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    logits = np.zeros((8, 2))
    logits[np.arange(8), y] = 3.0
    logits[2] = [-3.0, 3.0]                          # one error in group 1
    m = summarize(logits, y, g)
    assert m["acc"] == 7 / 8 and m["wga"] == 0.5 and m["worst_group"] == 1
    assert np.isclose(m["shortcut_gap"], 1.0 - 0.75)


def test_ece_small_when_calibrated_large_when_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    calibrated = rng.uniform(size=conf.size) < conf
    assert expected_calibration_error(conf, calibrated) < 0.02
    overconfident = rng.uniform(size=conf.size) < conf - 0.2
    assert expected_calibration_error(conf, overconfident) > 0.15


# ---- augmentation ----------------------------------------------------------------

def _bank(n=4, size=64, crop=48):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)
    return imgs, DeviceImageBank(imgs, torch.device("cpu"), crop)


def _denorm(x):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255
    return x * std + mean


def test_center_crop_is_exact():
    imgs, bank = _bank()
    out = _denorm(bank.batch(torch.arange(4), augment="none"))
    ref = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()[:, :, 8:56, 8:56]
    assert torch.allclose(out, ref, atol=1e-3)


def test_affine_crop_geometry_is_exact_without_rescaling():
    # With scale such that the sampled box equals the crop size, grid_sample must
    # reproduce the pixels of that box exactly (pixel centers align).
    imgs, bank = _bank()
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    gen = torch.Generator().manual_seed(1)
    area_frac = (48 * 48) / (64 * 64)
    out = bank._random_resized_crop_flip(x, gen, scale=(area_frac, area_frac), ratio=(1.0, 1.0))
    found = 0
    for b in range(4):
        for i in range(17):
            for j in range(17):
                box = x[b, :, i:i + 48, j:j + 48]
                if torch.allclose(out[b], box, atol=1e-3) or torch.allclose(out[b], box.flip(-1), atol=1e-3):
                    found += 1
    assert found == 4


def test_rrc_shapes_and_randomness():
    _, bank = _bank()
    gen = torch.Generator().manual_seed(0)
    a = bank.batch(torch.arange(4), augment="rrc_flip", gen=gen)
    b = bank.batch(torch.arange(4), augment="rrc_flip", gen=gen)
    assert a.shape == (4, 3, 48, 48) and not torch.allclose(a, b)


# ---- samplers --------------------------------------------------------------------

def test_uniform_stream_covers_each_example_once_per_pass():
    s = UniformStream(10, 4, torch.Generator().manual_seed(0))
    seen = torch.cat([s.next_batch() for _ in range(5)])          # exactly two passes
    assert torch.bincount(seen, minlength=10).tolist() == [2] * 10


def test_stream_resume_reproduces_batches():
    gen = torch.Generator().manual_seed(0)
    s = UniformStream(13, 4, gen)
    for _ in range(3):
        s.next_batch()
    state, gstate = s.state_dict(), gen.get_state()
    expected = [s.next_batch() for _ in range(6)]
    gen2 = torch.Generator()
    gen2.set_state(gstate)
    s2 = UniformStream(13, 4, gen2)
    s2.load_state_dict(state)
    assert all(torch.equal(a, s2.next_batch()) for a in expected)


def test_group_balanced_batches():
    groups = np.array([0] * 50 + [1] * 3 + [2] * 2 + [3] * 45)
    s = GroupBalancedStream(groups, 8, torch.Generator().manual_seed(0))
    for _ in range(5):
        assert np.bincount(groups[s.next_batch().numpy()], minlength=4).tolist() == [2, 2, 2, 2]


# ---- configuration ---------------------------------------------------------------

def test_cli_parsing_and_eval_schedule():
    from srl.train import config_from_cli, make_eval_steps

    cfg = config_from_cli(["--seed", "none", "--amp", "false", "--rho", "0.9",
                           "--stop_after_step", "5", "--arch", "resnet18"])
    assert cfg.seed is None and cfg.amp is False and cfg.rho == 0.9
    assert cfg.stop_after_step == 5 and cfg.arch == "resnet18"
    steps = make_eval_steps(2000, 100, 25, 200)
    assert len(steps) == 27 and steps[:3] == [0, 25, 50] and steps[-1] == 2000
