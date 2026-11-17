"""One training run: fixed gradient-step budget, logits logged at every
evaluation step, test outputs sealed, resumable after a Colab disconnect.

Run directory layout (runs/<run_name>/):
    config.json           every setting plus config hash and data fingerprints
    env.json              software, GPU and git commit, one entry per session
    sets.npz              img_id, y, g for every set (train and evals)
    train_log.csv         per evaluation step: loss, train/validation metrics, timing
    logits/<set>.npy      float16 [n_evals, n_examples, 2] for train and validation sets
    logits/eval_steps.npy the gradient step of each evaluation
    embeddings/<set>.npy  float16 penultimate features of the final model
    sealed/test.npy       test logits, same layout; read only by scripts/final_eval.py
    sealed/test_embeddings.npy
    summary.json          timing and throughput
    DONE                  completion marker (a rerun with the same config is skipped)
    ckpt.pt               resumable state, deleted on completion

Decisions worth knowing:
* Constant learning rate (as in the Group DRO and JTT papers). The state of
  an ERM run at step t then equals a separate ERM run of t steps, so the
  logged train-set logits give JTT's identification stage for free.
* The training seed defaults to the data seed. Two methods run with the same
  seed see the same training images, the same head initialization and, when
  they use the same sampler, the same batches: a paired design.
* No metric is ever computed on sealed sets during training.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cache import DeviceImageBank, ImageCache
from .constants import N_GROUPS, SEALED_SETS
from .data import load_splits, resolve_sets
from .metrics import summarize
from .models import build_model, forward_with_features
from .objectives import make_objective
from .samplers import make_stream
from .utils import (configure_determinism, env_info, load_json, rho_key, save_json,
                    set_global_seeds, sha256_json)


@dataclass
class TrainConfig:
    # data
    cache_dir: str = ""
    splits_path: str = ""              # required when split == "controlled"
    split: str = "controlled"          # "controlled" or "official"
    rho: float | None = 0.95
    data_seed: int = 0
    seed: int | None = None            # training seed; None -> data_seed (paired design)
    # method
    method: str = "erm"
    # model
    arch: str = "resnet50"
    pretrained: bool = True
    weights: str = "IMAGENET1K_V1"
    # optimization (defaults follow the DFR codebase's ERM base model)
    lr: float = 1e-3
    weight_decay: float = 1e-3
    momentum: float = 0.9
    batch_size: int = 32
    steps: int = 2000
    augment: str = "rrc_flip"          # "rrc_flip" or "none"
    crop: int = 224
    amp: bool = True
    # evaluation and logging
    eval_every: int = 100
    dense_eval_every: int = 25         # denser evaluation early in training...
    dense_eval_until: int = 200        # ...up to this step (JTT needs early checkpoints)
    eval_batch_size: int = 256
    ckpt_every: int = 500
    save_embeddings: bool = True
    save_final_model: bool = False
    deterministic: bool = True
    out_dir: str = "runs"
    tag: str = ""
    stop_after_step: int | None = None  # end this session early, keeping the checkpoint
    device: str = "auto"


# Fields that do not change what is trained, so they are excluded from the hash.
_NON_RESULT_FIELDS = {"cache_dir", "splits_path", "out_dir", "tag", "stop_after_step", "device",
                      "eval_batch_size", "ckpt_every", "save_embeddings", "save_final_model",
                      "deterministic"}


def train_seed(cfg: TrainConfig) -> int:
    return cfg.data_seed if cfg.seed is None else cfg.seed


def config_hash(cfg: TrainConfig, data_fingerprints: dict) -> str:
    fields = {k: v for k, v in asdict(cfg).items() if k not in _NON_RESULT_FIELDS}
    fields["seed"] = train_seed(cfg)
    if cfg.split != "controlled":
        fields["rho"] = None
    return sha256_json({"config": fields, "data": data_fingerprints})[:16]


def run_name(cfg: TrainConfig, h: str) -> str:
    parts = [cfg.split]
    if cfg.split == "controlled":
        parts.append(f"rho{rho_key(cfg.rho)}")
    parts += [f"ds{cfg.data_seed}", f"ts{train_seed(cfg)}", cfg.method, cfg.arch, h[:8]]
    if cfg.tag:
        parts.append(cfg.tag)
    return "_".join(parts)


def make_eval_steps(total: int, every: int, dense_every: int, dense_until: int) -> list[int]:
    steps = set(range(0, total + 1, every)) | {0, total}
    if dense_every:
        steps |= set(range(0, min(dense_until, total) + 1, dense_every))
    return sorted(steps)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):           # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                             # PyTorch without weights_only
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def predict(model, bank: DeviceImageBank, batch_size: int, use_amp: bool, features: bool = False):
    was_training = model.training
    model.eval()
    logits, feats = [], []
    for start in range(0, len(bank), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(bank)), device=bank.device)
        x = bank.batch(idx, augment="none")
        with torch.autocast(device_type=bank.device.type, dtype=torch.float16, enabled=use_amp):
            lg, f = forward_with_features(model, x)
        logits.append(lg.float().cpu())
        if features:
            feats.append(f.float().cpu())
    if was_training:
        model.train()
    out_logits = torch.cat(logits).numpy()
    return (out_logits, torch.cat(feats).numpy()) if features else out_logits


def train(cfg: TrainConfig) -> Path:
    t_start = time.time()
    device = _resolve_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")
    seed = train_seed(cfg)

    # ---- data and run identity ---------------------------------------------
    cache = ImageCache(cfg.cache_dir)
    fingerprints = {"cache_sha256": cache.fingerprint}
    if cfg.split == "controlled":
        if not cfg.splits_path:
            raise ValueError("--splits_path is required for the controlled split")
        splits, splits_sha = load_splits(cfg.splits_path)
        fingerprints["splits_sha256"] = splits_sha
        sets = resolve_sets("controlled", splits=splits, rho=cfg.rho, data_seed=cfg.data_seed)
    else:
        sets = resolve_sets("official", pool=cache.index)

    h = config_hash(cfg, fingerprints)
    run_dir = Path(cfg.out_dir) / run_name(cfg, h)
    if (run_dir / "DONE").exists():
        print(f"[skip] {run_dir.name} is already complete")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {**asdict(cfg), "train_seed": seed, "config_hash": h,
                                        **fingerprints, "sets": sets.description})
    sessions = load_json(run_dir / "env.json") if (run_dir / "env.json").exists() else []
    save_json(run_dir / "env.json", sessions + [env_info()])

    set_global_seeds(seed)
    configure_determinism(cfg.deterministic)

    names = ["train"] + list(sets.evals)
    ids = {"train": sets.train, **sets.evals}
    labels = {name: cache.labels(ids[name]) for name in names}       # (y, a, g)
    banks = {name: DeviceImageBank(cache.take(ids[name]), device, cfg.crop) for name in names}
    np.savez(run_dir / "sets.npz",
             **{f"{n}_{k}": v for n in names
                for k, v in (("img_id", ids[n]), ("y", labels[n][0]), ("g", labels[n][2]))})
    y_tr, _, g_tr = labels["train"]
    y_tr_t = torch.as_tensor(y_tr, device=device)
    g_tr_t = torch.as_tensor(g_tr, device=device)
    n_train = len(y_tr)
    train_props = np.bincount(g_tr, minlength=N_GROUPS) / n_train
    print(f"[run] {run_dir.name}\n[run] {sets.description}; device={device}, amp={use_amp}; "
          + ", ".join(f"{n}={len(ids[n])}" for n in names))

    # ---- model, optimizer, objective, sampler --------------------------------
    model = build_model(cfg.arch, cfg.pretrained, cfg.weights).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    scaler = _make_scaler(use_amp)
    objective = make_objective(cfg.method, y_tr, g_tr)
    gen = torch.Generator().manual_seed(seed)          # data order and augmentation only
    stream = make_stream(objective.sampler, n_train, g_tr, cfg.batch_size, gen)

    eval_steps = make_eval_steps(cfg.steps, cfg.eval_every, cfg.dense_eval_every,
                                 cfg.dense_eval_until)
    eval_pos = {s: i for i, s in enumerate(eval_steps)}
    logit_buf = {n: np.full((len(eval_steps), len(banks[n]), 2), np.nan, dtype=np.float16)
                 for n in names}
    log_rows: list[dict] = []
    step, n_done = 0, 0
    loss_sum = torch.zeros((), device=device)   # summed on device: no per-step GPU sync
    loss_count = 0
    train_time, eval_time = 0.0, 0.0

    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        # Load to CPU: RNG states and sampler permutations must stay CPU tensors;
        # model and optimizer tensors are moved to the right device on load.
        state = _torch_load(ckpt_path, "cpu")
        if state["config_hash"] != h:
            raise RuntimeError(f"{ckpt_path} belongs to a different configuration")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if state["scaler"]:                    # empty when saved by a disabled (CPU) scaler
            scaler.load_state_dict(state["scaler"])
        objective.load_state_dict(state["objective"])
        gen.set_state(state["gen"])
        stream.load_state_dict(state["stream"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        logit_buf, log_rows = state["logit_buf"], state["log_rows"]
        step, n_done = state["step"], state["n_evals_done"]
        loss_sum = torch.tensor(state["loss_sum"], device=device)
        loss_count = state["loss_count"]
        train_time, eval_time = state["train_time"], state["eval_time"]
        print(f"[resume] from step {step} ({n_done} evaluations already logged)")

    def save_checkpoint():
        state = {
            "config_hash": h, "step": step, "n_evals_done": n_done,
            "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "objective": objective.state_dict(), "gen": gen.get_state(),
            "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "logit_buf": logit_buf, "log_rows": log_rows,
            "loss_sum": float(loss_sum), "loss_count": loss_count,
            "train_time": train_time, "eval_time": eval_time,
        }
        tmp = ckpt_path.with_name("ckpt.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)

    def evaluate(at_step: int, mean_loss: float) -> dict:
        row = {"step": at_step, "epoch": at_step * cfg.batch_size / n_train,
               "train_loss": mean_loss, "train_seconds": train_time}
        for n in names:
            logits = predict(model, banks[n], cfg.eval_batch_size, use_amp)
            logit_buf[n][eval_pos[at_step]] = logits.astype(np.float16)
            if n in SEALED_SETS:
                continue                     # stored, never scored here
            m = summarize(logits, labels[n][0], labels[n][2])
            for k in ("acc", "wga", "mean_group_acc", "worst_class_acc", "nll"):
                row[f"{n}_{k}"] = m[k]
        row.update(objective.log_items())
        return row

    def print_row(row: dict):
        parts = [f"step {row['step']:5d}", f"ep {row['epoch']:5.1f}"]
        if not np.isnan(row["train_loss"]):
            parts.append(f"loss {row['train_loss']:.4f}")
        for n in names:
            if n in SEALED_SETS:
                continue
            part = f"{n} acc {row[f'{n}_acc']:.3f} wga {row[f'{n}_wga']:.3f}"
            parts.append(part)
        print(" | ".join(parts), flush=True)

    # ---- training loop -------------------------------------------------------
    model.train()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    start_step = step
    t_mark = time.time()
    while True:
        if step in eval_pos and eval_pos[step] >= n_done:
            sync()
            train_time += time.time() - t_mark
            t_eval = time.time()
            mean_loss = float(loss_sum) / loss_count if loss_count else float("nan")
            if loss_count and not np.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite training loss before step {step}")
            row = evaluate(step, mean_loss)
            log_rows.append(row)
            n_done += 1
            loss_sum.zero_()
            loss_count = 0
            print_row(row)
            pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
            eval_time += time.time() - t_eval
            t_mark = time.time()

        stop_now = cfg.stop_after_step is not None and cfg.stop_after_step <= step < cfg.steps
        if step < cfg.steps and step != start_step and (stop_now or step % cfg.ckpt_every == 0):
            sync()
            train_time += time.time() - t_mark
            save_checkpoint()
            t_mark = time.time()
        if step >= cfg.steps or stop_now:
            break

        idx = stream.next_batch().to(device, non_blocking=True)
        x = banks["train"].batch(idx, augment=cfg.augment, gen=gen)
        yb, gb = y_tr_t[idx], g_tr_t[idx]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        per_example = F.cross_entropy(logits.float(), yb, reduction="none")
        loss = objective.loss(per_example, yb, gb, step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.detach()
        loss_count += 1
        step += 1

    if step < cfg.steps:
        print(f"[pause] stopped at step {step}; rerun the same command to resume")
        return run_dir

    # ---- finalize ------------------------------------------------------------
    (run_dir / "logits").mkdir(exist_ok=True)
    (run_dir / "sealed").mkdir(exist_ok=True)
    np.save(run_dir / "logits" / "eval_steps.npy", np.asarray(eval_steps))
    for n in names:
        folder = "sealed" if n in SEALED_SETS else "logits"
        np.save(run_dir / folder / f"{n}.npy", logit_buf[n])
    if cfg.save_embeddings:
        (run_dir / "embeddings").mkdir(exist_ok=True)
        for n in names:
            _, feats = predict(model, banks[n], cfg.eval_batch_size, use_amp, features=True)
            path = (run_dir / "sealed" / f"{n}_embeddings.npy" if n in SEALED_SETS
                    else run_dir / "embeddings" / f"{n}.npy")
            np.save(path, feats.astype(np.float16))
    if cfg.save_final_model:
        torch.save(model.state_dict(), run_dir / "final_model.pt")
    throughput = cfg.steps * cfg.batch_size / max(train_time, 1e-9)
    summary = {
        "train_seconds": round(train_time, 1), "eval_seconds": round(eval_time, 1),
        "train_images_per_second": round(throughput, 1),
        "wall_seconds_this_session": round(time.time() - t_start, 1),
        "n_evals": len(eval_steps), "train_group_props": train_props.tolist(),
    }
    if device.type == "cuda":
        summary["max_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    save_json(run_dir / "summary.json", summary)
    print(f"[time] training {train_time:.0f}s ({throughput:.0f} img/s), evaluation {eval_time:.0f}s")
    pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"[done] {run_dir}")
    return run_dir


# ---- command line --------------------------------------------------------------

def _parse_bool(text: str) -> bool:
    if text.lower() in {"1", "true", "yes", "y"}:
        return True
    if text.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def config_from_cli(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train one model (see srl/train.py docstring).")
    defaults = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        default = getattr(defaults, f.name)
        kind = str(f.type)
        if "bool" in kind:
            parser.add_argument(f"--{f.name}", type=_parse_bool, default=default)
        elif "None" in kind:
            base = float if "float" in kind else int
            parser.add_argument(f"--{f.name}", default=default,
                                type=lambda s, b=base: None if s.lower() == "none" else b(s))
        elif "float" in kind:
            parser.add_argument(f"--{f.name}", type=float, default=default)
        elif "int" in kind:
            parser.add_argument(f"--{f.name}", type=int, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=str, default=default)
    return TrainConfig(**vars(parser.parse_args(argv)))
