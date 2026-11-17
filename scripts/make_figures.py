
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {"erm": ("ERM", "k", "o"), "rwg": ("RWG", "C0", "s"), "jtt": ("JTT", "C1", "^"),
         "dfr": ("DFR", "C2", "D"), "gdro": ("Group DRO", "C3", "v")}
PROTO = {"S1": ("S1: val$_{bal}$ WGA", "-"), "S2": ("S2: val$_{bal}$ acc", "--"),
         "S3": ("S3: val$_{id}$ worst-class", ":"), "S4": ("S4: val$_{id}$ acc", "-.")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def agg(d, method, protocol, col="test_wga"):
    s = d[(d.method == method) & (d.protocol == protocol)].groupby("rho")[col]
    return s.mean(), s.std()


def fig_rho(d, out):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in ("erm", "rwg", "jtt", "dfr"):
        mu, sd = agg(d, m, "S1")
        name, c, mk = STYLE[m]
        ax.errorbar(mu.index, mu.values, yerr=sd.values, color=c, marker=mk, ms=3.5, lw=1.2,
                    capsize=2, label=f"{name} WGA")
    mu, sd = agg(d, "erm", "S1", "test_id_acc")
    ax.plot(mu.index, mu.values, "k--", lw=1, label="ERM ID acc")
    g = d[(d.method == "gdro") & (d.protocol == "S1")]
    if len(g):
        ax.errorbar([0.95], [g.test_wga.mean()], yerr=[g.test_wga.std()], color="C3", marker="v",
                    ms=4, ls="none", capsize=2, label="Group DRO ($\\eta_q$=1) WGA")
    ax.set_xlabel(r"training correlation $\rho$")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=1)
    fig.savefig(out / "fig_rho.pdf")
    plt.close(fig)


def fig_selection(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
    for ax, m in zip(axes, ("erm", "rwg", "jtt")):
        for p, (lab, ls) in PROTO.items():
            mu, sd = agg(d, m, p)
            if len(mu):
                ax.errorbar(mu.index, mu.values, yerr=sd.values, ls=ls, marker=".", capsize=1.5,
                            lw=1.1, label=lab)
        ax.set_title(STYLE[m][0])
        ax.set_xlabel(r"$\rho$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test worst-group acc.")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.savefig(out / "fig_selection.pdf")
    plt.close(fig)


def fig_mechanism(d, jtt, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1))
    if jtt is not None:
        g = jtt.groupby("rho")[["precision_minority", "recall_minority"]]
        mu, sd = g.mean(), g.std()
        for col, lab, c in (("precision_minority", "precision", "C1"),
                            ("recall_minority", "recall", "C4")):
            axes[0].errorbar(mu.index, mu[col], yerr=sd[col], marker="o", ms=3, capsize=2,
                             color=c, label=f"error set {lab} (minority)")
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("JTT error set at $T=25$")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")
    for m in ("erm", "rwg", "jtt", "dfr"):
        mu, sd = agg(d, m, "S1", "test_worst_group_ece")
        name, c, mk = STYLE[m]
        axes[1].errorbar(mu.index, mu.values, yerr=sd.values, color=c, marker=mk, ms=3,
                         capsize=2, label=name)
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_title("worst-group ECE (S1)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")
    fig.savefig(out / "fig_mechanism.pdf")
    plt.close(fig)


def paired(d):
    base = d[d.method == "erm"].set_index(["protocol", "rho", "data_seed"])["test_wga"]
    rows = []
    for (p, m, r), g in d[d.method != "erm"].groupby(["protocol", "method", "rho"]):
        gains = np.array([w - base.get((p, r, s), np.nan) for s, w in zip(g.data_seed, g.test_wga)])
        rows.append({"protocol": p, "method": m, "rho": r, "n": len(gains),
                     "mean_gain": gains.mean(), "min_gain": gains.min(), "max_gain": gains.max(),
                     "wins": int((gains > 0).sum()), "wins_3pt": int((gains > 0.03).sum())})
    return pd.DataFrame(rows)


def table(d, out):
    def cell(m, p, r, col):
        x = d[(d.method == m) & (d.protocol == p) & (np.isclose(d.rho, r))][col]
        return "--" if x.empty else f"{100 * x.mean():.1f}$\\pm${100 * x.std():.1f}"
    lines = [r"\begin{tabular}{l cc cc c}", r"\toprule",
             r" & \multicolumn{2}{c}{WGA, $\rho=0.95$} & \multicolumn{2}{c}{WGA, $\rho=0.99$} & ID acc. \\",
             r"Method & S1 & S4 & S1 & S4 & $\rho=0.95$, S1 \\", r"\midrule"]
    for m in ("erm", "rwg", "jtt", "gdro", "dfr"):
        lines.append(" & ".join([STYLE[m][0], cell(m, "S1", .95, "test_wga"), cell(m, "S4", .95, "test_wga"),
                                 cell(m, "S1", .99, "test_wga"), cell(m, "S4", .99, "test_wga"),
                                 cell(m, "S1", .95, "test_id_acc")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/main/final_eval.csv")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/figures")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.csv)
    d["rho"] = d["rho"].round(4)
    files = glob.glob(f"{args.runs}/*jtt*/jtt_error_set.json")
    jtt = None
    if files:
        jtt = pd.DataFrame([json.load(open(f)) | {"run": f} for f in files])
        jtt["rho"] = jtt.run.str.extract(r"rho([0-9.]+)_")[0].astype(float)
    fig_rho(d, out)
    fig_selection(d, out)
    fig_mechanism(d, jtt, out)
    table(d, out)
    pg = paired(d)
    pg.to_csv(out / "paired_gains.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pg[pg.protocol.isin(["S1", "S4"])].sort_values(["method", "protocol", "rho"]).to_string(index=False))
    print(f"\nwrote figures and table to {out}")


if __name__ == "__main__":
    sys.exit(main())
