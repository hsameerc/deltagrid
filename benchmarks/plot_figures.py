"""
Builds figures/rigorous_dashboard.png from the JSON files written by benchmark.py.

Panels: MQAR (capacity), Needle (train 32/64 -> test up to 2048), Adding (train T=50 -> test up to 800),
        Sine forecast (in-dist / OOD / noisy), Real memory (peak RSS).
Transformer uses the tuned run (lr 1e-3, 16x steps) when available; other baselines use the long-budget run.
Usage: python tests/plot_benchmark.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
COL = {"deltagrid": "#1f77b4", "gru": "#2ca02c", "lstm": "#ff7f0e", "transformer": "#d62728"}


def load(name):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    # Fallback to consolidated rigorous_results.json
    res_path = os.path.join(FIG, "rigorous_results.json")
    if os.path.exists(res_path):
        try:
            data = json.load(open(res_path))
            for k in ("mqar", "needle", "adding", "sine", "memory", "charlm"):
                if k in name.lower() and k in data:
                    return {k: data[k]}
            return data
        except Exception:
            pass
    return None


def merged(task, base, long_, tuned):
    """Returns {kind: [runs]} using DeltaGrid from `base`, GRU/LSTM from `long_`, Transformer from `tuned` (fallback long_)."""
    out = {}
    b, l, t = load(base), load(long_), load(tuned)
    res = load("rigorous_results.json")

    def get_task_dict(source):
        if not source:
            return None
        if task in source:
            return source[task]
        return None

    src_b = get_task_dict(b) or get_task_dict(res)
    src_l = get_task_dict(l) or src_b
    src_t = get_task_dict(t) or src_l

    if src_b and "deltagrid" in src_b:
        out["deltagrid"] = src_b["deltagrid"]
    for k in ("gru", "lstm"):
        if src_l and k in src_l:
            out[k] = src_l[k]
        elif src_b and k in src_b:
            out[k] = src_b[k]
    for src in (src_t, src_l, src_b):
        if src and "transformer" in src:
            out["transformer"] = src["transformer"]
            break
    return out


def curve(ax, runs_by_kind, keys, xs, title, xlabel, ylabel, logx=True, logy=False):
    if not runs_by_kind:
        ax.set_title(title, fontweight="bold")
        ax.text(0.5, 0.5, "No data found\n(Run benchmark.py to generate)",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        return

    for kind, runs in runs_by_kind.items():
        if not runs:
            continue
        ys = []
        for k in keys:
            vals = [r["metrics"][k] for r in runs if "metrics" in r and k in r["metrics"]]
            if not vals:
                vals = [0.0]
            ys.append(vals)
        m = [sum(v) / len(v) for v in ys]
        s = [(sum((x - mm) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 for v, mm in zip(ys, m)]
        ax.errorbar(xs[:len(m)], m, yerr=s, marker="o", capsize=3, label=kind, color=COL.get(kind, "#333333"), linewidth=2)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("HSRU DeltaGrid - rigorous benchmarks (mean +- std over seeds, parameter-matched baselines)", fontsize=15, fontweight="bold")

mq = merged("mqar", "rigorous_mqar.json", "rigorous_mqar_baselines_long.json", "rigorous_mqar_transformer_tuned.json")
pairs = [4, 8, 16, 24, 32, 48]
curve(axes[0, 0], mq, [f"pairs={p}" for p in pairs], pairs, "MQAR: recall vs #pairs (trained on <=24)", "key-value pairs", "accuracy (%)")
axes[0, 0].axvline(24, color="gray", linestyle="--", alpha=0.5)

nd = merged("needle", "rigorous_needle.json", "rigorous_needle_baselines_long.json", "rigorous_needle_transformer_tuned.json")
lens = [64, 128, 256, 512, 1024, 2048]
curve(axes[0, 1], nd, [f"L={l}" for l in lens], lens, "Needle (random position): trained on L=32/64", "sequence length", "accuracy (%)")

ad = merged("adding", "rigorous_adding.json", "rigorous_adding.json", "rigorous_adding.json")
tl = [50, 100, 200, 400, 800]
curve(axes[0, 2], ad, [f"T={t}" for t in tl], tl, "Adding problem: trained on T=50", "sequence length", "MSE (log)", logy=True)
axes[0, 2].axhline(0.1667, color="gray", linestyle="--", label="predict-mean")

res_combined = load("rigorous_results.json")

sn = load("rigorous_sine.json") or (res_combined if res_combined and "sine" in res_combined else None)
if sn and "sine" in sn:
    ax = axes[1, 0]
    scen = ["in-dist", "OOD-freq(0.25-0.35)", "noisy(0.05)"]
    w = 0.2
    kinds = [k for k in ["deltagrid", "gru", "lstm", "transformer"] if k in sn["sine"]]
    for i, kind in enumerate(kinds):
        ys = [sum(r["metrics"][s] for r in sn["sine"][kind] if s in r["metrics"]) / max(1, len(sn["sine"][kind])) for s in scen]
        ax.bar([j + i * w for j in range(3)], ys, w, label=kind, color=COL.get(kind, "#333333"))
    if "deltagrid" in sn["sine"]:
        zero = [sum(r["metrics"].get(s + " [predict-0]", 0.0) for r in sn["sine"]["deltagrid"]) / max(1, len(sn["sine"]["deltagrid"])) for s in scen]
        ax.plot([j + 1.5 * w for j in range(3)], zero, "k--", marker="x", label="predict-0")
    ax.set_xticks([j + 1.5 * w for j in range(3)])
    ax.set_xticklabels(scen)
    ax.set_yscale("log")
    ax.set_title("Sine forecast (32-step horizon MSE)", fontweight="bold")
    ax.set_ylabel("MSE (log)")
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
else:
    axes[1, 0].set_title("Sine forecast (32-step horizon MSE)", fontweight="bold")
    axes[1, 0].text(0.5, 0.5, "No sine benchmark data found\n(Run benchmark.py --tasks sine)",
                    ha="center", va="center", transform=axes[1, 0].transAxes, color="gray", fontsize=10)

mem = load("rigorous_memory.json") or (res_combined if res_combined and "memory" in res_combined else None)
if mem and "memory" in mem and "rows" in mem["memory"]:
    ax = axes[1, 1]
    rows = mem["memory"]["rows"]
    for key, lab, col in (("dg_stream", "DeltaGrid streaming (chunked, carried state)", "#1f77b4"),
                          ("dg_full", "DeltaGrid one-shot prefill", "#17becf"),
                          ("tf_full", "Transformer one-shot (naive attention)", "#d62728")):
        pts = [(r["length"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if pts:
            ax.plot(*zip(*pts), marker="o", label=lab, color=col, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Measured peak RSS delta (CPU, batch=1)", fontweight="bold")
    ax.set_xlabel("tokens")
    ax.set_ylabel("MB")
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
else:
    axes[1, 1].set_title("Measured peak RSS delta (CPU, batch=1)", fontweight="bold")
    axes[1, 1].text(0.5, 0.5, "No memory benchmark data found\n(Run benchmark.py --tasks memory)",
                    ha="center", va="center", transform=axes[1, 1].transAxes, color="gray", fontsize=10)

ch = load("rigorous_charlm.json") or (res_combined if res_combined and "charlm" in res_combined else None)
if ch and "charlm" in ch:
    ax = axes[1, 2]
    keys = ["val_bpc@ctx256", "val_bpc@ctx1024"]
    kinds = [k for k in ["deltagrid", "gru", "lstm", "transformer"] if k in ch["charlm"]]
    for i, kind in enumerate(kinds):
        ys = [sum(r["metrics"][k] for r in ch["charlm"][kind] if k in r["metrics"]) / max(1, len(ch["charlm"][kind])) for k in keys]
        ax.bar([j + i * 0.2 for j in range(2)], ys, 0.2, label=kind, color=COL.get(kind, "#333333"))
    ax.set_xticks([0.3, 1.3])
    ax.set_xticklabels(["context 256 (trained)", "context 1024"])
    ax.set_title("Byte-level LM on Python source (val bits/char, lower=better)", fontweight="bold", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
else:
    axes[1, 2].set_title("Byte-level LM on Python source", fontweight="bold", fontsize=10)
    axes[1, 2].text(0.5, 0.5, "No charlm benchmark data found\n(Run benchmark.py --tasks charlm)",
                    ha="center", va="center", transform=axes[1, 2].transAxes, color="gray", fontsize=10)

plt.tight_layout()
out = os.path.join(FIG, "rigorous_dashboard.png")
plt.savefig(out, dpi=140)
print("saved", out)

