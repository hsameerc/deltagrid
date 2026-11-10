"""
=============================================================================
HSRU DeltaGrid: Multi-Axis Routing Upgrade Benchmark & Empirical Evaluation
=============================================================================

Compares routing architectures for associative memory scaling:
  1. Baseline 2-Axis: (1024, 1024) -> 1,048,576 slots, Top-K=8 (64 active slots, 8 KB SRAM)
  2. Upgraded 3-Axis: (128, 128, 128) -> 2,097,152 slots, Top-K=3 (27 active slots, 3.4 KB SRAM)
  3. Upgraded 4-Axis: (32, 32, 32, 32) -> 1,048,576 slots, Top-K=2 (16 active slots, 2.0 KB SRAM)
  4. Ultra 4-Axis:    (40, 40, 40, 40) -> 2,560,000 slots, Top-K=2 (16 active slots, 2.0 KB SRAM)

Evaluates:
  - Recall (> 0.80)
  - Mean Cosine Similarity
  - Active Compute SRAM Window (KB)
  - Execution Time / Throughput

Author: Sameer Humagain (HSRU Project, 2026)
=============================================================================
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, List, Any, Tuple
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.memory import DeltaGridLayer

try:
    import hsru_delta_cuda
    _HAS_CUDA = hasattr(hsru_delta_cuda, "pkm_step")
except ImportError:
    hsru_delta_cuda = None
    _HAS_CUDA = False


def evaluate_routing_config(
    num_facts: int,
    grid_dim: Tuple[int, ...],
    top_k: int,
    head_dim: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluates associative recall and cosine similarity for a specific routing configuration."""
    torch.manual_seed(42)
    use_cuda = (device == "cuda" and _HAS_CUDA and hsru_delta_cuda is not None)

    layer = DeltaGridLayer(
        d_model=head_dim * 2,
        num_heads=2,
        grid_dim=grid_dim,
        top_k=top_k,
    ).to(device).eval()

    torch.manual_seed(42)
    keys = torch.randn(num_facts, head_dim, device=device)
    vals = torch.randn(num_facts, head_dim, device=device)

    total_slots = layer.total_slots
    active_slots = layer.active_slots
    sram_kb = (active_slots * head_dim * 2) / 1024.0  # FP16 active register slice

    all_k = keys.unsqueeze(0).unsqueeze(0)
    wk, wq, idx = layer._route_sparse_pkm(all_k, all_k)
    idx = idx.to(torch.int64)

    state = torch.zeros(1, 1, total_slots, head_dim, device=device)
    decay = torch.ones(1, device=device)
    b_one = torch.ones(1, 1, device=device)

    t0 = time.time()
    for t in range(num_facts):
        vt = vals[t:t + 1].unsqueeze(1)
        wkt = wk[:, :, t]
        wqt = wq[:, :, t]
        idxt = idx[:, :, t]
        if use_cuda:
            _, state = hsru_delta_cuda.pkm_step(vt, b_one, wkt, wqt, idxt, decay, state)
        else:
            expand_idx = idxt.unsqueeze(-1).expand(1, 1, active_slots, head_dim)
            s_act = torch.gather(state, 2, expand_idx)
            pred_v = torch.matmul(wkt.unsqueeze(-2), s_act).squeeze(-2)
            rt = vt - pred_v
            delta = (b_one * wkt).unsqueeze(-1) * rt.unsqueeze(-2)
            s_act_new = decay * s_act + delta
            state.scatter_(2, expand_idx, s_act_new)

    write_time = time.time() - t0

    # Test retrieval across sample facts
    eval_count = min(num_facts, 1000)
    eval_indices = torch.linspace(0, num_facts - 1, eval_count).long()
    sims = []

    for idx_eval in eval_indices:
        t_idx = idx_eval.item()
        wqt = wq[:, :, t_idx]
        idxt = idx[:, :, t_idx]
        vt_true = vals[t_idx]

        if use_cuda:
            dummy_v = torch.zeros(1, 1, head_dim, device=device)
            b_zero = torch.zeros(1, 1, device=device)
            y_ret, _ = hsru_delta_cuda.pkm_step(dummy_v, b_zero, wqt, wqt, idxt, decay, state)
            y_ret = y_ret.squeeze()
        else:
            expand_idx = idxt.unsqueeze(-1).expand(1, 1, active_slots, head_dim)
            s_act = torch.gather(state, 2, expand_idx)
            y_ret = torch.matmul(wqt.unsqueeze(-2), s_act).squeeze().squeeze()

        cos_sim = F.cosine_similarity(y_ret.unsqueeze(0), vt_true.unsqueeze(0)).item()
        sims.append(cos_sim)

    mean_cos = float(sum(sims) / len(sims))
    recall_acc = float(sum(1 for s in sims if s > 0.80) / len(sims) * 100.0)

    # Collision statistics on unique active slots
    flat_idx = idx.reshape(-1).cpu()
    unique_slots = flat_idx.unique().numel()
    slot_occupancy_pct = (unique_slots / total_slots) * 100.0

    return {
        "facts": num_facts,
        "total_slots": total_slots,
        "active_slots": active_slots,
        "sram_kb": sram_kb,
        "mean_cosine": mean_cos,
        "recall_acc": recall_acc,
        "write_time_sec": write_time,
        "unique_slots": unique_slots,
        "slot_occupancy_pct": slot_occupancy_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Axis Routing Upgrade Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 80)
    print("      HSRU DELTAGRID: MULTI-AXIS ROUTING UPGRADE BENCHMARK")
    print("=" * 80)
    print(f"Device: {args.device} (CUDA Fused Kernel: {_HAS_CUDA})")

    configs = [
        {
            "name": "2-Axis Baseline (1024 x 1024)",
            "grid_dim": (1024, 1024),
            "top_k": 8,
            "label": "2-Axis (1024x1024), K=8 [8 KB SRAM]",
            "color": "#1f77b4",
            "marker": "o",
        },
        {
            "name": "4-Axis Match (32 x 32 x 32 x 32)",
            "grid_dim": (32, 32, 32, 32),
            "top_k": 2,
            "label": "4-Axis (32^4 = 1.05M), K=2 [2 KB SRAM]",
            "color": "#2ca02c",
            "marker": "s",
        },
        {
            "name": "3-Axis Double (128 x 128 x 128)",
            "grid_dim": (128, 128, 128),
            "top_k": 3,
            "label": "3-Axis (128^3 = 2.10M), K=3 [3.4 KB SRAM]",
            "color": "#ff7f0e",
            "marker": "^",
        },
        {
            "name": "4-Axis Ultra (40 x 40 x 40 x 40)",
            "grid_dim": (40, 40, 40, 40),
            "top_k": 2,
            "label": "4-Axis Ultra (40^4 = 2.56M), K=2 [2 KB SRAM]",
            "color": "#d62728",
            "marker": "D",
        },
    ]

    fact_points = [1000, 3000, 5000, 10000, 20000, 50000, 100000]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n>>> Benchmarking {cfg_name} | Grid: {cfg['grid_dim']} | Top-K: {cfg['top_k']} <<<")
        results = []
        for facts in fact_points:
            res = evaluate_routing_config(
                num_facts=facts,
                grid_dim=cfg["grid_dim"],
                top_k=cfg["top_k"],
                device=args.device,
            )
            results.append(res)
            print(
                f"  Facts: {facts:>6} | Recall >0.8: {res['recall_acc']:>5.1f}% | "
                f"Mean Cos: {res['mean_cosine']:.4f} | SRAM: {res['sram_kb']:.1f} KB | "
                f"Time: {res['write_time_sec']:.2f}s"
            )
        all_results[cfg_name] = results

    # Save metrics JSON
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "multiaxis_routing_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved metrics to {json_path}")

    # Plot Comparison Graph
    if args.save_plot:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

        for cfg in configs:
            name = cfg["name"]
            res_list = all_results[name]
            x = [r["facts"] for r in res_list]
            recalls = [r["recall_acc"] for r in res_list]
            cosines = [r["mean_cosine"] for r in res_list]

            ax1.plot(
                x, recalls,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=2.2,
                markersize=6,
            )
            ax2.plot(
                x, cosines,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=2.2,
                markersize=6,
            )

        # Recall Subplot
        ax1.set_xscale("log")
        ax1.set_title("Associative Recall Rate (cos > 0.80) vs Stored Facts", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Sequentially Stored Facts", fontsize=11)
        ax1.set_ylabel("Recall Accuracy (%)", fontsize=11)
        ax1.set_ylim(-2, 105)
        ax1.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% Critical Knee")
        ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        # Mean Cosine Subplot
        ax2.set_xscale("log")
        ax2.set_title("Mean Cosine Similarity vs Stored Facts", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Sequentially Stored Facts", fontsize=11)
        ax2.set_ylabel("Mean Cosine Similarity", fontsize=11)
        ax2.set_ylim(0.4, 1.02)
        ax2.axhline(0.8, color="red", linestyle=":", alpha=0.6, label="0.80 Fidelity Floor")
        ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

        fig.suptitle(
            "DeltaGrid Routing Architecture Scaling: 2-Axis vs 3-Axis vs 4-Axis",
            fontsize=14, fontweight="bold", y=0.98,
        )
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "multiaxis_routing_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"[+] Saved comparison figure to {plot_path}")


if __name__ == "__main__":
    main()

