"""
=============================================================================
HSRU Delta-PKM: Memory Phase Transition Empirical Benchmark & Plotter
=============================================================================
Evaluates retrieval accuracy (>0.80 cosine) and mean cosine similarity across
fact volumes from 100 up to 200,000 facts, comparing:
  - DeltaGrid 128x128 (16,384 virtual slots, Top-K=4 -> K=16 active)
  - Full PKM 1024x1024 (1,048,576 virtual slots, Top-K=8 -> K=64 active)

Points evaluated:
  128x128:   [100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
  1024x1024: [100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 50000, 100000, 200000]
=============================================================================
"""

import os
import sys
import time
import json
import argparse
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_sram_constrained import evaluate_associative_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="HSRU Delta-PKM Memory Phase Transition Benchmark")
    parser.add_argument("--grid", choices=["128", "1024", "all"], default="all", help="Grid configuration to benchmark")
    parser.add_argument("--force", action="store_true", help="Force recomputation instead of loading cached metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to execute benchmark on")
    return parser.parse_args()


def benchmark_grid(points, slots, grid_dim, top_k, label, device="cuda"):
    print(f"\n=== Benchmarking {label} ({slots:,} slots, Grid: {grid_dim}, Top-K: {top_k}^2={top_k*top_k}) ===")
    results = []
    for p in points:
        t0 = time.time()
        mean_cos, acc = evaluate_associative_accuracy(
            num_facts=p,
            slots=slots,
            head_dim=64,
            mode='pkm',
            grid_dim=grid_dim,
            top_k=top_k,
            device=device
        )
        dt = time.time() - t0
        results.append({
            "facts": p,
            "mean_cosine": mean_cos,
            "recall_acc": acc,
            "time_sec": dt
        })
        print(f"  Facts: {p:>6,} | Recall >0.8: {acc:>5.1f}% | Mean Cosine: {mean_cos:.4f} | Time: {dt:.2f}s")
    return results


def print_table(results, title):
    print("\n" + "=" * 88)
    print(f"  {title.upper()}")
    print("=" * 88)
    print(f"  {'Stored Facts':>12} | {'Recall (>0.80)':>14} | {'Mean Cosine':>12} | {'Time':>8} | Regime")
    print("  " + "-" * 84)
    for r in results:
        p = r["facts"]
        acc = r["recall_acc"]
        cos = r["mean_cosine"]
        t = r["time_sec"]
        if acc >= 98.0:
            regime = "Phase 1: Lossless Plateau"
        elif acc >= 90.0:
            regime = "*** PHASE TRANSITION KNEE ***"
        else:
            regime = "Phase 2: Sliding FIFO Window"
        print(f"  {p:>12,} | {acc:>13.1f}% | {cos:>12.4f} | {t:>7.2f}s | {regime}")
    print("=" * 88)


def main():
    args = parse_args()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "phase_transition_metrics_200k.json")

    points_128 = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
    points_1024 = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 50000, 100000, 200000]

    cached = {}
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path, "r") as f:
                cached = json.load(f)
            print(f"[+] Loaded existing cached metrics from {json_path}")
        except Exception:
            cached = {}

    results_128 = cached.get("grid_128x128", [])
    results_1024 = cached.get("grid_1024x1024", [])

    if args.grid in ("128", "all") and (not results_128 or args.force):
        results_128 = benchmark_grid(points_128, 16384, (128, 128), 4, "DeltaGrid 128x128", args.device)
    
    if args.grid in ("1024", "all") and (not results_1024 or len(results_1024) < len(points_1024) or args.force):
        results_1024 = benchmark_grid(points_1024, 1048576, (1024, 1024), 8, "Full PKM 1024x1024", args.device)

    # Save consolidated metrics
    data_to_save = {
        "points_128": points_128,
        "points_1024": points_1024,
        "grid_128x128": results_128,
        "grid_1024x1024": results_1024
    }
    with open(json_path, "w") as f:
        json.dump(data_to_save, f, indent=2)

    if results_128:
        print_table(results_128, "DeltaGrid 128x128 (16,384 Slots, K=16 Active)")
    if results_1024:
        print_table(results_1024, "Full PKM 1024x1024 (1,048,576 Slots, K=64 Active)")

    # Generate Comparative Dual-Panel Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Left: Recall > 0.80 (%)
    if results_128:
        pts_128 = [r["facts"] for r in results_128]
        acc_128 = [r["recall_acc"] for r in results_128]
        ax1.plot(pts_128, acc_128, marker='o', linewidth=2.5, markersize=7, color='#ff4757', label='DeltaGrid 128x128 (16k Slots, K=16)')
        ax1.axvline(x=2560, color='#ff4757', linestyle='--', linewidth=1.5, alpha=0.7, label='16k Phase Knee (~2,560 facts)')

    if results_1024:
        pts_1024 = [r["facts"] for r in results_1024]
        acc_1024 = [r["recall_acc"] for r in results_1024]
        ax1.plot(pts_1024, acc_1024, marker='s', linewidth=2.5, markersize=7, color='#2ed573', label='Full PKM 1024x1024 (1M Slots, K=64)')
        ax1.axvline(x=70000, color='#2ed573', linestyle='--', linewidth=1.5, alpha=0.7, label='1M Phase Knee (~70k facts)')

    ax1.set_xscale('log')
    ax1.set_ylim(-5, 105)
    ax1.set_title("Associative Recall Accuracy (>0.80 Cosine) vs. Stored Facts", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Stored Facts (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", framealpha=0.95, fontsize=9.0)

    # Right: Mean Cosine Similarity
    if results_128:
        cos_128 = [r["mean_cosine"] for r in results_128]
        ax2.plot(pts_128, cos_128, marker='o', linewidth=2.5, markersize=7, color='#ff4757', label='DeltaGrid 128x128 (16k Slots, K=16)')
        ax2.axvline(x=2560, color='#ff4757', linestyle='--', linewidth=1.5, alpha=0.7)

    if results_1024:
        cos_1024 = [r["mean_cosine"] for r in results_1024]
        ax2.plot(pts_1024, cos_1024, marker='s', linewidth=2.5, markersize=7, color='#2ed573', label='Full PKM 1024x1024 (1M Slots, K=64)')
        ax2.axvline(x=70000, color='#2ed573', linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_xscale('log')
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Mean Cosine Similarity vs. Stored Facts", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Stored Facts (Log Scale)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Mean Cosine Similarity", fontsize=11, fontweight='bold')
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=9.0)

    plot_path = os.path.join(out_dir, "phase_transition_comprehensive_200k.png")
    fig.suptitle("HSRU Delta-PKM: Memory Phase Transition Empirical Benchmark (100 to 200,000 Facts)", fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] Comprehensive plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
