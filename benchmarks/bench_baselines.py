"""
=============================================================================
  DeltaGrid Rigorous Benchmark Suite  (v2)
=============================================================================
Fixes the weaknesses of the original 6-panel dashboard:

  * Baselines      : every task is also run on GRU, LSTM and a causal Transformer
                     (parameter-matched to DeltaGrid, identical optimiser/steps).
  * Random layout  : needles / pairs sit at RANDOM positions (the old tasks put the
                     answer at a fixed index, which a model can learn without memory).
  * Distractors    : multiple key/value pairs, only one is queried.
  * Generalisation : train short -> test long (length), unseen frequencies (OOD),
                     input noise, and more pairs than seen in training (capacity).
  * Statistics     : N seeds, mean +- std (the old dashboard was a single run).
  * Held-out data  : all numbers are on freshly sampled evaluation data, never
                     training-batch loss.
  * Real data      : byte-level language modelling on Python stdlib source code.
  * Real memory    : peak RSS measured in isolated subprocesses (the old memory plot
                     was 0.0 MB on non-CUDA devices, i.e. measured nothing).

Tasks:  mqar | needle | adding | sine | parity | charlm | memory
Usage:  python tests/benchmark.py --tasks mqar needle adding sine
        python tests/benchmark.py --quick          # smoke test
=============================================================================
"""
import argparse
import json
import math
import os
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.delta_grid import DeltaGridLM, DeltaGridContinuous  # noqa: E402

DEVICE = torch.device("cpu")  # CPU: deterministic and works everywhere (no CUDA/MPS kernel gaps)
KINDS = ["deltagrid", "gru", "lstm", "transformer"]


# =============================================================================
# Baselines
# =============================================================================
def sinusoid(t: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(t, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(t, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Baseline(nn.Module):
    """GRU / LSTM / causal Transformer with the same I/O contract as DeltaGrid."""

    def __init__(self, kind, d, layers, vocab=None, in_dim=None, out_dim=None, horizon=None):
        super().__init__()
        self.kind, self.d, self.horizon, self.out_dim = kind, d, horizon, out_dim
        self.inp = nn.Embedding(vocab, d) if vocab else nn.Linear(in_dim, d)
        if kind == "gru":
            self.core = nn.GRU(d, d, layers, batch_first=True)
        elif kind == "lstm":
            self.core = nn.LSTM(d, d, layers, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab if vocab else out_dim)
        self.fhead = None
        if horizon:
            self.fhead = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, horizon * out_dim))

    def hidden(self, x):
        h = self.inp(x)
        if self.kind == "transformer":
            t = h.size(1)
            h = h + sinusoid(t, self.d, h.device)
            mask = nn.Transformer.generate_square_subsequent_mask(t, device=h.device)
            h = self.core(h, mask=mask, is_causal=True)
        else:
            h, _ = self.core(h)
        return self.norm(h)

    def forward(self, x, return_forecast=False):
        h = self.hidden(x)
        out = self.out(h)
        if return_forecast:
            fc = self.fhead(h[:, -1]).view(x.size(0), self.horizon, self.out_dim)
            return out, fc
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


D_CANDIDATES = [24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256]


def build(kind: str, mode: str, *, vocab=None, in_dim=None, out_dim=None, horizon=None,
          dg_d=64, layers=2, heads=2) -> nn.Module:
    """Build a model; baselines get the width whose parameter count is closest to DeltaGrid's."""

    def make_dg():
        if mode == "tok":
            return DeltaGridLM(vocab_size=vocab, d_model=dg_d, num_layers=layers, num_heads=heads)
        return DeltaGridContinuous(input_dim=in_dim, output_dim=out_dim, d_model=dg_d, num_layers=layers,
                                   num_heads=heads, forecast_horizon=horizon)

    if kind == "deltagrid":
        return make_dg().to(DEVICE)
    target = n_params(make_dg())
    best = min(D_CANDIDATES, key=lambda d: abs(n_params(Baseline(kind, d, layers, vocab, in_dim, out_dim, horizon)) - target))
    return Baseline(kind, best, layers, vocab, in_dim, out_dim, horizon).to(DEVICE)


def logits_of(model, x):
    out = model(x)
    if hasattr(out, "logits"):
        return out.logits
    return out[0] if isinstance(out, tuple) else out


def cont_forward(model, x, forecast=False):
    """-> (step_out [B,T,out], forecast [B,H,out] or None) for any model kind."""
    if forecast:
        r = model(x, return_forecast=True)
        if hasattr(r, "out") and hasattr(r, "forecast"):
            return r.out, r.forecast
        return (r[0], r[2]) if isinstance(model, DeltaGridContinuous) else (r[0], r[1])
    r = model(x)
    if hasattr(r, "out"):
        return r.out, None
    return (r[0] if isinstance(r, tuple) else r), None


# =============================================================================
# Training harness
# =============================================================================
def fit(model, sample: Callable, loss_fn: Callable, steps: int, lr: float, seed: int, log=None) -> float:
    gen = torch.Generator().manual_seed(seed + 12345)
    lr = lr * float(os.environ.get("BENCH_LR_MULT", "1.0"))  # lets you sweep learning rates for fairness checks
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    warm = max(1, steps // 20)
    t0 = time.time()
    model.train()
    for step in range(steps):
        scale = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * scale
        loss = loss_fn(model, sample(gen, step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step + 1) % max(1, steps // 4) == 0:
            print(f"      {log} step {step + 1:>5}/{steps} loss={loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return time.time() - t0


def ms(vals: List[float]) -> Tuple[float, float]:
    t = torch.tensor(vals, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


# =============================================================================
# TASK: MQAR  (Multi-Query Associative Recall - Arora et al., "Zoology", 2023)
# =============================================================================
MQ_K, MQ_V = 96, 96
MQ_VOCAB = 1 + MQ_K + MQ_V


def gen_mqar(bs: int, n_pairs: int, n_q: int, gen) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.argsort(torch.rand(bs, MQ_K, generator=gen), dim=1)[:, :n_pairs] + 1
    vals = torch.randint(0, MQ_V, (bs, n_pairs), generator=gen) + 1 + MQ_K
    kv = torch.stack([keys, vals], -1).reshape(bs, 2 * n_pairs)
    qi = torch.randint(0, n_pairs, (bs, n_q), generator=gen)
    q, tgt = keys.gather(1, qi), vals.gather(1, qi)
    return torch.cat([kv, q], 1), tgt


def run_mqar(kind, seed, steps, bs=32):
    train_pairs = [4, 8, 16, 24]
    model = build(kind, "tok", vocab=MQ_VOCAB)

    def sample(gen, step):
        n = train_pairs[step % len(train_pairs)]
        return gen_mqar(bs, n, n, gen) + (n,)

    def loss_fn(m, b):
        x, y, n = b
        lg = logits_of(m, x)[:, 2 * n:]
        return F.cross_entropy(lg.reshape(-1, MQ_VOCAB), y.reshape(-1))

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[mqar/{kind}/s{seed}]")
    res = {}
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        for n in [4, 8, 16, 24, 32, 48]:
            x, y = gen_mqar(256, n, n, gen)
            pred = logits_of(model, x)[:, 2 * n:].argmax(-1)
            res[f"pairs={n}"] = (pred == y).float().mean().item() * 100
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: NEEDLE  (random position + distractor pairs + train short / test long)
# =============================================================================
ND_K, ND_V, ND_NOISE = 16, 32, 32
ND_QUERY = 1 + ND_K + ND_V + ND_NOISE
ND_VOCAB = ND_QUERY + 1


def gen_needle(bs: int, L: int, gen, n_pairs: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(1 + ND_K + ND_V, 1 + ND_K + ND_V + ND_NOISE, (bs, L), generator=gen)
    keys = torch.argsort(torch.rand(bs, ND_K, generator=gen), dim=1)[:, :n_pairs] + 1
    vals = torch.randint(0, ND_V, (bs, n_pairs), generator=gen) + 1 + ND_K
    slots = torch.argsort(torch.rand(bs, L // 2 - 1, generator=gen), dim=1)[:, :n_pairs] * 2  # random, non-overlapping
    rows = torch.arange(bs).unsqueeze(1)
    x[rows, slots] = keys
    x[rows, slots + 1] = vals
    qi = torch.randint(0, n_pairs, (bs,), generator=gen)
    qk, tv = keys[torch.arange(bs), qi], vals[torch.arange(bs), qi]
    x[:, -2], x[:, -1] = ND_QUERY, qk
    return x, tv


def run_needle(kind, seed, steps, bs=32):
    train_lens = [32, 64]
    model = build(kind, "tok", vocab=ND_VOCAB)

    def sample(gen, step):
        return gen_needle(bs, train_lens[step % 2], gen)

    def loss_fn(m, b):
        return F.cross_entropy(logits_of(m, b[0])[:, -1], b[1])

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[needle/{kind}/s{seed}]")
    res = {}
    gen = torch.Generator().manual_seed(777)
    with torch.no_grad():
        for L in [64, 128, 256, 512, 1024, 2048]:
            correct = 0
            for _ in range(4):
                x, y = gen_needle(64, L, gen)
                correct += (logits_of(model, x)[:, -1].argmax(-1) == y).sum().item()
            res[f"L={L}"] = correct / 256 * 100
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: ADDING  (train T=50, test T up to 800; metric = MSE on fresh data)
# =============================================================================
def gen_adding(bs: int, T: int, gen):
    vals = torch.rand(bs, T, generator=gen)
    idx = torch.argsort(torch.rand(bs, T, generator=gen), dim=1)[:, :2]
    mark = torch.zeros(bs, T)
    mark.scatter_(1, idx, 1.0)
    tgt = vals.gather(1, idx).sum(1, keepdim=True)
    return torch.stack([vals, mark], -1), tgt


def run_adding(kind, seed, steps, bs=64):
    model = build(kind, "cont", in_dim=2, out_dim=1)

    def sample(gen, step):
        return gen_adding(bs, 50, gen)

    def loss_fn(m, b):
        return F.mse_loss(cont_forward(m, b[0])[0][:, -1], b[1])

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[adding/{kind}/s{seed}]")
    res = {}
    gen = torch.Generator().manual_seed(555)
    with torch.no_grad():
        for T in [50, 100, 200, 400, 800]:
            x, y = gen_adding(256, T, gen)
            res[f"T={T}"] = F.mse_loss(cont_forward(model, x)[0][:, -1], y).item()
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: SINE  (forecast; held-out phase, OOD frequency, noisy input, naive baselines)
# =============================================================================
def gen_wave(bs: int, L: int, gen, fmin=0.05, fmax=0.25):
    f = fmin + (fmax - fmin) * torch.rand(bs, 1, generator=gen)
    ph = 2 * math.pi * torch.rand(bs, 1, generator=gen)
    t = torch.arange(L + 1, dtype=torch.float32).unsqueeze(0)
    w = torch.sin(t * f + ph) + 0.5 * torch.cos(t * f * 2.5) + 0.25 * torch.sin(t * f * 4.2)
    return w.unsqueeze(-1)  # [bs, L+1, 1]


def run_sine(kind, seed, steps, bs=32):
    ctx, hor = 32, 32
    model = build(kind, "cont", in_dim=1, out_dim=1, horizon=hor, dg_d=128, heads=4)

    def sample(gen, step):
        return gen_wave(bs, ctx + hor, gen)

    def loss_fn(m, w):
        out, fc = cont_forward(m, w[:, :ctx], forecast=True)
        return F.mse_loss(out, w[:, 1:ctx + 1]) + 2.0 * F.mse_loss(fc, w[:, ctx:ctx + hor])

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[sine/{kind}/s{seed}]")
    gen = torch.Generator().manual_seed(333)
    res = {}
    scenarios = {"in-dist": dict(fmin=0.05, fmax=0.25, noise=0.0),
                 "OOD-freq(0.25-0.35)": dict(fmin=0.25, fmax=0.35, noise=0.0),
                 "noisy(0.05)": dict(fmin=0.05, fmax=0.25, noise=0.05)}
    with torch.no_grad():
        for name, sc in scenarios.items():
            errs, hold, zero = [], [], []
            for _ in range(8):
                w = gen_wave(128, ctx + hor, gen, sc["fmin"], sc["fmax"])
                xin = w[:, :ctx] + sc["noise"] * torch.randn(128, ctx, 1, generator=gen)
                tgt = w[:, ctx:ctx + hor]
                errs.append(F.mse_loss(cont_forward(model, xin, forecast=True)[1], tgt).item())
                hold.append(F.mse_loss(w[:, ctx - 1:ctx].expand(-1, hor, -1), tgt).item())
                zero.append((tgt ** 2).mean().item())
            res[name] = sum(errs) / 8
            if kind == "deltagrid":  # naive baselines are model independent; store once
                res[name + " [hold-last]"] = sum(hold) / 8
                res[name + " [predict-0]"] = sum(zero) / 8
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: PARITY  (state-tracking limit probe; linear RNNs with eigenvalues in [0,1] are known to struggle)
# =============================================================================
def run_parity(kind, seed, steps, bs=64):
    model = build(kind, "tok", vocab=2)

    def sample(gen, step):
        T = 8 + (step % 3) * 4  # 8, 12, 16
        x = torch.randint(0, 2, (bs, T), generator=gen)
        return x, x.sum(1) % 2

    def loss_fn(m, b):
        return F.cross_entropy(logits_of(m, b[0])[:, -1], b[1])

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[parity/{kind}/s{seed}]")
    res = {}
    gen = torch.Generator().manual_seed(111)
    with torch.no_grad():
        for T in [8, 16, 32, 64]:
            x = torch.randint(0, 2, (512, T), generator=gen)
            res[f"T={T}"] = (logits_of(model, x)[:, -1].argmax(-1) == x.sum(1) % 2).float().mean().item() * 100
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: CHARLM  (real data: byte-level LM on Python standard-library source)
# =============================================================================
def load_corpus(train_bytes=3_000_000, val_bytes=300_000):
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("test", "tests", "site-packages", "idlelib", "turtledemo", "__pycache__")]
        files += [os.path.join(dp, f) for f in fn if f.endswith(".py")]
    files.sort()
    blobs = []
    for f in files:
        try:
            blobs.append(open(f, "rb").read())
        except OSError:
            pass
    split = int(len(blobs) * 0.9)  # file-level split -> no leakage between train and val
    tr = b"\n".join(blobs[:split])[:train_bytes]
    va = b"\n".join(blobs[split:])[:val_bytes]
    return torch.tensor(list(tr), dtype=torch.long), torch.tensor(list(va), dtype=torch.long)


def run_charlm(kind, seed, steps, bs=16, seq=256):
    train, val = load_corpus()
    model = build(kind, "tok", vocab=256, dg_d=96, heads=4)

    def sample(gen, step):
        ix = torch.randint(0, len(train) - seq - 1, (bs,), generator=gen)
        x = torch.stack([train[i:i + seq + 1] for i in ix])
        return x[:, :-1], x[:, 1:]

    def loss_fn(m, b):
        return F.cross_entropy(logits_of(m, b[0]).reshape(-1, 256), b[1].reshape(-1))

    secs = fit(model, sample, loss_fn, steps, 3e-3, seed, log=f"[charlm/{kind}/s{seed}]")
    res = {}
    with torch.no_grad():
        for ctx in (256, 1024):
            tot, n = 0.0, 0
            for i in range(0, min(len(val) - ctx - 1, 40 * ctx), ctx):
                x = val[i:i + ctx + 1].unsqueeze(0)
                tot += F.cross_entropy(logits_of(model, x[:, :-1])[0], x[0, 1:], reduction="sum").item()
                n += ctx
            res[f"val_bpc@ctx{ctx}"] = tot / n / math.log(2)
    return {"metrics": res, "params": n_params(model), "train_s": secs}


# =============================================================================
# TASK: MEMORY  (real peak RSS in isolated subprocesses + analytic decode state)
# =============================================================================
def _rss_mb() -> float:
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def memprobe(mode: str, length: int):
    torch.manual_seed(0)
    d, layers, heads, vocab = 128, 4, 4, 1000
    with torch.no_grad():
        if mode.startswith("dg"):
            model = DeltaGridLM(vocab, d_model=d, num_layers=layers, num_heads=heads).eval()
        else:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            model = nn.ModuleDict(dict(e=nn.Embedding(vocab, d), t=nn.TransformerEncoder(layer, layers, enable_nested_tensor=False))).eval()
        base = _rss_mb()
        toks = torch.randint(0, vocab, (1, length))
        if mode == "dg_stream":  # constant-state streaming in 256-token chunks
            states, off = None, 0
            for i in range(0, length, 256):
                _, states = model(toks[:, i:i + 256], states=states, seq_offset=off)
                off += 256
        elif mode == "dg_full":
            model(toks)
        else:
            t = length
            mask = nn.Transformer.generate_square_subsequent_mask(t)
            model.t(model.e(toks), mask=mask, is_causal=True)
    print(json.dumps({"mode": mode, "length": length, "peak_delta_mb": round(_rss_mb() - base, 1)}))


def run_memory(_kind=None, _seed=None, _steps=None):
    rows = []
    for length in (1024, 4096, 8192, 16384, 65536):
        row = {"length": length}
        for mode in ("dg_stream", "dg_full", "tf_full"):
            if mode == "tf_full" and length > 16384:
                row[mode] = None
                continue
            if mode == "dg_full" and length > 16384:
                row[mode] = None
                continue
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--memprobe", mode, str(length)],
                               capture_output=True, text=True, timeout=900)
            try:
                row[mode] = json.loads(p.stdout.strip().splitlines()[-1])["peak_delta_mb"]
            except Exception:
                row[mode] = "ERR"
        rows.append(row)
        print("   ", row, flush=True)
    d, layers, heads = 128, 4, 4
    dh = d // heads
    state_kb = layers * heads * dh * dh * 4 / 1024
    kv = {L: layers * 2 * L * d * 4 / 1024 / 1024 for L in (1024, 4096, 16384, 65536, 1_000_000)}
    return {"rows": rows, "delta_state_KB(const)": state_kb, "transformer_kv_cache_MB": kv}


TASKS = {"mqar": (run_mqar, 1500), "needle": (run_needle, 1500), "adding": (run_adding, 1200),
         "sine": (run_sine, 700), "parity": (run_parity, 1200), "charlm": (run_charlm, 800)}


# =============================================================================
# Reporting
# =============================================================================
def summarise(task: str, per_kind: Dict[str, List[dict]]):
    keys = list(next(iter(per_kind.values()))[0]["metrics"].keys())
    keys = [k for k in keys if "[" not in k]
    lower_better = task in ("adding", "sine", "charlm")
    print(f"\n{'=' * 100}\n{task.upper()}   ({'lower' if lower_better else 'higher'} is better; mean +- std over seeds)\n{'=' * 100}")
    print(f"{'model':<12}{'params':>9} | " + " ".join(f"{k:>20}" for k in keys))
    for kind, runs in per_kind.items():
        cells = []
        for k in keys:
            m, s = ms([r["metrics"][k] for r in runs])
            cells.append(f"{m:>12.4f}+-{s:<6.3f}" if lower_better else f"{m:>12.1f}+-{s:<6.1f}")
        print(f"{kind:<12}{runs[0]['params']:>9,} | " + " ".join(cells))
    if "deltagrid" in per_kind:
        extra = [k for k in per_kind["deltagrid"][0]["metrics"] if "[" in k]
        for k in extra:
            m, _ = ms([r["metrics"][k] for r in per_kind["deltagrid"]])
            print(f"  (naive) {k}: {m:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["mqar", "needle", "adding", "sine", "memory", "charlm"], choices=list(TASKS) + ["memory"])
    ap.add_argument("--models", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--out", default="figures/rigorous_results.json")
    ap.add_argument("--memprobe", nargs=2, metavar=("MODE", "LEN"))
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    a = ap.parse_args()

    if a.memprobe:
        memprobe(a.memprobe[0], int(a.memprobe[1]))
        return
    if a.quick:
        a.seeds, a.steps_scale = 1, 0.03

    torch.set_num_threads(a.threads if a.threads else max(1, (os.cpu_count() or 4)))
    out: Dict[str, dict] = {}
    t_all = time.time()
    for task in a.tasks:
        if task == "memory":
            print(f"\n{'=' * 100}\nMEMORY (peak RSS delta in MB, isolated subprocess; batch=1, d=128, 4 layers)\n{'=' * 100}")
            out["memory"] = run_memory()
            print(json.dumps({k: v for k, v in out["memory"].items() if k != "rows"}, indent=1))
            continue
        fn, steps = TASKS[task]
        steps = max(20, int(steps * a.steps_scale))
        per_kind: Dict[str, List[dict]] = {}
        for kind in a.models:
            per_kind[kind] = []
            for s in range(a.seeds):
                torch.manual_seed(s)
                per_kind[kind].append(fn(kind, s, steps))
        summarise(task, per_kind)
        out[task] = {k: v for k, v in per_kind.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    existing: Dict[str, dict] = {}
    if os.path.exists(a.out):
        try:
            with open(a.out, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(out)
    with open(a.out, "w") as f:
        json.dump(existing, f, indent=1)
    print(f"\n[+] raw results -> {a.out}   (total {time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
