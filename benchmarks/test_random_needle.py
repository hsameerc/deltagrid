import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hsru_delta import DeltaGridLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ND_K, ND_V, ND_NOISE = 16, 32, 32
ND_QUERY = 1 + ND_K + ND_V + ND_NOISE
ND_VOCAB = ND_QUERY + 1

def gen_needle_random(bs: int, L: int, gen, n_pairs: int = 4, depth_ratio: float = None):
    """
    Generates a Needle-in-a-Haystack sequence.
    If depth_ratio is None: the needle is placed at a completely random position.
    If depth_ratio in [0.0, 1.0]: the needle is placed specifically at that relative depth.
    """
    x = torch.randint(1 + ND_K + ND_V, 1 + ND_K + ND_V + ND_NOISE, (bs, L), generator=gen)
    keys = torch.argsort(torch.rand(bs, ND_K, generator=gen), dim=1)[:, :n_pairs] + 1
    vals = torch.randint(0, ND_V, (bs, n_pairs), generator=gen) + 1 + ND_K

    if depth_ratio is None:
        # Uniformly random non-overlapping slot positions across sequence
        slots = torch.argsort(torch.rand(bs, L // 2 - 1, generator=gen), dim=1)[:, :n_pairs] * 2
    else:
        # Targeted depth bucket
        target_pos = max(0, min(int((L - 4) * depth_ratio), L - 4))
        target_pos = (target_pos // 2) * 2  # align to even
        slots = torch.zeros(bs, n_pairs, dtype=torch.long)
        slots[:, 0] = target_pos
        # Distractors randomly elsewhere
        other_slots = torch.argsort(torch.rand(bs, L // 2 - 1, generator=gen), dim=1)[:, :n_pairs - 1] * 2
        for i in range(1, n_pairs):
            slots[:, i] = (other_slots[:, i - 1] + 2) % (L - 4)

    rows = torch.arange(bs).unsqueeze(1)
    x[rows, slots] = keys
    x[rows, slots + 1] = vals

    # Target is needle at slots[:, 0]
    target_key = keys[:, 0]
    target_val = vals[:, 0]

    # Query at sequence end
    x[:, -2] = ND_QUERY
    x[:, -1] = target_key
    return x.to(DEVICE), target_val.to(DEVICE)

def main():
    print("=" * 70)
    print("      DELTAGRID NEEDLE-IN-A-HAYSTACK DEPTH & RANDOM POSITION TEST")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Build DeltaGridLM
    model = DeltaGridLM(vocab_size=ND_VOCAB, d_model=64, num_layers=2, num_heads=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

    gen = torch.Generator(device="cpu").manual_seed(42)

    print("\nTraining DeltaGrid on random-position needles (L in [32, 64], 1500 steps)...")
    train_lens = [32, 64]
    t0 = time.time()
    for step in range(1, 1501):
        L = train_lens[step % 2]
        x, y = gen_needle_random(32, L, gen)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = F.cross_entropy(logits[:, -1], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"  Step {step:4d}/1500 | Loss: {loss.item():.4f} | Time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("EVALUATION 1: UNIFORM RANDOM POSITION NEEDLE (L=64 to L=2048)")
    print("Trained on L<=64 -> Extrapolating up to L=2048")
    print("=" * 70)

    model.eval()
    eval_gen = torch.Generator(device="cpu").manual_seed(999)
    with torch.no_grad():
        for L in [64, 128, 256, 512, 1024, 2048]:
            correct = 0
            total = 256
            for _ in range(4):
                x, y = gen_needle_random(64, L, eval_gen)
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                pred = logits[:, -1].argmax(-1)
                correct += (pred == y).sum().item()
            acc = correct / total * 100
            print(f"  Sequence Length L={L:4d} | Random Position Accuracy: {acc:5.1f}%")

    print("\n" + "=" * 70)
    print("EVALUATION 2: DEPTH-WISE NEEDLE ACCURACY AT L=1024 (16x Extrapolation)")
    print("=" * 70)

    depth_ratios = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
    with torch.no_grad():
        for d in depth_ratios:
            correct = 0
            total = 256
            for _ in range(4):
                x, y = gen_needle_random(64, 1024, eval_gen, depth_ratio=d)
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                pred = logits[:, -1].argmax(-1)
                correct += (pred == y).sum().item()
            acc = correct / total * 100
            pos_approx = int(1024 * d)
            print(f"  Needle Depth: {int(d*100):2d}% (Token ~{pos_approx:4d}) | Accuracy: {acc:5.1f}%")

    print("\n" + "=" * 70)
    print("EVALUATION 3: DEPTH-WISE NEEDLE ACCURACY AT L=2048 (32x Extrapolation)")
    print("=" * 70)

    with torch.no_grad():
        for d in depth_ratios:
            correct = 0
            total = 256
            for _ in range(4):
                x, y = gen_needle_random(64, 2048, eval_gen, depth_ratio=d)
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                pred = logits[:, -1].argmax(-1)
                correct += (pred == y).sum().item()
            acc = correct / total * 100
            pos_approx = int(2048 * d)
            print(f"  Needle Depth: {int(d*100):2d}% (Token ~{pos_approx:4d}) | Accuracy: {acc:5.1f}%")

    print("=" * 70)

if __name__ == "__main__":
    main()

