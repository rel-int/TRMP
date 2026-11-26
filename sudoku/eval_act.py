"""
Sudoku-extreme WITHOUT the lattice, test: a checkpoint of
``sudoku.train_act`` on the first 1,000 test puzzles, in three phases:
(1) deterministic -- one noise-free trajectory, the board after exactly
32 .. 2,048 segments; (2) a sigma scan -- 8 rollouts to 256 segments at
sigma 0.1 / 0.2 / 0.3 / 0.5 (Gaussian noise added to the answer trace
before every segment), the best best-of-8 wins; (3) 64 rollouts at that
sigma to 1,024 segments: noisy single, best-of-64 (the rollout with the
largest halt logit, each rollout answering with its most confident
segment) and pass@64 (any rollout right). The noise generators have fixed
seeds and the model shape comes from the checkpoint. fp32 throughout, on
the CUDA segment kernels where available (``--kernel auto``).

    python -m sudoku.eval_act runs/act_s8/best.pt

Writes ``test.json`` next to the checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.act import rollout
from discopy.neural.cell import as_widths, count
from discopy.neural.lattice import lattice_of

from kernel import Act
from sudoku import data, geometric

#: The seeds of the noise generators of the sigma scan and of the final
#: rollouts (rollout ``r`` uses ``seed + r``).
SCAN_SEED, BIG_SEED = 31337, 27182


def build(widths, rounds: int, cycles: int, detached: int,
          kernel: str = "auto") -> Act:
    """ An :class:`Act` on the sudoku peer table, on the blocks of the
    grid's symmetry group when ``kernel="geometric"``. """
    net = geometric.Act if kernel == "geometric" else Act
    return net(as_widths(widths), data.PEERS, 9, rounds, cycles, detached)


def ready(net: Act, kernel: str, compiled: bool) -> Act:
    """ Set the kernel policy and, on CUDA, compile the PyTorch cycle. """
    net.kernel = kernel
    if compiled and next(net.parameters()).is_cuda:
        net.compile(unroll=False, dynamic=False)
    return net


def load_checkpoint(path, device, kernel: str = "auto",
                    compiled: bool = True) -> tuple:
    """ ``(net, checkpoint)``: a checkpoint's model, built at the shape the
    checkpoint stores. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    shape = stored["shape"]
    net = build(shape["widths"], shape["rounds"], shape["cycles"],
                shape["detached"], kernel).to(device)
    net.load_state_dict(stored["state_dict"])
    return ready(net, kernel, compiled), stored


def lattices_of(puzzles, solutions, n: int, device) -> tuple:
    """ ``(x, target, given)`` of the first ``n`` puzzles of a split. """
    x = lattice_of(puzzles[:n]).to(device)
    target = torch.as_tensor(np.asarray(solutions[:n], dtype=np.int64),
                             device=device) - 1
    return x, target, x.sum(-1) == 1


def hours(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def noisy_rollouts(net, split, caps, sigma: float, k: int, seed: int,
                   batch: int) -> dict:
    """ ``k`` noisy rollouts to ``max(caps)``: per cap ``(C, Q)``, each
    ``(k, n)``, the most confident answer so far and its halt logit. """
    x, target, given = split
    C, Q = {c: [] for c in caps}, {c: [] for c in caps}
    for r in range(k):
        gen = torch.Generator(device=x.device).manual_seed(seed + r)
        cs, qs = {c: [] for c in caps}, {c: [] for c in caps}
        for s in range(0, len(x), batch):
            out = rollout(net, x[s:s + batch], target[s:s + batch],
                          given[s:s + batch], caps, sigma, gen)
            for c in caps:
                cs[c].append(out[c]["best"])
                qs[c].append(out[c]["best_q"])
        for c in caps:
            C[c].append(torch.cat(cs[c]).float())
            Q[c].append(torch.cat(qs[c]))
    return {c: (torch.stack(C[c]), torch.stack(Q[c])) for c in caps}


def fold(C, Q) -> dict:
    """ Single / best-of-k (the largest halt logit) / pass@k of ``(k, n)``. """
    col = torch.arange(C.shape[1], device=C.device)
    return {"single": float(C.mean()),
            "best_of_k": float(C[Q.argmax(0), col].mean()),
            "pass_at_k": float((C.sum(0) > 0).float().mean())}


def test(net, split, args, log) -> dict:
    """ The three test phases; returns everything and the headline. """
    net.eval()
    x, target, given = split
    det_caps = sorted(set(args.det_caps) | {args.big_cap})
    big_caps = sorted({c for c in args.big_caps if c <= args.big_cap}
                      | {args.big_cap})
    tick = time.perf_counter()
    log(f"[1/3] deterministic, caps {det_caps}")
    got = {c: {"board": [], "cell": []} for c in det_caps}
    for s in range(0, len(x), args.eval_batch):
        out = rollout(net, x[s:s + args.eval_batch],
                      target[s:s + args.eval_batch],
                      given[s:s + args.eval_batch], det_caps)
        for c in det_caps:
            got[c]["board"].append(out[c]["fixed"])
            got[c]["cell"].append(out[c]["cell"])
    fixed = {str(c): {key: float(torch.cat(v).float().mean())
                      for key, v in row.items()} for c, row in got.items()}
    for c in det_caps:
        log(f"   cap {c:5d}  board {fixed[str(c)]['board']:.4f}  "
            f"cell {fixed[str(c)]['cell']:.4f}")
    log(f"[2/3] sigma scan: k={args.scan_k} cap={args.scan_cap} "
        f"sigmas={args.scan_sigmas}")
    scan = {}
    for sigma in args.scan_sigmas:
        C, Q = noisy_rollouts(net, split, (args.scan_cap, ), sigma, args.scan_k,
                              SCAN_SEED, args.eval_batch)[args.scan_cap]
        scan[sigma] = fold(C, Q)
        log(f"   sigma {sigma:.2f}  single {scan[sigma]['single']:.4f}  "
            f"best_of_k {scan[sigma]['best_of_k']:.4f}  "
            f"pass@k {scan[sigma]['pass_at_k']:.4f}")
    sigma = max(scan, key=lambda s: scan[s]["best_of_k"])
    log(f"[3/3] {args.big_k} rollouts at sigma {sigma} to {args.big_cap}, "
        f"read at {big_caps}")
    rolled = noisy_rollouts(net, split, big_caps, sigma, args.big_k, BIG_SEED,
                            args.eval_batch)
    big = {str(c): fold(*rolled[c]) for c in big_caps}
    for c in big_caps:
        row = big[str(c)]
        log(f"   cap {c:5d}  single {row['single']:.4f}  best-of-{args.big_k} "
            f"{row['best_of_k']:.4f}  pass@{args.big_k} {row['pass_at_k']:.4f}")
    top = big[str(args.big_cap)]
    headline = {"deterministic": fixed[str(args.big_cap)]["board"],
                "noisy_single": top["single"],
                f"best_of_{args.big_k}": top["best_of_k"],
                f"pass_at_{args.big_k}": top["pass_at_k"],
                "sigma": sigma, "cap": args.big_cap, "n": len(x)}
    log(f"TEST @{args.big_cap}: deterministic {headline['deterministic']:.4f}"
        f"  noisy single {top['single']:.4f}  best-of-{args.big_k} "
        f"{top['best_of_k']:.4f}  pass@{args.big_k} {top['pass_at_k']:.4f}  "
        f"(sigma {sigma}, {hours(time.perf_counter() - tick)})")
    return {"headline": headline, "fixed": fixed,
            "sigma_scan": {str(s): v for s, v in scan.items()},
            "best_sigma": sigma, "big": big,
            "seconds": time.perf_counter() - tick}


def test_checkpoint(args, out: Path, log) -> dict:
    """ ``args.checkpoint`` through the three test phases. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    net, stored = load_checkpoint(args.checkpoint, device, args.kernel,
                                  not args.no_compile)
    log(f"{args.checkpoint}: {count(net):,} params, {net.rounds}x{net.cycles} "
        f"detached {net.detached}, step {stored.get('step')}, seed "
        f"{stored.get('seed')}")
    common = data.load(args.data)
    split = lattices_of(common["test_puzzles"], common["test_solutions"],
                        args.test_n, device)
    result = test(net, split, args, log)
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "out": str(out)},
            "step": stored.get("step"), "seed": stored.get("seed"),
            "test": result, **result["headline"]}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt written by sudoku.train_act")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $SUDOKU_DATA or data/)")
    p.add_argument("--kernel", default="auto",
                   choices=("auto", "off", "fp32", "tf32", "fma",
                            "geometric"))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--eval-batch", type=int, default=500,
                   help="puzzles per rollout batch")
    p.add_argument("--det-caps", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--scan-sigmas", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--scan-k", type=int, default=8)
    p.add_argument("--scan-cap", type=int, default=256)
    p.add_argument("--big-k", type=int, default=64)
    p.add_argument("--big-cap", type=int, default=1024)
    p.add_argument("--big-caps", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512, 1024])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)

    result = test_checkpoint(args, out, log)
    (out / "test.json").write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {out / 'test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
