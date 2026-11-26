"""
SAT WITHOUT the lattice, test: a checkpoint of ``sat.train_act`` on the
full 10,000-formula test split of its family, in three readings -- the
deterministic halt-rule decode (``k = 1``), and best-of-8 and best-of-64
noisy rollouts (Gaussian noise of sigma 0.3 on the answer trace before
every segment) ranked by the halt head. Every rollout halts at the first
segment whose instance halt logit clears the threshold, or at the cap of
16, and answers with its most confident segment.

    python -m sat.eval_act runs/act_sr_easy/best.pt
    python -m sat.eval_act runs/act_sr_medium/best.pt --splits medium hard

``--splits`` names the difficulties to decode and defaults to the one the
checkpoint trained on; naming ``hard`` tests the zero-shot transfer. Each
``k`` reports ``single`` (the mean over the rollouts), ``best_of_k`` (the
one the halt head ranks highest) and ``pass_at_k`` (any rollout right) --
the gap between the last two is what the head's calibration is worth. A
solve is an exact clause check, and the halt rule never reads it.

Rollouts run side by side, ``k`` copies of a formula in one wide batch, so
the batch shrinks with ``k`` to keep the product inside one device. The
whole ``k = 1, 8, 64`` sweep is about 35 minutes on the easy splits and
several hours on medium or hard, where ``--k 1`` alone is minutes. fp32
throughout (no TF32: the reported runs had it off), never compiled. Writes ``test_<checkpoint stem>.json`` next to
the checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from discopy.neural.cell import count
from sat import act, data
from sat.solve import hours


def load_checkpoint(path, device):
    """ ``(net, stored)``: the ACT model a checkpoint stores, at its
    shape. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    net = act.build(**stored["shape"]).to(device).eval()
    net.load_state_dict(stored["state_dict"])
    return net, stored


def chunk_for(k: int, mean_vars: float, chunk: int | None) -> int:
    """
    Formulas per rollout batch. One batch holds ``k`` copies of every
    formula, so the width that fits is set by ``k * formulas * variables``;
    this keeps that product roughly constant across the splits, which
    differ by an order of magnitude in size.
    """
    base = chunk or max(20, int(200 * 150 / max(mean_vars, 1.0)))
    return max(2, base // max(1, round(k / 2.5)))


@torch.no_grad()
def score(net, pool, k: int, cap: int, sigma: float, threshold: float,
          seed: int, chunk: int) -> dict:
    """ One reading of a pool: the mean over its chunks, weighted by the
    formulas in each. """
    device = next(net.parameters()).device
    totals, n = {}, 0
    for start in range(0, len(pool), chunk):
        batch = data.to_device(data.collate(pool[start:start + chunk]), device)
        if k == 1:
            got = act.evaluate_act(net, batch, cap, threshold)
        else:
            got = act.evaluate_best_of_k(net, batch, cap, k, sigma, threshold,
                                         seed)
        for key, value in got.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch.n_instances
        n += batch.n_instances
    return {key: value / max(n, 1) for key, value in totals.items()}


def test_checkpoint(args, log) -> dict:
    """ ``args.checkpoint`` read at every ``--k`` on every ``--splits``
    difficulty. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, stored = load_checkpoint(args.checkpoint, device)
    family = stored.get("family", "sr")
    splits = args.splits or [stored.get("difficulty", "easy")]
    log(f"{args.checkpoint}: {count(net):,} params, shape {net.shape()}, "
        f"{family} trained on {stored.get('difficulty')}, step "
        f"{stored.get('step')}, seed {stored.get('seed')}, device {device}")
    out = {}
    for difficulty in splits:
        pool = data.load(family, difficulty, "test", args.data, args.test_n,
                         args.seed, log)
        mean_vars = sum(i.n_vars for i in pool) / len(pool)
        out[difficulty] = {}
        for k in args.k:
            chunk = chunk_for(k, mean_vars, args.chunk)
            tick = time.perf_counter()
            got = score(net, pool, k, args.cap, args.sigma, args.threshold,
                        args.seed, chunk)
            got["seconds"] = time.perf_counter() - tick
            got["chunk"] = chunk
            out[difficulty][f"k{k}"] = got
            headline = got["solved"] if k == 1 else got["best_of_k"]
            extra = (f"depth {got['depth']:.2f}" if k == 1 else
                     f"single {got['single']:.4f}  pass@{k} "
                     f"{got['pass_at_k']:.4f}")
            log(f"{difficulty} k={k:2d}: {headline:.4f}  {extra}  "
                f"({len(pool)} formulas, {hours(got['seconds'])})")
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "data": str(data.root(args.data))},
            "shape": net.shape(), "family": family,
            "trained_on": stored.get("difficulty"), "step": stored.get("step"),
            "seed": stored.get("seed"), "test": out,
            "solve_rate": {f"{d}.k{k}":
                           (r[f"k{k}"]["solved"] if k == 1
                            else r[f"k{k}"]["best_of_k"])
                           for d, r in out.items() for k in args.k}}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt of sat.train_act")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $SAT_DATA or "
                        "data/g4satbench)")
    p.add_argument("--splits", nargs="+", default=None,
                   choices=("easy", "medium", "hard"),
                   help="difficulties to decode (default: the checkpoint's; "
                        "hard is the zero-shot transfer)")
    p.add_argument("--k", type=int, nargs="+", default=(1, 8, 64),
                   help="rollouts per formula (default: 1 8 64)")
    p.add_argument("--cap", type=int, default=16,
                   help="segment cap of a rollout")
    p.add_argument("--sigma", type=float, default=0.3,
                   help="answer-trace noise, for k > 1")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="the instance halt logit the halt rule must clear")
    p.add_argument("--seed", type=int, default=123,
                   help="seed of the noise (default: 123)")
    p.add_argument("--chunk", type=int, default=None,
                   help="formulas per rollout batch at k = 1 (default: from "
                        "the split's mean size); it shrinks with k")
    p.add_argument("--test-n", type=int, default=None,
                   help="a seeded subset of N formulas of each split (a "
                        "smoke run); omit for the whole split")
    p.add_argument("--no-compile", action="store_true",
                   help="accepted for a uniform CLI; sat is never compiled")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)

    result = test_checkpoint(args, log)
    result["config"]["out"] = str(out)
    path = out / f"test_{Path(args.checkpoint).stem}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
