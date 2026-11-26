"""
SAT WITH the lattice, test: a checkpoint of ``sat.train_lattice`` decoded
by the repair search on the full 10,000-formula test split of its family,
once, at 8 chains and 40 model calls per formula.

    python -m sat.eval_lattice runs/sat_sr_easy/best.pt
    python -m sat.eval_lattice runs/sat_sr_medium/best.pt --splits medium hard

``--splits`` names the difficulties to decode and defaults to the one the
checkpoint trained on. Naming ``hard`` tests the zero-shot transfer: no
parameter depends on the number of variables, so a medium-trained model
runs on 200 to 400 variable formulas unchanged. Hard formulas are large,
so they decode in smaller batches (``--chunk``) to keep 8 tiled chains
inside one device.

Every accepted assignment is verified against the formula, so the search
reports solved or timed out and never a wrong answer -- ``wrong`` in the
log line is a guarantee that reads 0, not a measurement. fp32 throughout
(no TF32: the reported runs had it off), never compiled. Writes ``test_<checkpoint stem>.json`` next to the
checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from discopy.neural.cell import count
from sat import data
from sat.model import build
from sat.solve import SolveConfig, describe, evaluate

#: Formulas per search batch, by the difficulty being decoded.
CHUNKS = {"easy": 1000, "medium": 1000, "hard": 500}


def load_checkpoint(path, device):
    """ ``(net, stored)``: the model a checkpoint stores, at its shape. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    net = build(**stored["shape"]).to(device).eval()
    net.load_state_dict(stored["state_dict"])
    return net, stored


def score(net, pool, cfg: SolveConfig, chunk: int = 1000, log=None) -> dict:
    """ The repair search on a pool of formulas. """
    net.eval()
    return evaluate(net, pool, cfg, chunk, log)


def test_checkpoint(args, log) -> dict:
    """ ``args.checkpoint`` decoded on the test split of every
    ``--splits`` difficulty. """
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
        cfg = SolveConfig(chains=args.chains, budget=args.budget,
                          seed=args.seed)
        out[difficulty] = score(net, pool, cfg,
                                args.chunk or CHUNKS[difficulty], log)
        log(describe(difficulty, out[difficulty]))
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "data": str(data.root(args.data))},
            "shape": net.shape(), "family": family,
            "trained_on": stored.get("difficulty"), "step": stored.get("step"),
            "seed": stored.get("seed"), "test": out,
            "solve_rate": {k: v["rate"] for k, v in out.items()}}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt of sat.train_lattice")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $SAT_DATA or "
                        "data/g4satbench)")
    p.add_argument("--splits", nargs="+", default=None,
                   choices=("easy", "medium", "hard"),
                   help="difficulties to decode (default: the checkpoint's; "
                        "hard is the zero-shot transfer)")
    p.add_argument("--seed", type=int, default=123,
                   help="seed of the search (default: 123)")
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--budget", type=int, default=40,
                   help="model calls per formula")
    p.add_argument("--chunk", type=int, default=None,
                   help="formulas per search batch (default: 1000, 500 on "
                        "hard)")
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
