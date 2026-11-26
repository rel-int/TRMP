"""
Maze-unique WITH the lattice, test: a checkpoint of ``maze.train_lattice``
scored by chain search on the 1,000 test mazes -- 64 chains per maze, a
budget of 1,000 model calls per maze, 1,024 rows per model call (16 mazes
at a time), accept threshold theta_cls 0.53, kill threshold theta from the
checkpoint (0.1), temperature 1.5, a valid minimal path preferred among the
chains that finish together, search seed 1.  On CUDA every model call runs
its 24 cycles on the Triton round (bf16 dots, bf16 storage) under bf16
autocast, the rest compiled (``--kernel off`` runs every round in PyTorch,
``--kernel geometric`` on the blocks of the grid, :mod:`maze.geometric`).
An accepted board is ``correct`` (the answer's path), ``alternative``
(another minimal path), ``nonminimal`` (a valid longer path) or
``invalid``; ``lenient`` = correct + alternative is the accuracy.

    python -m maze.eval_lattice runs/maze_lattice_s1/best.pt

``--split train --skip 1000 --n 100 --budget 400 --chains 16`` is the
validation protocol the trainer scores its milestones with (``--split
valid`` is the same 100 mazes).  Checkpoints that carry an extra
``recipe`` key load too.  Writes ``test.json`` next to the checkpoint, or
into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from discopy.neural.cell import count
from discopy.neural.lattice import TEMP, THETA

from maze import data, geometric
from maze.lattice import THETA_CLS_EVAL
from maze.model import build
from maze.solve import SolveConfig, describe, score

#: The keys of a checkpoint's ``shape`` that :func:`maze.model.build` takes.
SHAPE_KEYS = ("widths", "rounds", "cycles", "detached", "deep", "side")
#: Train-split rows before the 100 validation mazes.
VALID_SKIP = 1000


def ready(net, kernel: str, compiled: bool = True) -> None:
    """ ``kernel`` triton puts every forward-only cycle on the Triton round
    at bf16 dots and bf16 storage (it needs CUDA and triton: no fallback);
    ``off`` runs every round in PyTorch, ``geometric`` on the blocks of the
    grid (a :mod:`maze.geometric` net). """
    if kernel == "triton":
        from maze.kernel import HAVE_TRITON
        if not (torch.cuda.is_available() and HAVE_TRITON):
            raise SystemExit("--kernel triton needs a CUDA GPU and triton; "
                             "--kernel off runs the PyTorch rounds instead")
    elif kernel not in ("off", "geometric"):
        raise SystemExit(
            f"--kernel is triton, off or geometric, not {kernel!r}")
    net.fast_kernel = kernel == "triton"
    net.fast_kernel_prec, net.fast_kernel_store = "bf16", "bf16"
    net.fast_compile = "default" if compiled else "none"


def evaluator_for(shape: dict, device, kernel: str = "triton",
                  compiled: bool = True):
    """
    A chain-search net of ``shape``, compiled on CUDA.  It is built under a
    forked CPU generator, so building one in the middle of a training run
    draws nothing from that run's generator; its weights are loaded after.
    """
    extra = {k: v for k, v in shape.items() if k not in SHAPE_KEYS}
    if extra:
        raise SystemExit(f"unsupported checkpoint shape keys {sorted(extra)}")
    with torch.random.fork_rng(devices=[]):
        net = (geometric.Net if kernel == "geometric" else build)(**shape)
    net = net.to(device).eval()
    ready(net, kernel, compiled)
    if compiled and device.type == "cuda":
        net.compile()
    return net


def load_checkpoint(path, device, kernel: str = "triton",
                    compiled: bool = True) -> tuple:
    """ ``(evaluator, checkpoint)`` of a ``best.pt``, the weights loaded
    strictly. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    net = evaluator_for(dict(stored["shape"]), device, kernel, compiled)
    net.load_state_dict(stored["state_dict"])
    return net, stored


def mazes(split: str, n: int, skip: int, device, directory=None,
          k: int = 1) -> tuple:
    """ ``(x, solutions (n, k, S, 5), shortest)`` of ``n`` mazes of a split
    after ``skip``, on the device; ``valid`` is the train split after its
    first 1,000 rows. """
    if split == "valid":
        split, skip = "train", VALID_SKIP + skip
    return tuple(torch.as_tensor(column).to(device)
                 for column in data.held_out(split, n, skip, k, directory))


def solve_config(budget: int, chains: int, batch: int, theta: float,
                 theta_cls: float, temp: float, sync: int, seed: int,
                 device) -> SolveConfig:
    return SolveConfig(max_rounds=budget, n_chains=chains, batch_size=batch,
                       theta=theta, theta_cls=theta_cls, temp=temp,
                       sync_every=sync, seed=seed,
                       amp_dtype="bfloat16" if device.type == "cuda" else None)


def test_checkpoint(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, stored = load_checkpoint(args.checkpoint, device, args.kernel,
                                  not args.no_compile)
    config = stored.get("config") or stored.get("recipe") or {}
    theta = args.theta if args.theta is not None \
        else config.get("theta") or THETA
    log(f"{args.checkpoint}: {count(net):,} params, shape {net.shape()}, "
        f"step {stored.get('step')}, training seed {stored.get('seed')}; "
        f"{args.split} mazes {args.skip}..{args.skip + args.n - 1}, budget "
        f"{args.budget}, {args.chains} chains, {args.batch} rows, theta "
        f"{theta}, theta_cls {args.theta_cls}, seed {args.seed}, kernel "
        f"{args.kernel}")
    pool = mazes(args.split, args.n, args.skip, device, args.data)
    cfg = solve_config(args.budget, args.chains, args.batch, theta,
                       args.theta_cls, args.temp, args.sync, args.seed, device)
    test = score(net, pool, cfg)
    log(describe(args.split, test))
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "out": str(out), "theta": theta},
            "shape": net.shape(), "step": stored.get("step"),
            "seed": args.seed, "test": test,
            "accuracy": test["lenient"] / test["n"]}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt written by maze.train_lattice")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $MAZE_DATA or data/)")
    p.add_argument("--split", default="test", choices=("test", "valid", "train"))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--budget", type=int, default=1000,
                   help="model calls per maze")
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--batch", type=int, default=1024, help="rows per model call")
    p.add_argument("--seed", type=int, default=1,
                   help="seed of the search (1 for every seed's test)")
    p.add_argument("--theta", type=float, default=None,
                   help="kill threshold (default: the checkpoint's, 0.1)")
    p.add_argument("--theta-cls", type=float, default=THETA_CLS_EVAL)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--sync", type=int, default=1)
    p.add_argument("--kernel", default="triton",
                   choices=("triton", "off", "geometric"))
    p.add_argument("--no-compile", action="store_true")
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
