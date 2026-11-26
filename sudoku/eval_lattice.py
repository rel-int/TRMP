"""
Sudoku-extreme WITH the lattice, test: a checkpoint of
``sudoku.train_lattice`` scored by chain search on the first 1,000 test
puzzles (budget 1,000 model calls per puzzle, sync 1): 64 chains per
puzzle, theta_cls 0.8, 4,096 rows per model call, bf16 autocast and a
compiled cycle on CUDA, the PyTorch path (``--kernel auto`` runs the CUDA
segment kernels instead, tf32 under bf16; a factor-graph net always runs
the PyTorch path; ``--kernel geometric`` runs either net on the blocks of
the grid's symmetry group, :mod:`sudoku.geometric`). The model shape and
graph come from the checkpoint: a shape without a ``graph`` is a
peer-graph net, unless its weights hold a unit box (``unit.*``), which
marks a factor-graph net. The search is
seeded by ``--seed``, by default the training seed stored in the
checkpoint, so the defaults repeat a run's test.

    python -m sudoku.eval_lattice runs/lattice_s1/best.pt

Writes ``test.json`` next to the checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from discopy.neural.cell import as_widths, count
from discopy.neural.lattice import SolveConfig, describe, evaluate, lattice_of

from kernel import FactorNet, Net
from sudoku import data, geometric

#: The chain search's acceptance threshold and rows per model call.
THETA_CLS, BATCH = 0.8, 4096


def pool_units(embedded: torch.Tensor) -> torch.Tensor:
    """ The sum over every unit's nine members of a per-cell ``(B, 81, h)``
    tensor, as ``(B, 27, h)`` in :data:`~sudoku.data.UNITS` order: row,
    column and box reductions, no gather. """
    rows, _, h = embedded.shape
    grid = embedded.view(rows, data.N, data.N, h)
    return torch.cat([
        grid.sum(2), grid.sum(1),
        grid.view(rows, 3, 3, 3, 3, h).sum((2, 4)).reshape(rows, data.N, h)], 1)


def build(widths, rounds: int, cycles: int, detached: int,
          graph: str = "peer", kernel: str = "off") -> Net:
    """ A lattice net on the sudoku peer table (``graph="peer"``, a
    :class:`Net`) or on the factor graph of its 27 units (``"fg"``, a
    :class:`FactorNet`), on the blocks of the grid's symmetry group when
    ``kernel="geometric"`` (:mod:`sudoku.geometric`). """
    peer, factor = (geometric.Net, geometric.FactorNet) \
        if kernel == "geometric" else (Net, FactorNet)
    if graph == "fg":
        return factor(as_widths(widths), data.UNITS, 9, rounds, cycles,
                      detached, pool_units)
    if graph != "peer":
        raise ValueError(f"graph must be 'peer' or 'fg', not {graph!r}")
    return peer(as_widths(widths), data.PEERS, 9, rounds, cycles, detached)


def graph_of(stored: dict) -> str:
    """ A checkpoint's graph: its shape's ``graph``; without one, ``"fg"``
    if its weights hold a unit box, else ``"peer"``. """
    return stored["shape"].get("graph") or (
        "fg" if "unit.phi.0.weight" in stored["state_dict"] else "peer")


def evaluator_for(shape: dict, device, kernel: str = "off",
                  compiled: bool = True) -> Net:
    """ A separately compiled (one cycle) copy for the chain search. """
    net = build(**shape, kernel=kernel).to(device).eval()
    net.kernel = kernel
    if compiled and device.type == "cuda":
        net.compile(unroll=False, mode="reduce-overhead")
    return net


def load_checkpoint(path, device, kernel: str = "off",
                    compiled: bool = True) -> tuple:
    """ ``(evaluator, checkpoint)``: the chain-search copy of a checkpoint's
    model, built at the shape and on the graph the checkpoint stores. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    shape = {k: stored["shape"][k]
             for k in ("widths", "rounds", "cycles", "detached")}
    evaluator = evaluator_for({**shape, "graph": graph_of(stored)}, device,
                              kernel, compiled)
    evaluator.load_state_dict(stored["state_dict"])
    return evaluator, stored


def score(net, puzzles, solutions, seed: int, n: int, budget: int,
          sync: int) -> dict:
    """ The chain search on the first ``n`` puzzles of a split. """
    device = next(net.parameters()).device
    x = lattice_of(puzzles[:n]).to(device)
    y = lattice_of(solutions[:n]).to(device)
    cfg = SolveConfig(max_rounds=budget, sync_every=sync, theta_cls=THETA_CLS,
                      batch_size=BATCH, seed=seed,
                      amp_dtype="bfloat16" if device.type == "cuda" else None)
    return evaluate(net, x, y, cfg)


def test_checkpoint(args, out: Path, log) -> dict:
    """ ``args.checkpoint`` scored on the test protocol. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    evaluator, stored = load_checkpoint(args.checkpoint, device, args.kernel,
                                        not args.no_compile)
    seed = stored.get("seed", 1) if args.seed is None else args.seed
    common = data.load(args.data)
    log(f"{args.checkpoint}: {count(evaluator):,} params, shape "
        f"{evaluator.shape()}, step {stored.get('step')}, seed {seed}")
    test = score(evaluator, common["test_puzzles"], common["test_solutions"],
                 seed, args.test_n, args.test_budget, args.test_sync)
    log(describe("test", test))
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "out": str(out)},
            "shape": evaluator.shape(), "step": stored.get("step"),
            "seed": seed, "test": test, "accuracy": test["board"]}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt written by sudoku.train_lattice")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $SUDOKU_DATA or data/)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed of the chain search (default: the training "
                        "seed stored in the checkpoint, else 1)")
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--test-budget", type=int, default=1000)
    p.add_argument("--test-sync", type=int, default=1)
    p.add_argument("--kernel", default="off",
                   choices=("off", "auto", "fp32", "tf32", "fma",
                            "geometric"))
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
