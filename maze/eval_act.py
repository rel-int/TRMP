"""
Maze-unique WITHOUT the lattice, test: a checkpoint of ``maze.train_act``
on the 1,000 test mazes, deterministically -- one noise-free trajectory of
segments per batch of 250 mazes, read after exactly 1, 2, 4, ..., 256
segments: ``exact`` (the board is the answer's path), ``lenient`` (a valid
minimal path), the cell accuracy, and the halt-rule answer (the board of
the first segment whose board halt logit is positive, else the board at
the cap) with its mean depth.  fp32 with TF32 matmuls, the PyTorch round
with a compiled cycle (``--kernel geometric``: on the blocks of the grid,
:mod:`maze.geometric`).  The model shape comes from the checkpoint.

    python -m maze.eval_act runs/maze_act_s1/best.pt

Writes ``test.json`` next to the checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from discopy.neural.cell import Widths, count

from maze import data, geometric
from maze.act import deterministic
from maze.model import build_act

SHAPE_KEYS = ("rounds", "cycles", "detached", "deep", "side", "halt_detach")


def load_checkpoint(path, device, compiled: bool = True,
                    kernel: str = "off") -> tuple:
    """ ``(net, checkpoint)``: the checkpoint's :class:`~maze.model.ActNet`
    at the shape it stores, in eval mode, its cycle compiled on CUDA; with
    ``kernel`` geometric, a :class:`maze.geometric.ActNet`. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    widths = Widths(*stored["widths"])
    shape = {k: v for k, v in (stored.get("shape") or {}).items()
             if k in SHAPE_KEYS}
    make = geometric.ActNet if kernel == "geometric" else build_act
    net = make(widths, **shape).to(device)
    net.load_state_dict(stored["state_dict"])
    net.eval()
    if compiled and device.type == "cuda":
        net.compile(unroll=False, dynamic=False)
    return net, stored


def test_checkpoint(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    net, stored = load_checkpoint(args.checkpoint, device, not args.no_compile,
                                  args.kernel)
    pool = data.lattices_of("test", args.test_n, device, directory=args.data)
    log(f"{args.checkpoint}: {tuple(net.widths)} = {count(net):,} params, "
        f"{net.rounds}x{net.cycles} detached {net.detached}, step "
        f"{stored.get('step')}, seed {stored.get('seed')}, {len(pool[0])} "
        "test mazes")
    tick = time.perf_counter()
    caps = sorted(set(args.det_caps))
    fixed = deterministic(net, pool, caps, args.batch, log)
    top = fixed[str(caps[-1])]
    headline = {"exact": top["exact"], "lenient": top["lenient"],
                "halt_rule": top["act_exact"], "halt_depth": top["act_depth"],
                "cap": caps[-1], "n": len(pool[0])}
    log(f"TEST @{caps[-1]}: exact {top['exact']:.4f}  lenient "
        f"{top['lenient']:.4f}  halt rule {top['act_exact']:.4f} (depth "
        f"{top['act_depth']:.3f})")
    return {"checkpoint": str(args.checkpoint),
            "config": {**vars(args), "out": str(out)},
            "step": stored.get("step"), "seed": stored.get("seed"),
            "n": len(pool[0]), "params": count(net), "fixed": fixed,
            "headline": headline, "seconds": time.perf_counter() - tick}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt written by maze.train_act")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $MAZE_DATA or data/)")
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=250,
                   help="mazes per trajectory batch")
    p.add_argument("--det-caps", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--kernel", default="off", choices=("off", "geometric"),
                   help="geometric runs the rounds on the blocks of the grid")
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
