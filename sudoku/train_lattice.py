"""
Sudoku-extreme WITH the lattice, training: the recursive GNN on the peer
graph (the default: widths 15, 39, 76, 20 = 32,763 parameters, gradient
through the last cycle only, ``--detached 11``) or, with ``--graph fg``,
on the factor graph of the 27 units (widths 13, 34, 68, 17 = 32,798
parameters, gradient through the last two cycles, ``--detached 10``); 5
rounds x 12 cycles per supervision step, trained on a persistent pool of
lattices.

    python -m sudoku.train_lattice --seed 1 --out runs/lattice_s1
    python -m sudoku.train_lattice --graph fg --seed 1 --out runs/fg_s1
    python -m sudoku.eval_lattice runs/lattice_s1/best.pt

Training: 20,000 steps on a pool of 512, AdamW (weight decay on the
matrices, betas 0.9/0.95), linear warmup then cosine to 5 %, EMA 0.9995
(bias corrected), bf16 autocast, ``torch.compile``, the PyTorch path. The
graph picks the recipe the other flags default to: peer lr 4.8119e-3, wd
0.04821, warmup over 10.09 % of the steps, grad clip 0.5 (``--kernel
auto`` runs the CUDA segment kernels instead, tf32 under bf16); factor
graph lr 3.257e-3, wd 0.02495, warmup 9.85 %, grad clip 1.0 (always the
PyTorch path). At 90 % and 100 % of the steps the EMA weights are scored
by chain search on 500 validation puzzles (budget 300 model calls, sync 4,
seeded by ``--seed``); the better milestone (the later on a tie) is kept.
Writes ``OUT/best.pt`` (its weights, the model shape, the step, its
validation score and the seed) and ``OUT/train.json``;
``sudoku.eval_lattice`` tests the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.cell import count
from discopy.neural.lattice import Config, Pool, describe
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything

from sudoku import data
from sudoku.eval_lattice import build, evaluator_for, score

#: Each graph's recipe: what the flags left unset default to. The factor
#: graph's is the 32k configuration (5 x 12, detached 10).
RECIPES = {
    "peer": dict(widths=(15, 39, 76, 20), detached=11, lr=4.8119e-3,
                 weight_decay=0.04821, warmup=0.1009, grad_clip=0.5),
    "fg": dict(widths=(13, 34, 68, 17), detached=10, lr=3.2570e-3,
               weight_decay=0.02495, warmup=0.0985, grad_clip=1.0),
}
NAMES = {"peer": "peer graph", "fg": "factor graph"}


def hours(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    common = data.load(args.data)
    stream = data.Stream(common["base_puzzles"], common["base_solutions"],
                         np.random.default_rng(args.seed), device=device)
    net = build(args.widths, args.rounds, args.cycles, args.detached,
                args.graph, args.kernel).to(device)
    cfg = Config(steps=args.steps, batch_size=args.batch, lr=args.lr,
                 weight_decay=args.weight_decay, warmup_frac=args.warmup,
                 ema_decay=args.ema, grad_clip=args.grad_clip, seed=args.seed,
                 compile=not args.no_compile)
    milestones = sorted(set(args.milestones or
                            (round(0.9 * args.steps), args.steps)))
    log(f"lattice: {NAMES[args.graph]} {tuple(net.widths)} = {count(net):,} "
        f"params  {net.rounds} rounds x {net.cycles} cycles, detached "
        f"{net.detached}  seed {args.seed}  kernel {args.kernel}  "
        f"milestones {milestones}  device {device}\n{cfg}")

    optimizer = adamw(net, cfg.lr, cfg.weight_decay, tuple(cfg.betas))
    scheduler = cosine_schedule(optimizer, int(cfg.warmup_frac * cfg.steps),
                                cfg.steps)
    ema = EMA(net, cfg.ema_decay)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    net.kernel = args.kernel
    if cfg.compile and device.type == "cuda":
        net.compile(mode=cfg.compile_mode)          # the whole unroll
    pool = Pool(net, stream, cfg)                   # draws the initial pool
    evaluator = evaluator_for(net.shape(), device, args.kernel,
                              not args.no_compile)
    valid = dict(n=args.valid_n, budget=args.valid_budget, sync=args.valid_sync)

    totals = dict.fromkeys(Pool.COUNTERS, 0)
    scores, logged = {}, 0
    start = tick = time.perf_counter()
    net.train()
    for step in range(1, cfg.steps + 1):
        stats = pool.iterate(optimizer, scheduler, ema, generator)
        for key in Pool.COUNTERS:
            totals[key] += stats[key]
        if step % args.log_every == 0 or step == cfg.steps:
            rate = (step - logged) / max(time.perf_counter() - tick, 1e-9)
            losses = {key: float(value) for key, value in stats.items()
                      if torch.is_tensor(value)}
            log(f"step {step:6d}/{cfg.steps}  loss {losses['loss']:.4f} "
                f"(bce {losses['bce']:.4f} sm {losses['sm']:.4f} "
                f"conf {losses['conf']:.4f})  solved {totals['solved_correct']}"
                f"  tp_conf {totals['tp_conflict']}  aged {totals['aged']}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {rate:.2f} steps/s  "
                f"ETA {hours((cfg.steps - step) / rate)}")
            totals, logged = dict.fromkeys(Pool.COUNTERS, 0), step
            tick = time.perf_counter()
        if step in milestones:
            weights = ema.weights()
            torch.save({"state_dict": weights, "shape": net.shape(),
                        "step": step}, out / f"milestone-{step}.pt")
            evaluator.load_state_dict(weights)
            scores[step] = score(evaluator, common["valid_puzzles"],
                                 common["valid_solutions"], args.seed, **valid)
            log(f"milestone {step}: {describe('valid', scores[step])}")
            tick = time.perf_counter()
    train_seconds = time.perf_counter() - start

    best = max(scores, key=lambda s: (scores[s]["correct"], s))
    stored = torch.load(out / f"milestone-{best}.pt", map_location="cpu",
                        weights_only=False)
    log(f"best milestone: step {best} ({scores[best]['correct']}/"
        f"{args.valid_n} valid)")
    torch.save({**stored, "valid": scores[best], "seed": args.seed},
               out / "best.pt")
    for step in scores:                     # only the best is kept
        (out / f"milestone-{step}.pt").unlink(missing_ok=True)
    return {"params": count(net), "config": {**vars(args), "out": str(out)},
            "milestones": milestones, "valid": scores, "best_step": best,
            "train_seconds": train_seconds}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1,
                   help="training seed, and the seed of the validation search")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $SUDOKU_DATA or data/)")
    p.add_argument("--graph", choices=sorted(RECIPES), default="peer",
                   help="peer (every cell wired to its 20 peers) or fg (the "
                        "factor graph of the 27 units); picks the defaults of "
                        "--widths, --detached, --lr, --weight-decay, --warmup "
                        "and --grad-clip")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=None)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--detached", type=int, default=None)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--milestones", type=int, nargs="+", default=None,
                   help="validation steps (default: 90%% and 100%% of --steps)")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--valid-n", type=int, default=500)
    p.add_argument("--valid-budget", type=int, default=300)
    p.add_argument("--valid-sync", type=int, default=4)
    p.add_argument("--kernel", default="off",
                   choices=("off", "auto", "fp32", "tf32", "fma",
                            "geometric"))
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args(argv)
    for key, value in RECIPES[args.graph].items():      # the graph's recipe
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None) -> int:
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)

    result = train(args, out, log)
    (out / "train.json").write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {out / 'train.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
