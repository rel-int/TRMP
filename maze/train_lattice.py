"""
Maze-unique WITH the lattice, training: the recursive grid GNN at widths
12, 31, 64, 16 (34,786 parameters), 8 rounds x 24 cycles per supervision
step with the first 20 cycles detached (a 32-round gradient window), trained
on a persistent pool of 96 lattices.

    python -m maze.train_lattice --seed 1 --out runs/maze_lattice_s1
    python -m maze.eval_lattice runs/maze_lattice_s1/best.pt

Training: 15,000 steps of AdamW (lr 3.4e-4, wd 0.18 on the matrices, betas
0.9/0.95), 10 % warmup then cosine to 5 %, gradient norm clip 1.0, EMA 0.999
(bias corrected); bf16 autocast, the forward-only cycles on the Triton round
(bf16 dots, bf16 storage; ``--kernel off`` runs them in PyTorch), the window
and the deep loss compiled.  The loss is deep-supervised over the four
window readouts against the union of the solutions the lattice still allows
(per-candidate BCE, symbol CE, board and per-cell conflict BCE).  Each step
projects the pool at kill threshold theta 0.1 and accept threshold 0.5 (one
sampled decision per board, temperature 1.5); a board leaves when it solves
to a valid minimal path, when its conflict is real or at age 100, and is
refilled from train mazes 0..999 under a random start/goal swap, as a raw,
partly solved or partly corrupted start.  Every 250 steps a log line and a
probe of 32 held-out mazes (fill, sound, killed, fires).

Selection: at every milestone (steps 8,000..15,000, every 250) the EMA
weights are scored by chain search on the 100 validation mazes (train rows
1000..1099: budget 400, 16 chains, theta_cls 0.53, search seed 1); the
checkpoint is replaced only on a strictly higher lenient count, and the
first milestone at 100/100 ends the run (the schedule horizon stays
15,000).  Writes ``OUT/best.pt`` and ``OUT/train.json``.

Seeding: the seed fixes the refill stream (numpy), then torch, CUDA and
numpy are seeded and the model is built on the CPU.  ``--params N`` picks
the widths closest to N parameters instead, by a scan that builds
throw-away nets after the seeding (so it changes the initial weights).
The matmul precision is left at its default ("highest").
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.cell import Widths, count
from discopy.neural.lattice import TEMP, THETA, THETA_CLS
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything

from maze import data, geometric
from maze.eval_lattice import evaluator_for, mazes, ready, solve_config
from maze.lattice import THETA_CLS_EVAL, Config, Pool, probe
from maze.model import build, widths_for
from maze.solve import describe, score

#: The recipe's widths (``--widths``, no scan).
WIDTHS = Widths(12, 31, 64, 16)
#: The recipe's milestones: every log step from 8,000 to 15,000.
MILESTONES = tuple(range(8000, 15001, 250))


def checkpoint(net, step: int, valid, seed: int, config: dict) -> dict:
    return {"state_dict": net.state_dict(), "shape": net.shape(),
            "step": step, "valid": valid, "seed": seed, "config": config}


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {**vars(args), "out": str(out)}
    milestones = sorted(set(args.milestones))
    stray = [s for s in milestones if s % args.log_every or s > args.steps]
    if stray:
        raise SystemExit(f"milestones {stray} are not multiples of --log-every "
                         f"({args.log_every}) at or below --steps ({args.steps})")

    tick = time.perf_counter()
    x, solutions, shortest = data.k_pool("train", args.puzzles, args.k,
                                         directory=args.data, log=log)
    log(f"train pool {x.shape} x {args.k} solutions "
        f"({time.perf_counter() - tick:.0f} s)")
    stream = data.Stream(x, solutions, shortest,
                         np.random.default_rng(args.seed), device=device,
                         hints=not args.no_hints)

    seed_everything(args.seed)
    widths = Widths(*args.widths) if args.widths \
        else widths_for(args.params) if args.params else WIDTHS
    make = geometric.Net if args.kernel == "geometric" else build
    net = make(widths, rounds=args.rounds, cycles=args.cycles,
               detached=args.detached).to(device)
    ready(net, args.kernel, not args.no_compile)
    log(f"model {tuple(widths)} = {count(net):,} parameters, "
        f"{args.rounds} rounds x {args.cycles} cycles "
        f"= {args.rounds * args.cycles} rounds a step"
        + ("; fast kernel bf16 dots, bf16 store"
           if args.kernel == "triton" else ""))

    cfg = Config(steps=args.steps, batch_size=args.batch, lr=args.lr,
                 weight_decay=args.weight_decay, warmup_frac=args.warmup,
                 max_age=args.max_age, ema_decay=args.ema,
                 grad_clip=args.grad_clip, theta=args.theta,
                 theta_cls=args.theta_cls, temp=args.temp,
                 cell_conf=not args.no_cell_conf, seed=args.seed,
                 compile=not args.no_compile)
    # the probe's mazes (with their K solutions) and the validation mazes:
    # the train rows after the pool, never drawn into it
    raw = mazes("train", args.probe, args.puzzles, device, args.data, args.k) \
        if args.probe else None
    valid = mazes("train", args.valid_n, args.puzzles, device, args.data) \
        if args.valid_n else None
    valid_cfg = solve_config(args.valid_budget, args.valid_chains,
                             args.valid_batch, args.theta, args.valid_theta_cls,
                             args.temp, 1, args.valid_seed, device)
    log(f"lattice: {cfg}")
    log(f"milestones {milestones[0] if milestones else '-'}.."
        f"{milestones[-1] if milestones else '-'} ({len(milestones)}), "
        + (f"scored on train rows {args.puzzles}..{args.puzzles + args.valid_n - 1}"
           f" (budget {args.valid_budget}, {args.valid_chains} chains, seed "
           f"{args.valid_seed}); the first {args.valid_n}/{args.valid_n} stops"
           if valid is not None else "not scored (--valid-n 0)"))
    if args.keep_milestones:
        (out / "milestones").mkdir(parents=True, exist_ok=True)

    # the training loop with the maze pool, the probe and the milestones
    probe_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    optimizer = adamw(net, cfg.lr, cfg.weight_decay, tuple(cfg.betas))
    scheduler = cosine_schedule(optimizer, int(cfg.warmup_frac * cfg.steps),
                                cfg.steps)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay else None
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    if cfg.compile and device.type == "cuda":
        net.compile()
    pool = Pool(net, stream, cfg)
    counters = pool.COUNTERS
    evaluator, scores = None, {}
    best, best_step, early = -1, None, False
    totals, history = dict.fromkeys(counters, 0), []
    start = tick = time.perf_counter()
    net.train()
    index = 0
    for index in range(1, cfg.steps + 1):
        stats = pool.iterate(optimizer, scheduler, ema, generator)
        for key in counters:
            totals[key] += stats[key]
        if index % args.log_every == 0:
            record = {"step": index, **totals,
                      **{key: stats[key] for key in pool.GAUGES},
                      "lr": scheduler.get_last_lr()[0],
                      **{key: float(value) for key, value in stats.items()
                         if torch.is_tensor(value)},
                      "seconds": time.perf_counter() - tick}
            totals, tick = dict.fromkeys(counters, 0), time.perf_counter()
            history.append(record)
            log(f"step {index:6d}/{cfg.steps}  loss {record['loss']:.4f}  "
                + "  ".join(f"{key} {record[key]:5d}" for key in counters)
                + "".join(f"  {key} {record[key]:.3f}" for key in pool.GAUGES)
                + f"  ({record['seconds']:.1f}s)")
            with ema.averaged(net) if ema is not None \
                    else contextlib.nullcontext():
                if raw is not None:
                    net.eval()
                    read = probe(net, raw, cfg.theta, cfg.theta_cls,
                                 probe_generator)
                    net.train()
                    record.update(read)
                    log("    " + "  ".join(
                        f"{key} {value:.3f}" for key, value in read.items()
                        if isinstance(value, float)))
                if index in milestones and args.keep_milestones:
                    torch.save(checkpoint(net, index, None, args.seed, config),
                               out / "milestones" / f"step{index}.pt")
                if index in milestones and valid is not None:
                    if evaluator is None:
                        evaluator = evaluator_for(net.shape(), device,
                                                  args.kernel,
                                                  not args.no_compile)
                    evaluator.load_state_dict(net.state_dict())
                    read = score(evaluator, valid, valid_cfg)
                    scores[index] = read
                    record["valid"] = read
                    log("    " + describe(f"valid step {index}", read))
                    if read["lenient"] > best:      # ties keep the earlier
                        best, best_step = read["lenient"], index
                        torch.save(checkpoint(net, index, read, args.seed,
                                              config), out / "best.pt")
                        log(f"    new best: step {index}, {best}/{read['n']}"
                            " lenient, checkpointed")
                    early = read["lenient"] >= read["n"]
            if early:
                log(f"step {index} scored {best}/{valid[0].shape[0]}: early "
                    f"stop (the {cfg.steps}-step schedule is unchanged)")
                break
        if args.stop_after is not None and index >= args.stop_after:
            log(f"stopping after step {index} (--stop-after) on the "
                f"{cfg.steps}-step schedule")
            break
    train_seconds = time.perf_counter() - start
    if best_step is None:                   # nothing scored (short runs)
        best_step = index
        with ema.averaged(net) if ema is not None else contextlib.nullcontext():
            torch.save(checkpoint(net, index, None, args.seed, config),
                       out / "best.pt")
        log(f"no milestone was scored: best.pt holds the EMA weights of "
            f"step {index}")
    log(f"best checkpoint: step {best_step}"
        + (f", {best}/{args.valid_n} lenient on validation" if scores else ""))
    return {"params": count(net), "config": config, "widths": list(widths),
            "valid": scores, "history": history, "best_step": best_step,
            "best_valid": best if scores else None, "early_stop": early,
            "last_step": index, "train_seconds": train_seconds}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ints = lambda s: tuple(int(v) for v in s.split(","))  # noqa: E731
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $MAZE_DATA or data/)")
    p.add_argument("--widths", type=ints, default=None,
                   help="dim,state,hidden,y (default 12,31,64,16)")
    p.add_argument("--params", type=int, default=None,
                   help="instead of the widths: the widths closest to this "
                        "parameter budget (a scan that draws from the seeded "
                        "generator)")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--cycles", type=int, default=24)
    p.add_argument("--detached", type=int, default=20)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=96, help="pool size")
    p.add_argument("--lr", type=float, default=3.4e-4)
    p.add_argument("--weight-decay", type=float, default=0.18)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-age", type=int, default=100)
    p.add_argument("--theta", type=float, default=THETA)
    p.add_argument("--theta-cls", type=float, default=THETA_CLS)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--k", type=int, default=8,
                   help="sampled minimal solutions per maze")
    p.add_argument("--puzzles", type=int, default=1000,
                   help="train mazes in the pool (the first N rows)")
    p.add_argument("--no-hints", action="store_true",
                   help="refill with raw starts only")
    p.add_argument("--no-cell-conf", action="store_true")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--probe", type=int, default=32,
                   help="held-out mazes read by the probe at every log line")
    p.add_argument("--milestones", type=ints, default=MILESTONES,
                   help="the steps whose EMA weights are scored (multiples "
                        "of --log-every)")
    p.add_argument("--valid-n", type=int, default=100,
                   help="validation mazes (the train rows after the pool); "
                        "0 scores nothing")
    p.add_argument("--valid-budget", type=int, default=400)
    p.add_argument("--valid-chains", type=int, default=16)
    p.add_argument("--valid-batch", type=int, default=1024)
    p.add_argument("--valid-seed", type=int, default=1)
    p.add_argument("--valid-theta-cls", type=float, default=THETA_CLS_EVAL)
    p.add_argument("--keep-milestones", action="store_true",
                   help="also write every milestone's EMA weights to "
                        "OUT/milestones/step<N>.pt")
    p.add_argument("--stop-after", type=int, default=None,
                   help="end the run after this step on the --steps schedule")
    p.add_argument("--kernel", default="triton",
                   choices=("triton", "off", "geometric"))
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args(argv)


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
