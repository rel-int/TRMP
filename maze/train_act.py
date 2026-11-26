"""
Maze-unique WITHOUT the lattice, training: the recursive grid GNN plus a
22-parameter halt head (widths 16, 42, 84, 21 = 60,720 parameters, the
widths ``--params 64000`` selects) run as a segmented ACT recursion: 12
rounds x 12 cycles per segment with the first 10 cycles detached (a
24-round gradient window), up to 32 segments per maze in training.

    python -m maze.train_act --seed 1 --out runs/maze_act_s1
    python -m maze.eval_act runs/maze_act_s1/best.pt

Training: 120,000 steps (one segment and one optimizer step each) on 128
slots of raw starts from train mazes 0..999 under a random start/goal swap,
AdamW (lr 7.5e-4, wd 0.04821 on the matrices, betas 0.9/0.95), 10 % warmup
then cosine to 5 %, EMA 0.9995 (bias corrected); deep cross-entropy + per-
cell halt BCE + board halt BCE.  Every board's gradient is clipped at 50 x
a slow reference of the batch median per-loss gradient norm, then the
global norm at 0.5; a step whose pre-clip norm exceeds 50 is rejected;
slots start at staggered depths.  fp32 with TF32 matmuls; the forward-only
rounds (the prefix and validation) run on the Triton round with bf16 dots
and bf16 storage (``--kernel triton``; ``off`` runs them as compiled
PyTorch cycles, ``geometric`` runs the library's segment on the blocks of
the grid, :mod:`maze.geometric`).  From step 40,000, every 5,000 steps, the EMA weights are
validated on mazes 1000..1099 at 128 segments (board accuracy, exact); the
checkpoint is replaced only on a strict improvement.  Writes
``OUT/best.pt`` (its weights, the model shape, the step, its log record
and the seed) and ``OUT/train.json``; ``maze.eval_act`` tests it.

Seeding: the seed seeds torch, CUDA and numpy, then ``--params`` scans the
widths by building throw-away nets (which draws from the generator) before
the model is built; ``--widths`` skips the scan and so gives other initial
weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.cell import Widths, count
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything

from maze import data, geometric
from maze.act import ActConfig, ActPool, quick_valid
from maze.model import build_act, widths_for

KEYS = ("loss", "ce", "q", "grad", "logit", "spike", "escape", "poisoned",
        "halted", "solved", "depth_sum", "capped")


def ready(net, kernel: str, compiled: bool) -> None:
    """ The fused segment for training and validation; ``kernel`` triton
    puts its forward-only rounds on the Triton round (bf16 / bf16), and
    ``geometric`` takes the library's segment on a :mod:`maze.geometric`
    net instead. """
    if kernel == "triton":
        from maze.kernel import HAVE_TRITON
        if not (torch.cuda.is_available() and HAVE_TRITON):
            raise SystemExit("--kernel triton needs a CUDA GPU and triton; "
                             "--kernel off runs the PyTorch rounds instead")
    net.fast = kernel != "geometric"
    net.fast_compile = "default" if compiled else "none"
    net.fast_kernel = kernel == "triton"
    net.fast_kernel_prec, net.fast_kernel_store = "bf16", "bf16"


def checkpoint(net, widths, step: int, record, seed: int,
               score_key: str) -> dict:
    return {"state_dict": net.state_dict(), "widths": tuple(widths),
            "shape": net.shape(), "config": "full", "task": "maze",
            "step": step, "record": record, "seed": seed,
            "score_key": score_key}


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    cfg = ActConfig(steps=args.steps, batch_size=args.batch, n_sup=args.n_sup,
                    lr=args.lr, weight_decay=args.weight_decay,
                    warmup_frac=args.warmup, lr_floor=args.lr_floor,
                    ema_decay=args.ema, grad_clip=args.grad_clip,
                    slot_clip=args.slot_clip,
                    slot_ref_decay=args.slot_ref_decay,
                    halt_board_weight=args.halt_board_weight,
                    skip_spike=args.skip_spike,
                    stagger_pool=not args.no_stagger)
    widths = Widths(*args.widths) if args.widths else widths_for(args.params)
    make = geometric.ActNet if args.kernel == "geometric" else build_act
    net = make(widths, args.rounds, args.cycles, args.detached).to(device)
    ready(net, args.kernel, not args.no_compile)
    window = (net.cycles - net.detached) * net.rounds
    log(f"act: maze {tuple(widths)} = {count(net):,} params  {net.rounds} "
        f"rounds x {net.cycles} cycles, detached {net.detached} (gradient "
        f"window {window} rounds, {net.rounds * net.cycles} rounds a segment)  "
        f"seed {args.seed}  kernel {args.kernel}  device {device}\n{cfg}")
    x_all, y_all = data.load("train", args.data)
    stream = data.ActStream(x_all[:args.puzzles], y_all[:args.puzzles],
                            np.random.default_rng(args.seed), device)
    valid = data.lattices_of("train", args.valid_n, device,
                             skip=len(x_all) - args.valid_n, directory=args.data)
    log(f"train pool {args.puzzles} mazes (start/goal swap on), validation "
        f"{args.valid_n} mazes (rows {len(x_all) - args.valid_n}.."
        f"{len(x_all) - 1}), board@{args.valid_cap} picks the checkpoint")

    optimizer = adamw(net, cfg.lr, cfg.weight_decay, tuple(cfg.betas))
    scheduler = cosine_schedule(optimizer, int(cfg.warmup_frac * cfg.steps),
                                cfg.steps, cfg.lr_floor)
    ema = EMA(net, cfg.ema_decay)
    pool = ActPool(net, stream, cfg)
    eval_steps = set(args.eval_steps)
    score_key = f"board_{args.valid_cap}"
    totals = {key: torch.zeros((), device=device) for key in KEYS}
    clip_med = clip_frac = clip_max = clip_nmed = clip_ref = 0.0
    history, reads, best, best_step, dead = [], {}, -1.0, None, 0
    start = tick = time.perf_counter()
    net.train()
    index = 0
    for index in range(1, cfg.steps + 1):
        stats = pool.iterate(optimizer, scheduler, ema)
        _, c_med, c_max, c_frac, c_nmed, c_ref = stats["trace"]
        clip_med += c_med
        clip_frac += c_frac
        clip_max = max(clip_max, c_max) if c_max == c_max else c_max
        clip_nmed += c_nmed
        clip_ref = c_ref
        for key in KEYS:
            totals[key] += stats[key]
        if index % args.log_every:
            continue
        value = {key: float(totals[key]) for key in KEYS}
        halted = max(value["halted"], 1.0)
        record = {"step": index, "consumed": stream.consumed,
                  "loss": value["loss"] / args.log_every,
                  "ce": value["ce"] / args.log_every,
                  "q": value["q"] / args.log_every,
                  "grad": value["grad"] / args.log_every,
                  "logit": value["logit"] / args.log_every,
                  "halted": int(value["halted"]),
                  "solved": value["solved"] / halted,
                  "depth": value["depth_sum"] / halted,
                  "capped": value["capped"] / halted,
                  "spikes": int(value["spike"]),
                  "escapes": int(value["escape"]),
                  "poisoned": int(value["poisoned"]),
                  "lr": scheduler.get_last_lr()[0],
                  "seconds": time.perf_counter() - tick,
                  "clip_med": clip_med / args.log_every, "clip_max": clip_max,
                  "clip_frac": clip_frac / args.log_every,
                  "clip_nmed": clip_nmed / args.log_every, "clip_ref": clip_ref}
        clip_med = clip_frac = clip_max = clip_nmed = 0.0
        totals = {key: torch.zeros((), device=device) for key in KEYS}
        dead = dead + 1 if record["spikes"] >= args.log_every else 0
        log(f"step {index:6d}/{cfg.steps}  loss {record['loss']:.4f} "
            f"(ce {record['ce']:.4f} q {record['q']:.4f})  "
            f"logit {record['logit']:+.2f}  grad {record['grad']:.2f}  "
            f"halted {record['halted']:5d}  solved {record['solved']:.3f} "
            f" depth {record['depth']:5.2f}  capped {record['capped']:.3f}"
            + (f"  SPIKES {record['spikes']}" if record['spikes'] else "")
            + f"  CLIP med {record['clip_med']:.3g} "
            f"max {record['clip_max']:.3g} "
            f"frac {record['clip_frac']:.3f} "
            f"nmed {record['clip_nmed']:.3g} "
            f"ref {record['clip_ref']:.3g}"
            f"  lr {record['lr']:.2e}  ({record['seconds']:.1f}s"
            f", {stream.consumed:,} puzzles)")
        if index in eval_steps:
            with ema.averaged(net):
                net.eval()
                tick_valid = time.perf_counter()
                read = quick_valid(net, *valid, cap=args.valid_cap,
                                   batch_size=args.eval_batch)
                read["valid_seconds"] = time.perf_counter() - tick_valid
                read.pop("board_act", None)
                net.train()
            record.update(read)
            reads[index] = read
            log("    valid  " + "  ".join(
                f"{key} {v:.4f}" if isinstance(v, float) else f"{key} {v}"
                for key, v in read.items()))
            # a key the validation does not report
            # (a --valid-cap other than 32 / 128) scores 0.0
            score = read.get(score_key, 0.0)
            if score > best:
                best, best_step = score, index
                with ema.averaged(net):
                    torch.save(checkpoint(net, widths, index, dict(record),
                                          args.seed, score_key), out / "best.pt")
                log(f"    new best {score:.4f}, checkpointed")
        tick = time.perf_counter()
        history.append(record)
        if dead >= 3:
            log(f"DEAD RUN: every step rejected for {dead * args.log_every} "
                f"steps; aborting at step {index}")
            break
        if args.stop_at is not None and index >= args.stop_at:
            log(f"stopping at step {index} (--stop-at) on the {cfg.steps}-step "
                "schedule")
            break
    train_seconds = time.perf_counter() - start
    if best_step is None:                     # never validated (short runs)
        best_step = index
        with ema.averaged(net):
            torch.save(checkpoint(net, widths, index, None, args.seed,
                                  score_key), out / "best.pt")
        log(f"no validation ran: best.pt holds the EMA weights of step {index}")
    log(f"best checkpoint: step {best_step}")
    return {"params": count(net), "config": {**vars(args), "out": str(out)},
            "widths": list(widths), "valid": reads, "history": history,
            "best_step": best_step, "last_step": index,
            "train_seconds": train_seconds}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $MAZE_DATA or data/)")
    p.add_argument("--params", type=int, default=64000,
                   help="the widths closest to this parameter budget (a scan "
                        "that draws from the seeded generator)")
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=None, help="dim,state,hidden,y instead of --params")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--detached", type=int, default=10)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-sup", type=int, default=32,
                   help="segment cap of one maze in training")
    p.add_argument("--lr", type=float, default=7.5e-4)
    p.add_argument("--lr-floor", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.04821)
    p.add_argument("--warmup", type=float, default=0.10)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--slot-clip", type=float, default=50.0,
                   help="per-board clip, in multiples of the slow reference")
    p.add_argument("--slot-ref-decay", type=float, default=0.99)
    p.add_argument("--halt-board-weight", type=float, default=1.0)
    p.add_argument("--skip-spike", type=float, default=50.0,
                   help="reject a step whose pre-clip gradient norm exceeds this")
    p.add_argument("--no-stagger", action="store_true",
                   help="start every slot at depth 0")
    p.add_argument("--puzzles", type=int, default=1000,
                   help="train mazes in the pool (the first N rows)")
    p.add_argument("--valid-n", type=int, default=100,
                   help="validation mazes (the last N rows of the train split)")
    p.add_argument("--valid-cap", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=100,
                   help="mazes per validation rollout batch")
    p.add_argument("--eval-steps", type=int, nargs="+",
                   default=list(range(40000, 120001, 5000)),
                   help="validation steps (multiples of --log-every)")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--stop-at", type=int, default=None,
                   help="end the run at this step (checked at log steps) "
                        "on the --steps schedule")
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
