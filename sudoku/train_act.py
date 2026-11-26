"""
Sudoku-extreme WITHOUT the lattice, training: the same peer-graph GNN plus
a 21-parameter halt head (32,784 parameters) run as a segmented ACT
recursion: 12 rounds x 8 cycles per segment with the first 6 cycles
detached (a 24-round gradient window), up to 32 segments per puzzle in
training.

    python -m sudoku.train_act --seed 8 --out runs/act_s8
    python -m sudoku.eval_act runs/act_s8/best.pt

Training: 275,000 steps (one segment and one optimizer step each) on 512
slots of raw starts from the 1,000 base puzzles under digit relabeling,
AdamW (lr 7.5e-4, wd 0.04821, betas 0.9/0.95), 10 % warmup then cosine to
5 %, EMA 0.9995 (bias corrected); deep CE + per-cell halt BCE + board halt
BCE. Every board's gradient is clipped at 50 x a slow reference of the
batch median per-loss gradient norm, then the global norm at 0.5; a step
whose pre-clip norm exceeds 50 is rejected; slots start at staggered
depths. fp32 throughout; the trunk runs on the CUDA segment kernels where
available (``--kernel auto``; ``off`` is the PyTorch path with per-board
gradients from ``torch.func``). Every 1,000 steps the EMA weights are
validated on 500 puzzles at a fixed 32 and 128 segments; the best board
accuracy at 128 is kept. Writes ``OUT/best.pt`` (its weights, the model
shape, the step, its log record and the seed) and ``OUT/train.json``;
``sudoku.eval_act`` tests the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.act import ActConfig, rollout
from discopy.neural.cell import count
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything

from kernel import ActPool
from sudoku import data
from sudoku.eval_act import build, hours, lattices_of, ready


def quick_valid(net, valid, cap: int = 128, shallow: int = 32,
                batch: int = 500) -> dict:
    """ Board accuracy at two fixed depths off one trajectory, the cell
    accuracy and the ACT depth. """
    net.eval()
    x, target, given = valid
    got = {key: [] for key in ("shallow", "deep", "cell", "depth")}
    for s in range(0, len(x), batch):
        out = rollout(net, x[s:s + batch], target[s:s + batch],
                      given[s:s + batch], {shallow, cap})
        got["shallow"].append(out[shallow]["fixed"])
        got["deep"].append(out[cap]["fixed"])
        got["cell"].append(out[cap]["cell"])
        got["depth"].append(out[cap]["depth"])
    mean = {key: float(torch.cat(v).float().mean()) for key, v in got.items()}
    return {f"board_{shallow}": mean["shallow"], f"board_{cap}": mean["deep"],
            f"cell_{cap}": mean["cell"], "act_depth": mean["depth"]}


def checkpoint(net, step: int, record, seed: int) -> dict:
    return {"state_dict": net.state_dict(), "widths": tuple(net.widths),
            "shape": net.shape(), "step": step, "record": record, "seed": seed}


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    net = build(args.widths, args.rounds, args.cycles, args.detached,
                args.kernel).to(device)
    ready(net, args.kernel, not args.no_compile)
    cfg = ActConfig(steps=args.steps, batch_size=args.batch, n_sup=args.n_sup,
                    lr=args.lr, weight_decay=args.weight_decay,
                    warmup_frac=args.warmup, ema_decay=args.ema,
                    grad_clip=args.grad_clip, slot_clip=args.slot_clip,
                    slot_ref_decay=args.slot_ref_decay,
                    halt_board_weight=args.halt_board_weight,
                    skip_spike=args.skip_spike,
                    stagger_pool=not args.no_stagger)
    log(f"act: peer graph {tuple(net.widths)} + halt head = {count(net):,} "
        f"params  {net.rounds} rounds x {net.cycles} cycles, detached "
        f"{net.detached}  seed {args.seed}  kernel {args.kernel}  "
        f"device {device}\n{cfg}")
    common = data.load(args.data)
    stream = data.DeviceStream(data.Stream(
        common["base_puzzles"], common["base_solutions"],
        np.random.default_rng(args.seed), device=device, probs=(1.0, 0.0, 0.0)))
    valid = lattices_of(common["valid_puzzles"], common["valid_solutions"],
                        args.valid_n, device)

    optimizer = adamw(net, cfg.lr, cfg.weight_decay, tuple(cfg.betas))
    scheduler = cosine_schedule(optimizer, int(cfg.warmup_frac * cfg.steps),
                                cfg.steps, cfg.lr_floor)
    ema = EMA(net, cfg.ema_decay)
    pool = ActPool(net, stream, cfg)
    keys = ("loss", "ce", "q", "grad", "logit", "spike", "halted", "solved",
            "depth_sum", "capped")
    totals = dict.fromkeys(keys, 0.0)
    clip_med = clip_frac = 0.0
    reads, best, best_step, dead = {}, -1.0, None, 0
    start = tick = time.perf_counter()
    net.train()
    for index in range(1, cfg.steps + 1):
        stats = pool.iterate(optimizer, scheduler, ema)
        for key in keys:
            totals[key] = totals[key] + stats[key]
        clip = stats["clip"]
        if clip is not None:
            clip_med += clip["median"]
            clip_frac += clip["clipped_frac"]
        if index % args.log_every:
            continue
        value = {key: float(v) for key, v in totals.items()}
        halted = max(value["halted"], 1.0)
        seconds = time.perf_counter() - tick
        rate = args.log_every / max(seconds, 1e-9)
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
                  "clip_med": clip_med / args.log_every,
                  "clip_frac": clip_frac / args.log_every,
                  "clip_ref": net.slot_ref_running,
                  "lr": scheduler.get_last_lr()[0], "seconds": seconds}
        totals = dict.fromkeys(keys, 0.0)
        clip_med = clip_frac = 0.0
        log(f"step {index:6d}/{cfg.steps}  loss {record['loss']:.4f} "
            f"(ce {record['ce']:.4f} q {record['q']:.4f})  logit "
            f"{record['logit']:+.2f}  grad {record['grad']:.2f}  halted "
            f"{record['halted']:5d}  solved {record['solved']:.3f}  depth "
            f"{record['depth']:5.2f}  capped {record['capped']:.3f}"
            + (f"  SPIKES {record['spikes']}" if record["spikes"] else "")
            + f"  clip med {record['clip_med']:.3g} frac "
            f"{record['clip_frac']:.3f} ref {record['clip_ref'] or 0.0:.3g}"
            f"  lr {record['lr']:.2e}  {rate:.2f} steps/s  "
            f"ETA {hours((cfg.steps - index) / rate)}")
        dead = dead + 1 if record["spikes"] >= args.log_every else 0
        if index % args.eval_every == 0:
            with ema.averaged(net):
                read = quick_valid(net, valid, args.valid_cap,
                                   batch=args.eval_batch)
                net.train()
                record.update(read)
                reads[index] = read
                log("    valid  " + "  ".join(f"{k} {v:.4f}"
                                              for k, v in read.items()))
                if read[f"board_{args.valid_cap}"] > best:
                    best, best_step = read[f"board_{args.valid_cap}"], index
                    torch.save(checkpoint(net, index, dict(record), args.seed),
                               out / "best.pt")
                    log(f"    new best {best:.4f}, checkpointed")
        tick = time.perf_counter()
        if dead >= 3:
            log(f"DEAD RUN: every step rejected for {dead * args.log_every} "
                f"steps; aborting at step {index}")
            break
    train_seconds = time.perf_counter() - start
    if best_step is None:                     # never validated (short runs)
        best_step = index
        with ema.averaged(net):
            torch.save(checkpoint(net, index, None, args.seed), out / "best.pt")
    log(f"best checkpoint: step {best_step}")
    return {"params": count(net), "config": {**vars(args), "out": str(out)},
            "valid": reads, "best_step": best_step, "last_step": index,
            "train_seconds": train_seconds}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $SUDOKU_DATA or data/)")
    p.add_argument("--steps", type=int, default=275000)
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=(15, 39, 76, 20))
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--detached", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--n-sup", type=int, default=32,
                   help="segment cap of one puzzle in training")
    p.add_argument("--lr", type=float, default=7.5e-4)
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
    p.add_argument("--kernel", default="auto",
                   choices=("auto", "off", "fp32", "tf32", "fma",
                            "geometric"))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000,
                   help="validation cadence (a multiple of --log-every)")
    p.add_argument("--valid-n", type=int, default=500)
    p.add_argument("--valid-cap", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=500,
                   help="puzzles per validation rollout batch")
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
