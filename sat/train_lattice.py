"""
SAT WITH the lattice, training: the factor-graph GNN of a CNF formula
(widths 48, 64, 64, 16 = 64,689 parameters, 6 rounds x 12 cycles per
supervision step with the gradient through the last two cycles,
``--detached 10``), trained on the unsupervised clause loss conditioned on
the lattice.

    python -m sat.train_lattice --family sr --seed 0 --out runs/sat_sr_easy
    python -m sat.eval_lattice runs/sat_sr_easy/best.pt

    python -m sat.train_lattice --family sr --difficulty medium --seed 0 \
        --out runs/sat_sr_medium
    python -m sat.eval_lattice runs/sat_sr_medium/best.pt --splits medium hard

Training: 20,000 steps on batches of 128 formulas drawn from the split's
80,000, AdamW (lr 7.5596e-4, wd 7.0407e-3, betas 0.9/0.999, decay on the
matrices), 400 steps of warmup then cosine to zero, gradient norm clipped
at 1.0, EMA 0.99 (bias corrected), fp32 throughout (no TF32, unlike the other
tasks: the reported runs had it off). A step starts blank
with probability 0.2 and otherwise from random commitments that leave
every clause satisfiable (``sat.lattice.start_lattice``), and its loss is
the clause objective of the residual formula plus the cross-entropy tying
the readout to the commitments. Every 500 steps the EMA weights are
scored by the repair search on a fixed 1,000 formulas of the validation
split (8 chains, 40 model calls, its own seed), and the best score is
kept.

Writes ``OUT/best.pt`` (its weights, the model shape, the family and
difficulty, the step, its validation score and the seed) and
``OUT/train.json``; ``sat.eval_lattice`` tests the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from discopy.neural.cell import count
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything
from sat import data, lattice
from sat.eval_lattice import score
from sat.model import build
from sat.solve import SolveConfig, describe, hours

#: The widths of both recipes: dim, state_dim, hidden, y_dim.
WIDTHS = (48, 64, 64, 16)


def checkpoint(net, args, step: int, record, seed: int) -> dict:
    return {"state_dict": net.state_dict(), "shape": net.shape(),
            "family": data.family_of(args.family),
            "difficulty": args.difficulty, "step": step, "record": record,
            "seed": seed}


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    net = build(args.widths, args.rounds, args.cycles, args.detached).to(device)
    family = data.family_of(args.family)
    log(f"lattice: {family} {args.difficulty}, factor graph {tuple(WIDTHS)} = "
        f"{count(net):,} params, {net.rounds} rounds x {net.cycles} cycles, "
        f"detached {net.detached}  seed {args.seed}  device {device}")

    train_pool = data.load(family, args.difficulty, "train", args.data,
                           args.train_n, args.seed, log)
    valid_pool = data.load(family, args.difficulty, "valid", args.data,
                           args.valid_n, args.seed, log)
    log(f"    {len(train_pool)} train, {len(valid_pool)} valid formulas")
    cfg = SolveConfig(chains=args.chains, budget=args.budget,
                      seed=args.seed + 777)

    optimizer = adamw(net, args.lr, args.weight_decay, betas=(0.9, 0.999))
    scheduler = cosine_schedule(optimizer, args.warmup, args.steps, floor=0.0)
    ema = EMA(net, args.ema)
    reads, best, best_step, record = {}, -1.0, None, None
    total, start, tick = 0.0, time.perf_counter(), time.perf_counter()
    net.train()
    for step in range(1, args.steps + 1):
        batch = data.to_device(
            data.sample(train_pool, args.batch, generator), device)
        x = (lattice.start_lattice(batch, args.start_probs, generator, device)
             if args.structural_start
             else lattice.initial_lattice(batch.n_vars, device))
        optimizer.zero_grad(set_to_none=True)
        _, _, every = net.run(batch, x)
        loss = lattice.deep_loss(every, batch, x if args.residual_loss else None,
                                 args.commit_weight)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                    args.grad_clip))
        optimizer.step()
        scheduler.step()
        ema.update(net)
        total += float(loss.detach())

        if step % args.log_every == 0:
            seconds = time.perf_counter() - tick
            rate = args.log_every / max(seconds, 1e-9)
            record = {"step": step, "loss": total / args.log_every,
                      "grad": grad, "lr": scheduler.get_last_lr()[0],
                      "seconds": seconds}
            total, tick = 0.0, time.perf_counter()
            log(f"step {step:6d}/{args.steps}  loss {record['loss']:.4f}  "
                f"grad {record['grad']:.2f}  lr {record['lr']:.2e}  "
                f"{rate:.2f} steps/s  "
                f"ETA {hours((args.steps - step) / rate)}")
        if step % args.valid_every == 0 or step == args.steps:
            with ema.averaged(net):
                read = score(net, valid_pool, cfg, args.chunk)
                net.train()
            reads[step] = read
            log("    " + describe("valid", read))
            if read["rate"] > best:
                best, best_step = read["rate"], step
                with ema.averaged(net):
                    torch.save(checkpoint(net, args, step, dict(read),
                                          args.seed), out / "best.pt")
                log(f"    new best {best:.4f}, checkpointed")
            tick = time.perf_counter()
    log(f"best checkpoint: step {best_step} ({best:.4f})")
    return {"params": count(net), "config": {**vars(args), "out": str(out)},
            "valid": reads, "best_step": best_step, "last_step": args.steps,
            "last_record": record, "train_seconds": time.perf_counter() - start}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $SAT_DATA or "
                        "data/g4satbench)")
    p.add_argument("--family", default="sr", choices=("sr", "3sat"),
                   help="the random family to train on (default: sr)")
    p.add_argument("--difficulty", default="easy", choices=("easy", "medium"),
                   help="variables per formula: 10-40 easy, 40-200 medium "
                        "(default: easy)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=WIDTHS)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--detached", type=int, default=10,
                   help="cycles run without a gradient, of --cycles")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=7.55960243384565e-4)
    p.add_argument("--weight-decay", type=float, default=7.040745655526356e-3)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--start-probs", type=float, nargs=3,
                   default=list(lattice.START_PROBS),
                   help="blank / viable / dead start probabilities")
    p.add_argument("--blank-start", dest="structural_start",
                   action="store_false",
                   help="start every step from the blank lattice")
    p.add_argument("--plain-loss", dest="residual_loss", action="store_false",
                   help="do not condition the loss on the commitments")
    p.add_argument("--commit-weight", type=float, default=1.0,
                   help="weight of the readout-to-commitment cross-entropy")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--valid-every", type=int, default=500)
    p.add_argument("--train-n", type=int, default=80000,
                   help="formulas of the train split to hold in the pool")
    p.add_argument("--valid-n", type=int, default=1000,
                   help="formulas the validation search decodes")
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--budget", type=int, default=40,
                   help="model calls per formula in the validation search")
    p.add_argument("--chunk", type=int, default=1000,
                   help="formulas per search batch")
    p.add_argument("--no-compile", action="store_true",
                   help="accepted for a uniform CLI; sat is never compiled")
    args = p.parse_args(argv)
    if abs(sum(args.start_probs) - 1.0) > 1e-6:
        raise SystemExit(f"--start-probs must sum to 1, got {args.start_probs}")
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
