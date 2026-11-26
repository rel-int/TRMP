"""
SAT WITHOUT the lattice, training: the same factor-graph GNN plus a halt
head that reads clauses (64,722 parameters), run as a segmented ACT
recursion -- 3 segments of 6 rounds x 12 cycles per optimizer step, with 6
of the 12 cycles detached and the differentiated window at a random
position inside the segment.

    python -m sat.train_act --family sr --seed 0 --out runs/act_sr_easy
    python -m sat.eval_act runs/act_sr_easy/best.pt

    python -m sat.train_act --family sr --difficulty medium --seed 0 \
        --out runs/act_sr_medium
    python -m sat.eval_act runs/act_sr_medium/best.pt --splits medium hard

Training: 10,000 steps on the easy splits and 20,000 on medium, batches of
128 formulas, AdamW (lr 7.5596e-4, wd 7.0407e-3, betas 0.9/0.999, decay on
the matrices), 400 steps of warmup then cosine to zero, gradient norm
clipped at 1.0, EMA 0.99 (bias corrected), fp32 throughout (no TF32, unlike the other
tasks: the reported runs had it off). The carried
state is detached between segments, so the backward depth stays one
window however many segments a step runs, and each segment's loss is the
deep clause objective plus the halt head's per-clause and per-instance
BCE against exact satisfaction; with ``--halt-detach`` (the default) the
head's gradient never reaches the trunk. The backward runs per segment
rather than on the sum, which is the same gradient at the peak memory of
one segment.

Every 500 steps the EMA weights are decoded on a fixed 1,000 formulas of
the validation split under the halt rule (cap 16, threshold 0) and the
best solve rate is kept; every solve there is an exact clause check, so it
is a solve rate and not a proxy.

Writes ``OUT/best.pt`` (its weights, the model shape, the family and
difficulty, the step, its validation record and the seed) and
``OUT/train.json``; ``sat.eval_act`` tests the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from discopy.neural.cell import count
from discopy.neural.train import EMA, adamw, cosine_schedule, seed_everything
from sat import act, data
from sat.solve import hours

#: Steps of the reported recipe, by difficulty.
STEPS = {"easy": 10000, "medium": 20000}
#: The widths of both recipes: dim, state_dim, hidden, y_dim.
WIDTHS = (48, 64, 64, 16)


def train_step(net, batch, x, segments: int, halt_weight: float,
               instance_weight: float) -> dict:
    """
    One optimizer-free step: ``segments`` segments from a blank state,
    each contributing the deep clause loss plus ``halt_weight`` times the
    halt loss, each backward run as its segment ends. The state is
    detached at every boundary, so the segments' graphs are independent
    and the sum of their backwards is the backward of their sum.
    """
    s = net.initial_act(batch, x)
    loss = unsat = halt_total = 0.0
    for _ in range(segments):
        s, every, halt, _ = net.segment(s, batch, x)
        sat = act.clause_sat(every[-1].detach(), batch)
        clause_loss = act.deep_loss(every, batch)
        halt_loss = net.halt.loss(halt, sat, batch, instance_weight)
        step_loss = (clause_loss + halt_weight * halt_loss) / segments
        step_loss.backward()
        loss += float(step_loss.detach())
        unsat += float(clause_loss.detach()) / segments
        halt_total += float(halt_loss.detach()) / segments
        s = act.detach_state(s)
    return {"loss": loss, "unsat": unsat, "halt": halt_total}


def checkpoint(net, args, step: int, record, seed: int) -> dict:
    return {"state_dict": net.state_dict(), "shape": net.shape(),
            "family": data.family_of(args.family),
            "difficulty": args.difficulty, "segments": args.segments,
            "step": step, "record": record, "seed": seed}


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    net = act.build(args.widths, args.rounds, args.cycles, args.detached,
                    halt_detach=args.halt_detach).to(device)
    net.random_window = args.random_window
    family = data.family_of(args.family)
    log(f"act: {family} {args.difficulty}, factor graph {tuple(WIDTHS)} + "
        f"halt head = {count(net):,} params, {net.rounds} rounds x "
        f"{net.cycles} cycles, detached {net.detached}, {args.segments} "
        f"segments  seed {args.seed}  device {device}")

    train_pool = data.load(family, args.difficulty, "train", args.data,
                           args.train_n, args.seed, log)
    valid_pool = data.load(family, args.difficulty, "valid", args.data,
                           args.valid_n, args.seed, log)
    log(f"    {len(train_pool)} train, {len(valid_pool)} valid formulas")
    valid = data.to_device(data.collate(valid_pool), device)

    optimizer = adamw(net, args.lr, args.weight_decay, betas=(0.9, 0.999))
    scheduler = cosine_schedule(optimizer, args.warmup, args.steps, floor=0.0)
    ema = EMA(net, args.ema)
    totals = {"loss": 0.0, "unsat": 0.0, "halt": 0.0}
    reads, best, best_step, record = {}, -1.0, None, None
    start, tick = time.perf_counter(), time.perf_counter()
    net.train()
    for step in range(1, args.steps + 1):
        batch = data.to_device(
            data.sample(train_pool, args.batch, generator), device)
        x = act.blank_clue(batch.n_vars, device)
        optimizer.zero_grad(set_to_none=True)
        stats = train_step(net, batch, x, args.segments, args.halt_weight,
                           args.instance_weight)
        grad = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                    args.grad_clip))
        optimizer.step()
        scheduler.step()
        ema.update(net)
        for key in totals:
            totals[key] += stats[key]

        if step % args.log_every == 0:
            seconds = time.perf_counter() - tick
            rate = args.log_every / max(seconds, 1e-9)
            record = {"step": step, "grad": grad,
                      "lr": scheduler.get_last_lr()[0], "seconds": seconds,
                      **{k: v / args.log_every for k, v in totals.items()}}
            totals = dict.fromkeys(totals, 0.0)
            tick = time.perf_counter()
            log(f"step {step:6d}/{args.steps}  loss {record['loss']:.4f} "
                f"(unsat {record['unsat']:.4f} halt {record['halt']:.4f})  "
                f"grad {record['grad']:.2f}  lr {record['lr']:.2e}  "
                f"{rate:.2f} steps/s  "
                f"ETA {hours((args.steps - step) / rate)}")
        if step % args.valid_every == 0 or step == args.steps:
            with ema.averaged(net):
                read = act.evaluate_act(net, valid, args.valid_cap,
                                        args.threshold)
                read["fixed"] = act.evaluate_fixed(net, valid,
                                                   args.valid_cap)["solved"]
                net.train()
            reads[step] = read
            log(f"    valid  solved {read['solved']:.4f}  frac "
                f"{read['frac']:.4f}  depth {read['depth']:.1f}  "
                f"fixed@{args.valid_cap} {read['fixed']:.4f}")
            if read["solved"] > best:
                best, best_step = read["solved"], step
                with ema.averaged(net):
                    torch.save(checkpoint(net, args, step, dict(read),
                                          args.seed), out / "best.pt")
                log(f"    new best {best:.4f}, checkpointed")
            tick = time.perf_counter()

    with ema.averaged(net):
        picked = act.evaluate_best_of_k(net, valid, args.valid_cap, args.k,
                                        args.sigma, args.threshold, args.seed)
    log(f"last weights on the validation formulas: single "
        f"{picked['single']:.4f}  best_of_{args.k} {picked['best_of_k']:.4f}  "
        f"pass_at_{args.k} {picked['pass_at_k']:.4f}")
    log(f"best checkpoint: step {best_step} ({best:.4f})")
    return {"params": count(net), "config": {**vars(args), "out": str(out)},
            "valid": reads, "best_step": best_step, "last_step": args.steps,
            "last_record": record, f"last_best_of_{args.k}": picked,
            "train_seconds": time.perf_counter() - start}


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
    p.add_argument("--difficulty", default="easy", choices=sorted(STEPS),
                   help="variables per formula: 10-40 easy, 40-200 medium "
                        "(default: easy)")
    p.add_argument("--steps", type=int, default=None,
                   help="default: 10,000 on easy, 20,000 on medium")
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=WIDTHS)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--detached", type=int, default=6,
                   help="cycles run without a gradient, of --cycles")
    p.add_argument("--segments", type=int, default=3,
                   help="supervision segments per optimizer step")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=7.55960243384565e-4)
    p.add_argument("--weight-decay", type=float, default=7.040745655526356e-3)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--halt-weight", type=float, default=1.0)
    p.add_argument("--instance-weight", type=float, default=1.0,
                   help="weight of the per-instance halt BCE")
    p.add_argument("--no-halt-detach", dest="halt_detach",
                   action="store_false",
                   help="let the halt loss reach the trunk")
    p.add_argument("--fixed-window", dest="random_window",
                   action="store_false",
                   help="keep the differentiated window at the end of a "
                        "segment instead of a random position")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="the instance halt logit the halt rule must clear")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--valid-every", type=int, default=500)
    p.add_argument("--train-n", type=int, default=80000,
                   help="formulas of the train split to hold in the pool")
    p.add_argument("--valid-n", type=int, default=1000,
                   help="formulas the validation rollout decodes")
    p.add_argument("--valid-cap", type=int, default=16,
                   help="segment cap of the validation rollout")
    p.add_argument("--k", type=int, default=8,
                   help="rollouts of the best-of-k line logged at the end")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="answer-trace noise of those rollouts")
    p.add_argument("--no-compile", action="store_true",
                   help="accepted for a uniform CLI; sat is never compiled")
    args = p.parse_args(argv)
    if args.steps is None:
        args.steps = STEPS[args.difficulty]
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
