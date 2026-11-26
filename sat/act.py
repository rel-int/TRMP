"""
The model without the lattice: the factor-graph :class:`~sat.model.Net`
run as a segmented recursion with adaptive computation time (ACT).

The lattice and ACT answer the same question -- how computation makes
progress across outer steps -- and they are incompatible. The search keeps
its progress in the discrete commitments and
:meth:`~sat.model.Net.initial` wipes messages, state and answer trace at
every step; ACT keeps its progress in the carried state and never resets
it. So the lattice is pinned blank here (:func:`blank_clue`), nothing is
committed or killed, the objective is the plain unconditioned clause loss
(:func:`unsat_loss`), the answer is the rounded readout, and escaping a bad
basin is the job of Gaussian noise on the answer trace (:func:`perturb`)
rather than of a commitment.

Two further substitutions are forced by SAT:

**The halt head reads clauses, not variables.** Sudoku's head is a
per-cell logit trained against per-cell correctness, which a unique
solution makes well defined. A satisfiable formula has many solutions, so
a variable being true is neither right nor wrong -- the smallest unit that
carries an exact target is the clause. :class:`HaltHead` therefore reads
each clause, and its per-instance logit is a size-corrected soft minimum
over them (:meth:`HaltHead.logit`), which stays comparable across
instances of very different clause counts.

**Verification is the metric, never the rule.** A satisfying assignment is
checkable for free, so :func:`clause_sat` replaces sudoku's comparison
against a target. It does *not* replace the halt rule, and it never
selects a rollout: halting on the verifier would make best-of-``k`` equal
pass@``k`` by construction, and the gap between those two is what measures
whether the head's confidence tracks correctness. On the splits reported
here the measurement comes back zero -- the head picks a satisfying
rollout whenever one exists -- which is a result and not the definition.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from discopy.neural.cell import as_widths

from .data import Batch, tile
from .model import Net, State, scatter_add, scatter_mean

EPS = 1e-6


def blank_clue(n_vars: int, device=None) -> torch.Tensor:
    """
    The all-undecided lattice :meth:`~sat.model.Net.initial` still reads,
    pinned for the whole run, so ``clue_encoder`` degenerates to a learned
    bias. The module is kept rather than removed: it costs one matmul a
    segment, it keeps :meth:`ActNet.segment`'s window re-injection
    identical to :meth:`~sat.model.Net.run`'s, which is the only way that
    encoder sees a gradient when the leading cycles are detached, and a
    lattice checkpoint and an ACT checkpoint stay loadable as each other.
    """
    return torch.ones(n_vars, 2, device=device)


def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor,
                      n: int) -> torch.Tensor:
    """ ``(n,)`` segmented ``logsumexp``, shifted by each segment's max. """
    top = src.new_full((n,), -float("inf")).scatter_reduce(
        0, index, src, reduce="amax", include_self=True)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    return top + torch.log(scatter_add((src - top[index]).exp(), index,
                                       n).clamp(min=EPS))


# -- the exact clause signals (the metric, not the halt rule) ----------------

def clause_sat(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``(C,)`` bool: is each clause satisfied by ``logits > 0``. """
    assignment = logits > 0
    literal = torch.where(batch.edge_sign == 1, assignment[batch.edge_var],
                          ~assignment[batch.edge_var])
    return scatter_add(literal.to(logits.dtype), batch.edge_clause,
                       batch.n_clauses) > 0.5


def solved_of(sat: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``(instances,)`` bool: every clause of the instance satisfied. """
    return scatter_add((~sat).to(torch.float32), batch.c_batch,
                       batch.n_instances) < 0.5


def sat_fraction(sat: torch.Tensor, batch: Batch) -> torch.Tensor:
    """
    ``(instances,)`` the fraction of the instance's clauses satisfied --
    the graded progress number, and the analogue of sudoku's per-cell
    accuracy. Reported alongside the solve rate; never used to select a
    rollout, which would be selecting on the verifier.
    """
    good = sat.to(torch.float32)
    total = scatter_add(torch.ones_like(good), batch.c_batch,
                        batch.n_instances)
    return scatter_add(good, batch.c_batch, batch.n_instances) / total.clamp(
        min=1)


# -- the unconditioned satisfaction loss ------------------------------------

def unsat_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``mean_c [-log P(clause c satisfied)]`` --
    :func:`sat.lattice.satisfaction_loss` with no lattice to condition on,
    written here so this module needs none. """
    probability = torch.sigmoid(logits)[batch.edge_var]
    per_literal = torch.where(batch.edge_sign == 1, probability,
                              1.0 - probability)
    log_unsat = scatter_add(torch.log((1.0 - per_literal).clamp(min=EPS)),
                            batch.edge_clause, batch.n_clauses)
    unsat = torch.exp(log_unsat).clamp(max=1.0 - EPS)
    return -torch.log((1.0 - unsat).clamp(min=EPS)).mean()


def deep_loss(every: tuple, batch: Batch) -> torch.Tensor:
    """ :func:`unsat_loss` averaged over every deep-supervision readout. """
    return torch.stack([unsat_loss(logits, batch) for logits in every]).mean()


# -- the halt head ----------------------------------------------------------

class HaltHead(nn.Module):
    """
    One logit per *clause* off the answer trace, aggregated to a
    per-instance soft minimum.

    The trace is per variable and the target per clause, so the head reads
    each clause's literals: the variable's trace and its polarity-signed
    copy, pooled over the clause by a size-corrected soft maximum, so
    "some literal is true" is expressible at all -- a mean pool cannot see
    ``edge_sign`` and can only learn a polarity-blind settledness proxy.
    Zero weight and bias ``-5`` are sudoku's init and survive the pooling
    exactly, so the model trains at full depth until the head has learned
    something, and building the head last leaves the trunk's weights those
    a headless net draws under the same seed.
    """

    def __init__(self, y_dim: int, polar: bool = True):
        super().__init__()
        self.polar = polar
        self.lin = nn.Linear(2 * y_dim if polar else y_dim, 1)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.fill_(-5.0)

    def read(self, answer: torch.Tensor, batch: Batch) -> torch.Tensor:
        """ ``(C,)`` per-clause halt logits. """
        if not self.polar:
            pooled = scatter_mean(answer[batch.edge_var], batch.edge_clause,
                                  batch.n_clauses)
            return self.lin(pooled).squeeze(-1)
        trace = answer[batch.edge_var]
        sign = (batch.edge_sign.to(trace.dtype) * 2.0 - 1.0).unsqueeze(-1)
        literal = self.lin(torch.cat([trace, trace * sign], -1)).squeeze(-1)
        arity = scatter_add(torch.ones_like(literal), batch.edge_clause,
                            batch.n_clauses).clamp(min=1)
        return scatter_logsumexp(literal, batch.edge_clause,
                                 batch.n_clauses) - arity.log()

    @staticmethod
    def logit(halt: torch.Tensor, batch: Batch) -> torch.Tensor:
        """
        ``(instances,)`` the instance-level halt logit: the size-corrected
        soft minimum ``-log mean_c exp(-q_c)`` over the instance's clauses.

        The correction is not cosmetic. Sudoku reduces over exactly 81
        cells, so the ``log n`` bias of a plain ``-logsumexp(-q)`` is a
        constant its ``-5`` init absorbs. Clause counts here vary by
        instance and by difficulty, so without ``- log(count)`` a larger
        formula reads as less confident and any fixed threshold becomes
        silently difficulty-dependent. Subtracting it leaves the aggregate
        minimum-dominated but size-invariant: at init every clause reads
        ``-5`` and so does every instance, whatever its size.
        """
        count = scatter_add(torch.ones_like(halt), batch.c_batch,
                            batch.n_instances).clamp(min=1)
        return -(scatter_logsumexp(-halt, batch.c_batch, batch.n_instances)
                 - count.log())

    def loss(self, halt: torch.Tensor, sat: torch.Tensor, batch: Batch,
             instance_weight: float = 1.0) -> torch.Tensor:
        """
        Per-clause BCE against exact clause satisfaction, plus the BCE of
        the *aggregated* logit against exact instance satisfaction.

        The second term has no sudoku counterpart and is not optional. A
        random assignment satisfies seven of eight 3-clauses, so the
        per-clause target is about 88 % positive from step zero: the head
        converges to "satisfied everywhere" and the soft minimum, which
        reads precisely the tail it has stopped modelling, carries no
        information. The instance-level target is balanced -- it is the
        solve rate -- and it is the quantity the halt rule consumes, so
        supervising it directly is what keeps :meth:`logit` calibrated.
        """
        target = sat.detach().to(halt.dtype)
        per_clause = F.binary_cross_entropy_with_logits(halt, target)
        if instance_weight <= 0:
            return per_clause
        hit = solved_of(sat, batch).detach().to(halt.dtype)
        instance = F.binary_cross_entropy_with_logits(
            self.logit(halt, batch), hit)
        return per_clause + instance_weight * instance


# -- the net ----------------------------------------------------------------

class ActNet(Net):
    """
    A :class:`~sat.model.Net` plus a :class:`HaltHead`, run one segment at
    a time over a carried :class:`~sat.model.State`.

    ``halt_detach`` has the head read a *detached* answer trace, so its
    loss trains the head alone: with an exact verifier available for the
    metric the head is pure readout, and letting it reach the trunk risks
    an objective that is already where it should be.

    ``random_window`` is training-only. The ``cycles - detached``
    differentiated cycles then sit at a uniformly random position inside
    the segment instead of always at its end: ``k`` leading cycles run
    under ``no_grad``, the window follows with its readouts, and the rest
    of the segment completes under ``no_grad`` so the carried state still
    sees every cycle. Memory stays that of one window and every position
    collects gradient over the run.
    """

    random_window: bool = True

    def __init__(self, widths, rounds: int = 6, cycles: int = 12,
                 detached: int = 6, fused: bool = True,
                 halt_detach: bool = True, halt_polar: bool = True):
        super().__init__(widths, rounds, cycles, detached, fused)
        # built last, so the trunk draws the weights a headless net draws
        self.halt = HaltHead(self.widths.y_dim, halt_polar)
        self.halt_detach, self.halt_polar = halt_detach, halt_polar

    def shape(self) -> dict:
        return {**super().shape(), "halt_detach": self.halt_detach,
                "halt_polar": self.halt_polar}

    def initial_act(self, batch: Batch, x: torch.Tensor) -> State:
        """ :meth:`~sat.model.Net.initial`, materialised: it returns
        expanded views of a zero scalar and of ``y0``, and a segmented loop
        index-assigns into what it carries. """
        return State(*(t.contiguous() for t in self.initial(batch, x)))

    def segment(self, s: State, batch: Batch, x: torch.Tensor,
                grad: bool = True) -> tuple:
        """
        One supervision segment from the carried state ``s``. Returns
        ``(state, every, halt, final)``: the state the segment ended on,
        still attached for the caller to detach, the readouts of every
        differentiated cycle, the raw per-clause halt logits off the final
        answer trace, and the final readout.
        """
        grad = grad and torch.is_grad_enabled()
        leading = self.detached
        if self.random_window and grad and self.training and leading:
            leading = int(torch.randint(0, self.detached + 1, ()))
        trailing = self.detached - leading

        with torch.no_grad():
            for _ in range(leading):
                s = self.cycle(s, batch)
        with torch.set_grad_enabled(grad):
            s = s._replace(clue=self.clue_encoder(x))
            every = []
            for _ in range(self.cycles - self.detached):
                s = self.cycle(s, batch)
                every.append(self.head(s.answer).squeeze(-1))
        final = every[-1]
        if trailing:
            with torch.no_grad():
                for _ in range(trailing):
                    s = self.cycle(s, batch)
            final = self.head(s.answer).squeeze(-1)
        with torch.set_grad_enabled(grad):
            answer = s.answer.detach() if self.halt_detach else s.answer
            halt = self.halt.read(answer, batch)
        return s, tuple(every), halt, final


def build(widths, rounds: int = 6, cycles: int = 12, detached: int = 6,
          fused: bool = True, halt_detach: bool = True,
          halt_polar: bool = True) -> ActNet:
    """ An :class:`ActNet` from a :class:`~discopy.neural.cell.Widths`, its tuple or
    its ``dict``. """
    return ActNet(as_widths(widths), rounds, cycles, detached, fused,
                  halt_detach, halt_polar)


def detach_state(s: State) -> State:
    """ Every carried tensor detached: the cut at a segment boundary. """
    return State(*(t.detach() for t in s))


def perturb(s: State, sigma: float, generator=None) -> State:
    """
    Gaussian noise on the answer trace -- the test-time knob that replaces
    the search's discrete commitment as the way out of a bad basin. The
    trace is written by :class:`~sat.model.Refresh`, straight out of a
    ``LayerNorm``, so its RMS is about one and ``sigma`` reads as a
    relative noise level. The clue is never kicked: it is the problem
    statement, not a trace.
    """
    if not sigma:
        return s
    noise = torch.randn(s.answer.shape, generator=generator,
                        device=s.answer.device, dtype=s.answer.dtype)
    return s._replace(answer=s.answer + sigma * noise)


# -- the rollouts -----------------------------------------------------------

@torch.no_grad()
def rollout(net, batch: Batch, segments: int, sigma: float = 0.0,
            threshold: float = 0.0, generator=None) -> dict:
    """
    One trajectory of ``segments`` segments from a blank state, reading at
    every segment:

    - ``act_*``: the answer frozen at the segment whose instance halt
      logit first cleared ``threshold``, or the last one if none did;
    - ``fixed_*``: the answer after exactly ``segments`` segments;
    - ``best_*``: the answer of the most confident segment so far, the
      running arg-max of the halt logit. That is test-time legal -- it
      reads the model's own logit and nothing else -- and it is what makes
      a deep noisy trajectory worth running: a rollout that passes through
      a satisfying assignment at segment 90 and wanders off still scores
      it.
    """
    net.eval()
    device = batch.edge_var.device
    x = blank_clue(batch.n_vars, device)
    s = net.initial_act(batch, x)
    n = batch.n_instances
    zeros = torch.zeros(n, device=device)
    false = torch.zeros(n, dtype=torch.bool, device=device)
    halted, step = false.clone(), torch.zeros(n, dtype=torch.long,
                                              device=device)
    act_solved, act_frac, act_q = false.clone(), zeros.clone(), zeros.clone()
    best_q = torch.full((n,), -float("inf"), device=device)
    best_solved, best_frac = false.clone(), zeros.clone()
    solved = frac = q = None

    for index in range(1, segments + 1):
        s = perturb(s, sigma, generator)
        s, every, halt, _ = net.segment(s, batch, x, grad=False)
        s = detach_state(s)
        q = HaltHead.logit(halt.float(), batch)
        sat = clause_sat(every[-1], batch)
        solved, frac = solved_of(sat, batch), sat_fraction(sat, batch)

        fresh = ~halted & (q > threshold)
        step = torch.where(fresh, torch.full_like(step, index), step)
        act_solved = torch.where(fresh, solved, act_solved)
        act_frac = torch.where(fresh, frac, act_frac)
        act_q = torch.where(fresh, q, act_q)
        halted = halted | fresh

        better = q > best_q
        best_solved = torch.where(better, solved, best_solved)
        best_frac = torch.where(better, frac, best_frac)
        best_q = torch.where(better, q, best_q)
        if bool(halted.all()):
            break                  # a halted instance's answer never moves

    return {"fixed_solved": solved, "fixed_frac": frac,
            "act_solved": torch.where(halted, act_solved, solved),
            "act_frac": torch.where(halted, act_frac, frac),
            "act_q": torch.where(halted, act_q, q),
            "act_depth": torch.where(halted, step,
                                     torch.full_like(step, index)).float(),
            "best_solved": best_solved, "best_frac": best_frac,
            "best_q": best_q}


@torch.no_grad()
def evaluate_fixed(net, batch: Batch, segments: int) -> dict:
    """ Solve rate and satisfied-clause fraction after exactly
    ``segments``, with no halt rule. """
    out = rollout(net, batch, segments, threshold=float("inf"))
    return {"solved": float(out["fixed_solved"].float().mean()),
            "frac": float(out["fixed_frac"].mean())}


@torch.no_grad()
def evaluate_act(net, batch: Batch, cap: int, threshold: float = 0.0) -> dict:
    """ Solve rate under the halt rule, and the mean depth it ran. """
    out = rollout(net, batch, cap, threshold=threshold)
    return {"solved": float(out["act_solved"].float().mean()),
            "frac": float(out["act_frac"].mean()),
            "depth": float(out["act_depth"].mean())}


@torch.no_grad()
def evaluate_best_of_k(net, batch: Batch, cap: int, k: int = 8,
                       sigma: float = 0.3, threshold: float = 0.0,
                       seed: int = 123) -> dict:
    """
    ``k`` noisy rollouts per instance, each halting by its own rule,
    keeping the one whose halt logit at its halting segment is largest.

    The rollouts are one wide :func:`~sat.data.tile`d batch, so this costs
    one trajectory ``k`` times as wide rather than ``k`` trajectories.
    ``pass_at_k`` is the oracle over the ``k`` and ``best_of_k`` what the
    model's own confidence picks; the gap between them is the calibration
    number.
    """
    device = batch.edge_var.device
    wide = tile(batch, k, device)
    generator = torch.Generator(device=device).manual_seed(
        int(np.random.SeedSequence([seed]).generate_state(1)[0]))
    out = rollout(net, wide, cap, sigma, threshold, generator)

    n = batch.n_instances
    solved = out["act_solved"].view(k, n)
    chosen = out["act_q"].view(k, n).argmax(0)
    columns = torch.arange(n, device=device)
    return {"single": float(solved.float().mean()),
            "best_of_k": float(solved[chosen, columns].float().mean()),
            "pass_at_k": float(solved.any(0).float().mean()),
            "frac": float(out["act_frac"].view(k, n)[chosen, columns].mean()),
            "depth": float(out["act_depth"].view(k, n)[chosen,
                                                       columns].mean())}
