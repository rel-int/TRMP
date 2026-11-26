"""
The fused maze round and the fast ACT segment: the PyTorch round of
:mod:`maze.model` reassociated (the subset the recipes run).

* every round after the first of a segment part runs the **fused round**:
  the port-expanded ``(B, 900, 7, hidden)`` tensor of
  :class:`discopy.neural.cell.Site` is never built, the neighbour ports are
  a stencil on the cell's own encoded belief and the three relation
  replies leave already encoded against the site's port map.  The carried
  state stays in the form of :class:`discopy.neural.recursion.PortState`:
  a segment part *enters* the fused form through one PyTorch-shaped round
  (:func:`entry_round`, which also serves a freshly refilled slot whose
  ports are zero) and *leaves* it through :func:`exit_state`, so the
  boundary between a ``no_grad`` prefix and the differentiated window
  differentiates exactly what the PyTorch round does.
* forward only (the lattice net's detached prefix and its inference steps,
  the ACT prefix and validation), the rounds run on the Triton round of
  :mod:`maze.kernel` when ``net.fast_kernel`` is set, else as whole
  compiled cycles.
* the ACT window's per-board clip (:mod:`discopy.neural.train`) is taken
  with **per-board parameter copies** in plain autograd: every parameter
  is expanded to ``(B, ...)``, the fused maps are composed per board, the
  window runs on ``bmm`` and ``sum_b loss_b`` is back-propagated once --
  the gradient landing on copy ``b`` is board ``b``'s row.  The window is
  recomputed block by block in the backward (:class:`Recompute`) and each
  block is compiled.

Not included (off in both recipes): a pipelined Triton round, a bf16
prefix autocast and region timers.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F
from discopy.neural.act import detach_state, halt_logit
from discopy.neural.recursion import PortState
from discopy.neural.train import ratios, rescale, scales

from .graph import DEGREE, neighbours
from .model import decode

__all__ = ["Maps", "fuse", "run_cycles", "act_step", "train_segment",
           "per_board_params", "Recompute"]


class Maps(NamedTuple):
    """
    The composed maps of one round (2-D / 1-D shared, or with a leading
    board axis when built from per-board parameter copies).
    """
    shared_w: torch.Tensor      #: (h, s) encode's shared half
    shared_b: torch.Tensor      #: (h, )
    wire_w: torch.Tensor        #: (h, s) the belief encoded against the port map
    wire_b: torch.Tensor
    hh_w: torch.Tensor          #: (3s, s) the GRU's hh
    hh_b: torch.Tensor
    line_w: torch.Tensor        #: (2h, s) phi | rho's member half on emit(state)
    line_b: torch.Tensor
    line_raw_w: torch.Tensor    #: (2h, dim) the same on a raw belief (entry)
    line_raw_b: torch.Tensor
    line_pool: torch.Tensor     #: (h, h) rho's pooled half
    line_port: torch.Tensor     #: (h, h) rho[2] into the port map
    line_port_b: torch.Tensor
    read_w: torch.Tensor
    read_b: torch.Tensor
    read_raw_w: torch.Tensor
    read_raw_b: torch.Tensor
    read_pool: torch.Tensor
    read_port: torch.Tensor
    read_port_b: torch.Tensor
    gate: torch.Tensor          #: (3s, h) encode[2] into the GRU's input gates
    gate_b: torch.Tensor        #: (3s, )
    clue_w: torch.Tensor        #: (3s, dim)
    answer_w: torch.Tensor      #: (3s, y)
    port_half: torch.Tensor     #: (h, dim) encode's port half (entry round)
    ln_g: torch.Tensor          #: (s, ) the cell norm
    ln_b: torch.Tensor
    r_ih_w: torch.Tensor        #: the refresh GRU and norm
    r_ih_b: torch.Tensor
    r_hh_w: torch.Tensor
    r_hh_b: torch.Tensor
    r_ln_g: torch.Tensor
    r_ln_b: torch.Tensor
    emit_w: torch.Tensor        #: (dim, s) for the exit
    emit_b: torch.Tensor
    enc_w: torch.Tensor         #: the encoder
    enc_b: torch.Tensor
    sm_w: torch.Tensor          #: the readout head
    sm_b: torch.Tensor
    halt_w: torch.Tensor        #: the halt head
    halt_b: torch.Tensor
    wide_w: torch.Tensor        #: (h + 2h + 2h + h + 3s, s) the five state
    #: blocks as one map (shared | line | read | wire | hh), for a round
    #: that applies them as one product (``net.fast_wide``)
    wide_b: torch.Tensor


#: The parameters a segment reads (``bce``, ``conf`` and ``y0`` are the
#: lattice's; a segment never touches them, and their rows would be zero --
#: left as autograd leaves them, ``.grad`` ``None``).
USED = ("cell.", "line.", "readout.", "refresh.", "encoder.", "sm.", "halt.")


def used(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in USED)


# -- small algebra that is batch-agnostic ------------------------------------

def mm(a, b):
    return torch.matmul(a, b)


def mv(a, b):
    return torch.matmul(a, b.unsqueeze(-1)).squeeze(-1)


def lin(x, w, b=None):
    """
    ``x @ w.T + b`` for ``x`` ``(B, n, in)`` [or ``(B, n, k, in)``] and a
    shared ``w`` ``(out, in)`` (one GEMM) or a per-board ``w`` ``(B, out,
    in)`` (one ``bmm``); ``b`` ``(out, )`` / ``(B, out)`` / ``None``.
    """
    if w.dim() == 2:
        return F.linear(x, w, b)
    rows = x.shape[0]
    flat = x.reshape(rows, -1, x.shape[-1])
    y = torch.matmul(flat, w.transpose(-1, -2))
    if b is not None:
        y = y + b.unsqueeze(-2)
    return y.reshape(*x.shape[:-1], w.shape[-2])


def layer_norm(x, g, b, eps: float):
    width = x.shape[-1]
    if g.dim() == 1:
        return F.layer_norm(x, (width, ), g, b, eps)
    return F.layer_norm(x, (width, ), None, None, eps) * g.unsqueeze(-2) \
        + b.unsqueeze(-2)


def gru(gi, gh, carried):
    """ :func:`discopy.neural.cell.gru_cell` on precomputed gates. """
    width = carried.shape[-1]
    gate_r, gate_z, gate_n = torch.split(gi, width, -1)
    hidden_r, hidden_z, hidden_n = torch.split(gh, width, -1)
    reset = torch.sigmoid(gate_r + hidden_r)
    keep = torch.sigmoid(gate_z + hidden_z)
    fresh = torch.tanh(gate_n + reset * hidden_n)
    return fresh + keep * (carried - fresh)


# -- the composed maps ---------------------------------------------------------

def fuse(net, params: dict | None = None) -> Maps:
    """
    :class:`Maps` of ``net`` from its parameters, or from ``params`` (a
    ``name -> tensor`` dict, each with a leading board axis, see
    :func:`per_board_params`).  Differentiable back to whichever was
    given.
    """
    w = net.widths
    h, s, dim = w.hidden, w.state_dim, w.dim
    P = params if params is not None else dict(net.named_parameters())
    first_w, first_b = P["cell.encode.0.weight"], P["cell.encode.0.bias"]
    shared_w, port_half = first_w[..., :s].contiguous(), \
        first_w[..., s:].contiguous()
    emit_w, emit_b = P["cell.emit.weight"], P["cell.emit.bias"]
    second_w, second_b = P["cell.encode.2.weight"], P["cell.encode.2.bias"]
    w_ih, b_ih = P["cell.update.weight_ih"], P["cell.update.bias_ih"]
    gate_in = w_ih[..., :h]
    rel = {}
    for name, short in (("line", "line"), ("readout", "read")):
        phi_w, phi_b = P[f"{name}.phi.0.weight"], P[f"{name}.phi.0.bias"]
        rho_w, rho_b = P[f"{name}.rho.0.weight"], P[f"{name}.rho.0.bias"]
        rho2_w, rho2_b = P[f"{name}.rho.2.weight"], P[f"{name}.rho.2.bias"]
        wide = torch.cat([phi_w, rho_w[..., :dim]], -2)
        wide_b = torch.cat([phi_b, rho_b], -1)
        rel.update({
            f"{short}_w": mm(wide, emit_w),
            f"{short}_b": mv(wide, emit_b) + wide_b,
            f"{short}_raw_w": wide.contiguous(), f"{short}_raw_b": wide_b,
            f"{short}_pool": rho_w[..., dim:].contiguous(),
            f"{short}_port": mm(port_half, rho2_w),
            f"{short}_port_b": mv(port_half, rho2_b)})
    wire_w, wire_b = mm(port_half, emit_w), mv(port_half, emit_b)
    hh_w, hh_b = P["cell.update.weight_hh"], P["cell.update.bias_hh"]
    return Maps(
        shared_w=shared_w, shared_b=first_b,
        wire_w=wire_w, wire_b=wire_b, hh_w=hh_w, hh_b=hh_b,
        **rel,
        wide_w=torch.cat([shared_w, rel["line_w"], rel["read_w"], wire_w,
                          hh_w], -2),
        wide_b=torch.cat([first_b, rel["line_b"], rel["read_b"], wire_b,
                          hh_b], -1),
        gate=mm(gate_in, second_w), gate_b=mv(gate_in, second_b) + b_ih,
        clue_w=w_ih[..., h:h + dim].contiguous(),
        answer_w=w_ih[..., h + dim:].contiguous(),
        port_half=port_half,
        ln_g=P["cell.norm.weight"], ln_b=P["cell.norm.bias"],
        r_ih_w=P["refresh.module.weight_ih"], r_ih_b=P["refresh.module.bias_ih"],
        r_hh_w=P["refresh.module.weight_hh"], r_hh_b=P["refresh.module.bias_hh"],
        r_ln_g=P["refresh.norm.weight"], r_ln_b=P["refresh.norm.bias"],
        emit_w=emit_w, emit_b=emit_b,
        enc_w=P["encoder.weight"], enc_b=P["encoder.bias"],
        sm_w=P["sm.weight"], sm_b=P["sm.bias"],
        # the lattice net has no halt head: its forward-only cycles read
        # none of these
        halt_w=P.get("halt.weight"), halt_b=P.get("halt.bias"))


def per_board_params(net, rows: int) -> dict:
    """ ``name -> (rows, ...)`` leaf copies of every parameter a segment
    reads; the gradient landing on copy ``b`` is board ``b``'s row. """
    return {name: p.detach().unsqueeze(0).expand(rows, *p.shape).clone()
            .requires_grad_(True)
            for name, p in net.named_parameters() if used(name)}


# -- the rounds ----------------------------------------------------------------

def replies(m: Maps, line, read, side: int):
    """
    The three relations' replies encoded against the port map, ``(3, B,
    cells, h)``: ``line`` / ``read`` are the ``phi | rho-member``
    pre-activations on every cell's belief, ``(B, cells, 2h)``.
    """
    rows, cells, two_h = line.shape
    h = two_h // 2
    embedded = F.relu(line[..., :h]).view(rows, side, side, h)
    member = line[..., h:].reshape(rows, side, side, h)
    by_row = lin(embedded.mean(2), m.line_pool).view(rows, side, 1, h)
    by_column = lin(embedded.mean(1), m.line_pool).view(rows, 1, side, h)
    l_row = lin(F.relu(member + by_row).view(rows, cells, h), m.line_port,
                m.line_port_b)
    l_col = lin(F.relu(member + by_column).view(rows, cells, h), m.line_port,
                m.line_port_b)
    embedded = F.relu(read[..., :h])
    pooled = lin(embedded.mean(1, keepdim=True), m.read_pool)
    r = lin(F.relu(read[..., h:] + pooled), m.read_port, m.read_port_b)
    return torch.stack([l_row, l_col, r])


def pool(shared, ports, wire, side: int):
    """ The site's mean over its seven ports: the three replies and the
    stencil of the encoded belief, zero off the grid. """
    rows, cells, h = shared.shape
    total = F.relu(shared + ports[0]) + F.relu(shared + ports[1]) \
        + F.relu(shared + ports[2])
    grid = F.pad(wire.view(rows, side, side, h), (0, 0, 1, 1, 1, 1))
    sh = shared.view(rows, side, side, h)
    total = total.view(rows, side, side, h) \
        + F.relu(sh + grid[:, :-2, 1:-1]) + F.relu(sh + grid[:, 2:, 1:-1]) \
        + F.relu(sh + grid[:, 1:-1, :-2]) + F.relu(sh + grid[:, 1:-1, 2:])
    return total.reshape(rows, cells, h) / DEGREE


def update(m: Maps, encoded, hidden, gates, state, eps: float):
    gi = lin(encoded, m.gate) + gates
    return layer_norm(gru(gi, hidden, state), m.ln_g, m.ln_b, eps)


def fused_round(m: Maps, state, ports, gates, side: int, eps: float,
                wide: bool = False):
    """ One fused round: ``(state, ports)`` in, ``(state, ports)`` out.
    ``wide``: the five maps on the state as one product, split. """
    if wide:
        h, s = m.shared_w.shape[-2], state.shape[-1]
        shared, line, read, wire, hidden = torch.split(
            lin(state, m.wide_w, m.wide_b), [h, 2 * h, 2 * h, h, 3 * s], -1)
    else:
        shared = lin(state, m.shared_w, m.shared_b)
        line = lin(state, m.line_w, m.line_b)
        read = lin(state, m.read_w, m.read_b)
        wire = lin(state, m.wire_w, m.wire_b)
        hidden = lin(state, m.hh_w, m.hh_b)
    new_ports = replies(m, line, read, side)
    encoded = pool(shared, ports, wire, side)
    return update(m, encoded, hidden, gates, state, eps), new_ports


def entry_round(m: Maps, cells, units, state, gates, side: int, eps: float):
    """
    The PyTorch-shaped first round of a segment part: the site on the
    carried port messages ``cells`` ``(B, cells, 7, dim)`` (zeros on a
    fresh slot), the relations on the carried beliefs ``units``.
    """
    shared = lin(state, m.shared_w, m.shared_b)
    per_port = lin(cells, m.port_half)
    encoded = F.relu(shared.unsqueeze(-2) + per_port).mean(-2)
    hidden = lin(state, m.hh_w, m.hh_b)
    line = lin(units, m.line_raw_w, m.line_raw_b)
    read = lin(units, m.read_raw_w, m.read_raw_b)
    new_ports = replies(m, line, read, side)
    return update(m, encoded, hidden, gates, state, eps), new_ports


def refresh(m: Maps, state, answer, eps: float):
    gi = lin(state, m.r_ih_w, m.r_ih_b)
    gh = lin(answer, m.r_hh_w, m.r_hh_b)
    return layer_norm(gru(gi, gh, answer), m.r_ln_g, m.r_ln_b, eps)


def gates_of(m: Maps, clue_term, answer):
    return clue_term + lin(answer, m.answer_w)


def fused_block(m: Maps, state, ports, gates, side: int, eps: float,
                rounds: int, wide: bool):
    """ ``rounds`` fused rounds (one recomputation / compilation unit). """
    for _ in range(rounds):
        state, ports = fused_round(m, state, ports, gates, side, eps, wide)
    return state, ports


def fused_cycle(m: Maps, state, ports, clue_term, answer, side: int,
                eps: float, rounds: int, wide: bool):
    """ ``rounds`` fused rounds then the refresh; also returns the state
    before the last round (what :func:`exit_state` needs). """
    gates = gates_of(m, clue_term, answer)
    prev = state
    for _ in range(rounds):
        prev = state
        state, ports = fused_round(m, state, ports, gates, side, eps, wide)
    return state, ports, refresh(m, state, answer, eps), prev


def entry_cycle(m: Maps, cells, units, state, clue_term, answer, side: int,
                eps: float, rounds: int, wide: bool):
    gates = gates_of(m, clue_term, answer)
    prev = state
    state, ports = entry_round(m, cells, units, state, gates, side, eps)
    for _ in range(rounds - 1):
        prev = state
        state, ports = fused_round(m, state, ports, gates, side, eps, wide)
    return state, ports, refresh(m, state, answer, eps), prev


@torch.no_grad()
def exit_state(net, state, prev, clue, answer) -> PortState:
    """
    The PyTorch form of the fused state ``state`` (after its last round,
    ``prev`` before it): ``cells = [neighbours(emit(state)),
    replies(emit(prev))]``, ``units = emit(state)``.
    """
    belief = net.cell.emit(state)
    return PortState(torch.cat([neighbours(belief, net.side),
                                net.replies(net.cell.emit(prev))], 2),
                     belief, state, clue, answer)


# -- recomputation -------------------------------------------------------------

class Recompute(torch.autograd.Function):
    """
    ``fn(*tensors)`` with nothing saved but its inputs: the forward runs
    under ``no_grad``, the backward runs it again with a gradient and
    differentiates through that.  Plain autograd (no saved-tensor hooks),
    so a ``torch.compile``d ``fn`` is fine in both passes.
    """
    @staticmethod
    def forward(ctx, fn, *tensors):
        ctx.fn = fn
        ctx.autocast = (torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(*tensors)
        with torch.no_grad():
            out = fn(*tensors)
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        saved = ctx.saved_tensors
        enabled, dtype = ctx.autocast
        with torch.enable_grad(), torch.autocast("cuda", dtype=dtype,
                                                 enabled=enabled):
            ins = [t.detach().requires_grad_(t.requires_grad) for t in saved]
            out = ctx.fn(*ins)
        wanted = [t for t in ins if t.requires_grad]
        pairs = [(o, g) for o, g in zip(out, grad_out)
                 if g is not None and o.requires_grad]
        grads = torch.autograd.grad([o for o, _ in pairs], wanted,
                                    [g for _, g in pairs], allow_unused=True)
        it = iter(grads)
        return (None, ) + tuple(next(it) if t.requires_grad else None
                                for t in ins)


# -- compiled callables --------------------------------------------------------

def _block_fn(side: int, eps: float, rounds: int, wide: bool):
    def fn(state, ports, gates, *maps):
        return fused_block(Maps(*maps), state, ports, gates, side, eps,
                           rounds, wide)
    return fn


def _entry_round_fn(side: int, eps: float):
    def fn(cells, units, state, gates, *maps):
        return entry_round(Maps(*maps), cells, units, state, gates, side, eps)
    return fn


def _refresh_fn(side: int, eps: float):
    def fn(state, clue_term, answer, *maps):
        m = Maps(*maps)
        new = refresh(m, state, answer, eps)
        return new, gates_of(m, clue_term, new)
    return fn


def _cycle_fn(side: int, eps: float, rounds: int, wide: bool):
    def fn(state, ports, clue_term, answer, *maps):
        return fused_cycle(Maps(*maps), state, ports, clue_term, answer,
                           side, eps, rounds, wide)
    return fn


def _entry_fn(side: int, eps: float, rounds: int, wide: bool):
    def fn(cells, units, state, clue_term, answer, *maps):
        return entry_cycle(Maps(*maps), cells, units, state, clue_term,
                           answer, side, eps, rounds, wide)
    return fn


def callable_for(net, kind: str, rounds: int | None = None):
    """ The (compiled, per ``net.fast_compile``) block / cycle / entry-cycle
    / entry-round / refresh function, cached on the net. """
    cache = net.__dict__.setdefault("_fast_cache", {})
    wide = bool(getattr(net, "fast_wide", False))
    rounds = net.rounds if rounds is None else rounds
    key = (kind, rounds, wide)
    if key not in cache:
        eps = float(net.cell.norm.eps)
        fn = {"block": lambda: _block_fn(net.side, eps, rounds, wide),
              "cycle": lambda: _cycle_fn(net.side, eps, rounds, wide),
              "entry": lambda: _entry_fn(net.side, eps, rounds, wide),
              "entry_round": lambda: _entry_round_fn(net.side, eps),
              "refresh": lambda: _refresh_fn(net.side, eps)}[kind]()
        mode = getattr(net, "fast_compile", "default")
        if mode and mode != "none":
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = max(
                64, torch._dynamo.config.cache_size_limit)
            fn = torch.compile(fn, dynamic=False, mode=mode)
        cache[key] = fn
    return cache[key]


# -- the segment ---------------------------------------------------------------

def run_cycles(net, m: Maps, s: PortState, n: int, grad: bool,
               readouts: bool) -> tuple:
    """
    ``n`` cycles from the PyTorch state ``s`` (its first round the
    :func:`entry_round`), back to a PyTorch state; ``every`` the readouts
    after each cycle when ``readouts``.  With a gradient the rounds are
    recomputed in the backward (:class:`Recompute`) when
    ``net.fast_checkpoint`` is set, and each block of rounds is its own
    compiled unit; forward-only, a round is the Triton round under
    ``net.fast_kernel``, else a cycle is one compiled unit.
    """
    side, eps, rounds = net.side, float(net.cell.norm.eps), net.rounds
    clue_term = lin(s.clue, m.clue_w, m.gate_b)
    # a fresh pool's state is ``initial``'s expanded zeros (stride 0); the
    # compiled units are specialised on strides, so hand them one layout
    s = PortState(*(None if t is None else t.contiguous() for t in s))
    state, answer = s.state, s.answer
    every = []
    checkpoint = grad and getattr(net, "fast_checkpoint", True)
    kernel = not grad and getattr(net, "fast_kernel", False) \
        and state.dtype == torch.float32 and m.shared_w.dim() == 2
    if kernel:
        # the Triton round (maze.kernel) for every round but the first of
        # a segment part; two scratches alternate, so the state before the
        # last round (``prev``) is still there for the exit
        from . import kernel as fast_kernel
        if not fast_kernel.HAVE_TRITON:
            raise RuntimeError("fast_kernel needs triton")
        entry1, refresh_ = callable_for(net, "entry_round"), \
            callable_for(net, "refresh")
        options = dict(rows_per_program=int(getattr(net, "fast_kernel_rows", 2)),
                       precision=getattr(net, "fast_kernel_prec", "tf32"),
                       num_warps=int(getattr(net, "fast_kernel_warps", 4)))
        # the storage dtype of what a round hands the next (state, replies,
        # phi, the wire and the gates): fp32, or bf16 for half the bytes;
        # the part's entry, the refresh between cycles and the exit stay
        # fp32 either way
        store = getattr(net, "fast_kernel_store", "fp32")
        store_dtype = fast_kernel.STORE_DTYPES[store]
        round_kernel = fast_kernel.round_kernel
        options["store"] = store
        cells, h = state.shape[1], m.shared_w.shape[0]
        scratch = [fast_kernel.Scratch(state.shape[0], cells, state.shape[2],
                                       h, state, store) for _ in range(2)]
        gates = gates_of(m, clue_term, answer)
        for index in range(n):
            first = 0
            if index == 0:
                prev = state
                state, ports = entry1(s.cells, s.units, state, gates, *m)
                first = 1
            stored = gates.to(store_dtype)        # a no-op at fp32
            for _ in range(first, rounds):
                prev = state
                target = scratch[1] if state is scratch[0].state else scratch[0]
                state, ports = round_kernel(
                    m, state, ports, stored, side, eps, scratch=target, **options)
            answer, gates = refresh_(state.float(), clue_term, answer, *m)
            if readouts:
                every.append(lin(answer, m.sm_w, m.sm_b))
        state, prev = state.float(), prev.float()  # no-ops at fp32
    elif not grad:
        entry, cycle = callable_for(net, "entry"), callable_for(net, "cycle")
        for index in range(n):
            if index == 0:
                state, ports, answer, prev = entry(
                    s.cells, s.units, state, clue_term, answer, *m)
            else:
                state, ports, answer, prev = cycle(state, ports, clue_term,
                                                   answer, *m)
            if readouts:
                every.append(lin(answer, m.sm_w, m.sm_b))
    else:
        # the window: blocks of ``fast_block`` rounds, each one compiled
        # unit recomputed in the backward; the last block of a cycle is
        # a single round so that ``prev`` (the state before the last
        # round) is at hand for the exit
        block = max(1, int(getattr(net, "fast_block", 1)))
        for index in range(n):
            gates = gates_of(m, clue_term, answer)
            first = 0
            if index == 0:
                prev = state
                state, ports = entry_round(m, s.cells, s.units, state, gates,
                                           side, eps)
                first = 1
            todo = rounds - first
            while todo > 0:
                size = 1 if todo == 1 else min(block, todo - 1)
                fn = callable_for(net, "block", size)
                prev = state
                if checkpoint:
                    state, ports = Recompute.apply(fn, state, ports, gates, *m)
                else:
                    state, ports = fn(state, ports, gates, *m)
                todo -= size
            answer = refresh(m, state, answer, eps)
            if readouts:
                every.append(lin(answer, m.sm_w, m.sm_b))
    out = exit_state(net, state.detach(), prev.detach(), s.clue, answer)
    # the carried state is detached by the caller anyway; keep the answer
    # (and the clue) attached so the halt head reads what the PyTorch
    # round's would
    return out._replace(answer=answer, state=state), every


def act_step(net, s: PortState, x, grad: bool = True,
             params: dict | None = None) -> tuple:
    """
    :meth:`maze.model.ActNet.act_step` on the fused segment: ``(state,
    every, halt, final)``.  ``params`` (per-board copies) makes the window a
    per-board pass -- the maps are then ``(B, ...)`` and every product a
    ``bmm``.
    """
    grad = grad and torch.is_grad_enabled()
    if grad and net.slot_clip is not None and params is None:
        raise ValueError("net.slot_clip is set but this segment is being "
                         "differentiated through plain batched autograd; "
                         "the trainer runs it with per-board copies")
    leading = net.detached
    window = net.cycles - leading
    with torch.no_grad():
        if leading:
            s, _ = run_cycles(net, fuse(net), s, leading, False, False)
    with torch.set_grad_enabled(grad):
        m = fuse(net, params)
        s = s._replace(clue=lin(x, m.enc_w, m.enc_b))
        s, every = run_cycles(net, m, s, window, grad, net.deep)
        if not net.deep:
            every.append(lin(s.answer, m.sm_w, m.sm_b))
        answer = s.answer
        halt = lin(answer.detach() if net.halt_detach else answer,
                   m.halt_w, m.halt_b).squeeze(-1)
    return s, tuple(every), halt, every[-1]


def _finite_max(t: torch.Tensor) -> float:
    finite = torch.isfinite(t)
    return float(t[finite].max()) if bool(finite.any()) else float("inf")


# the full clip statistics for the log (discopy.neural.train keeps only two)
def clip_stats(norms, ref, scale, reference: float | None = None) -> dict:
    """ What a log line reports of one step's clip: ``median`` / ``max`` of
    the clipped ratio, ``norm_median`` / ``norm_max`` of the raw norms,
    ``clipped_frac``, ``mean_scale``, ``nonfinite`` and ``reference`` (the
    value the cap was ``k`` times). """
    r = ratios(norms, ref)
    median = float(r.nanmedian())
    return dict(
        median=median, max=_finite_max(r),
        norm_median=float(norms.nanmedian()), norm_max=_finite_max(norms),
        clipped_frac=float((scale < 1.0).to(torch.float64).mean()),
        mean_scale=float(scale.to(torch.float64).mean()),
        nonfinite=int((~torch.isfinite(r)).sum()),
        reference=median if reference is None else float(reference))


def train_segment(trainer) -> tuple:
    """
    One training segment of the ACT pool with the per-board clip: the
    prefix batched and forward-only, the window on per-board parameter
    copies, one backward of ``sum_b loss_b``, every board's row (the halt
    head's excepted) scaled by ``min(1, k * reference / r_b)`` and the
    rows summed into ``.grad`` as the batch mean.
    """
    net, cfg = trainer.net, trainer.cfg
    k = net.slot_clip
    rows = trainer.x.shape[0]
    params = per_board_params(net, rows)
    with trainer.amp():
        state, every, halt, final = act_step(net, trainer.state, trainer.x,
                                             True, params)
        n = net.n_channels
        flat = trainer.target.reshape(-1)
        work = torch.promote_types(every[0].dtype, torch.float32)
        ce_cells = sum(F.cross_entropy(logits.reshape(-1, n).to(work), flat,
                                       reduction="none")
                       for logits in every) / len(every)
        with torch.no_grad():
            cell_ok = decode(final, trainer.x) == trainer.target
        q_cells = F.binary_cross_entropy_with_logits(
            halt, cell_ok.detach().to(halt.dtype), reduction="none")
        # board b's own loss: the rows are the gradients of these
        ce_b = ce_cells.reshape(rows, -1).mean(1)
        q_b = q_cells.reshape(rows, -1).mean(1)
        if cfg.halt_board_weight:
            # the board-level halt loss on the soft-min logit
            q_b = q_b + cfg.halt_board_weight \
                * F.binary_cross_entropy_with_logits(
                    halt_logit(halt).to(work),
                    cell_ok.all(-1).to(work),
                    reduction="none").to(q_b.dtype)
        board = ce_b + cfg.halt_weight * q_b
    board.sum().backward()
    ce, q = ce_b.mean().detach(), q_b.mean().detach()
    loss = ce + cfg.halt_weight * q
    ref = board.detach()
    counted = {name for name in params if not name.startswith("halt.")}
    work = torch.float64 if ref.dtype == torch.float64 else torch.float32
    # the squared norms are summed in registration order (deterministic;
    # iterating the set would follow the process's string hash seed)
    squares = sum(params[name].grad.reshape(rows, -1).to(work).pow(2).sum(1)
                  for name in params if name in counted)
    norms = torch.sqrt(squares)
    ref = ref.to(work)
    # the slow reference of the cap (None on the first step: this batch's
    # median)
    reference = net.slot_ref_running
    scale = scales(norms, k, ref, reference) if k is not None \
        else torch.ones_like(norms)
    stats = clip_stats(norms, ref, scale, reference) if k is not None else None
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name not in params:
                continue
            row = params[name].grad
            if row is None or not bool(row.any()):
                continue
            if name in counted and k is not None:
                row = rescale(row, scale)
            total = row.sum(0) / rows
            p.grad = total if p.grad is None else p.grad + total
    net.slot_clip_stats = stats
    return (detach_state(state), tuple(e.detach() for e in every),
            halt.detach(), final.detach(), loss, ce, q, cell_ok.all(-1))
