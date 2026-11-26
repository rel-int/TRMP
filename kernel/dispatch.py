"""
The nets that run on the segment kernels: the library's recursions
(:mod:`discopy.neural.recursion`) with a kernel policy on top.

* :class:`Net` -- :class:`~discopy.neural.recursion.PeerNet` whose
  supervision step (:meth:`Net.run`) runs on the kernels when
  :attr:`Dispatch.kernel` allows it;
* :class:`FactorNet` -- :class:`~discopy.neural.recursion.FactorNet`, which
  has no kernel: ``"auto"`` falls back to the PyTorch path, a mode raises;
* :class:`Act` -- :class:`Net` with a halt head, its segment's detached
  prefix and differentiated window on the kernels;
* :class:`ActPool` -- :class:`discopy.neural.act.ActPool` whose segment runs
  on the kernels, the per-board clip inside the kernel backward.

Every net prints once which path it takes.
"""

from __future__ import annotations

import torch

from discopy.neural import act, recursion

from .autograd import segment, segment_train
from .build import resolve, supports


class Dispatch:
    """ The segment-kernel policy of a net, and its supervision step. """
    #: The segment-kernel policy, a runtime choice: ``"auto"`` runs the CUDA
    #: kernels wherever :func:`kernel.supports` allows (``tf32`` under bf16
    #: autocast, ``fp32`` otherwise), ``"off"`` never, a mode name
    #: (``"fp32"``, ``"tf32"``, ``"fma"``) requires them.
    kernel: str | None = "auto"
    _kernel_logged = False

    def kernel_mode(self, t) -> str | None:
        """ The kernel mode for tensors like ``t`` (``None``: PyTorch path);
        the first call prints which path is taken. """
        mode = resolve(self, t.device, t.dtype)
        if not self._kernel_logged:
            self._kernel_logged = True
            why = (f"segment kernel {mode}" if mode is not None else
                   "PyTorch path: kernel=off"
                   if self.kernel in (None, "off", False) else
                   "PyTorch path: " + supports(self, t.device, t.dtype)[1])
            print(f"{type(self).__name__}: {why}", flush=True)
        return mode

    def run(self, x, grad: bool = True) -> tuple:
        """ One supervision step without the projection: ``(state, heads,
        every)``, the last readout and all of them. """
        mode = self.kernel_mode(x)
        if mode is None:
            return self.unroller(x, grad)
        return self._run_kernel(x, grad, mode)


class Net(Dispatch, recursion.PeerNet):
    """ The lattice net on a peer table, on the kernels where allowed. """
    def __init__(self, widths, peers, n_channels: int = 9, rounds: int = 4,
                 cycles: int = 6, detached: int = 0, halt: bool = False):
        super().__init__(widths, peers, n_channels, rounds, cycles, detached,
                         halt)

    def _run_kernel(self, x, grad: bool, mode: str) -> tuple:
        """ :meth:`unroll` on the segment kernels (float32 inside: a bf16
        clue from autocast is cast back before it enters). """
        state = self.initial(x)
        if state.clue.dtype != x.dtype:
            state = state._replace(clue=state.clue.to(x.dtype))
        if self.detached:
            with torch.no_grad():
                state, _ = segment(self, state, self.detached, every=False,
                                   mode=mode)
        wanted = grad and torch.is_grad_enabled()
        with torch.set_grad_enabled(wanted):
            clue = self.encoder(x)
            state = state._replace(
                clue=clue if clue.dtype == x.dtype else clue.to(x.dtype))
            window = self.cycles - self.detached
            state, answers = (
                segment_train(self, state, window, mode=mode) if wanted
                else segment(self, state, window, mode=mode))
            every = [self.heads(answers[i]) for i in range(window)]
        return state, every[-1], tuple(every)


class FactorNet(Dispatch, recursion.FactorNet):
    """ The lattice net on a factor graph, always on the PyTorch path. """
    def kernel_mode(self, t) -> None:
        """ Always the PyTorch path (``"auto"`` falls back to it, a kernel
        mode raises); the first call prints it. """
        off = self.kernel in (None, "off", False)
        if not off and self.kernel != "auto":
            raise RuntimeError(f"kernel={self.kernel!r} requested but "
                               "unsupported: a factor graph has no segment kernel")
        if not self._kernel_logged:
            self._kernel_logged = True
            why = "kernel=off" if off else "no segment kernel for a factor graph"
            print(f"{type(self).__name__}: PyTorch path: {why}", flush=True)
        return None


class Act(Net):
    """ :class:`Net` plus a halt head, run in segments. """
    #: The per-board clip as the kernel backward reads it: ``k``, the
    #: per-board losses of the step, the slow reference and the statistics.
    slot_clip: float | None = None
    slot_ref: torch.Tensor | None = None
    slot_ref_running: float | None = None
    slot_clip_stats: dict | None = None

    def __init__(self, widths, peers, n_channels: int = 9, rounds: int = 12,
                 cycles: int = 8, detached: int = 6):
        super().__init__(widths, peers, n_channels, rounds, cycles, detached,
                         halt=True)

    def prefix(self, s, mode: str | None = None):
        """ The leading ``detached`` cycles, forward only. """
        if mode is None:
            return super().prefix(s)
        with torch.no_grad():
            if self.detached:
                s, _ = segment(self, s, self.detached, every=False, mode=mode)
        return s

    def window(self, s, x, mode: str | None = None, cycle=None) -> tuple:
        """ :meth:`Recursion.window`, its cycles one segment on the kernels
        when ``mode`` is set. """
        if mode is None:
            return super().window(s, x, cycle)
        n = self.cycles - self.detached
        clue = self.encoder(x)
        if clue.dtype != s.state.dtype:
            clue = clue.to(s.state.dtype)
        s = s._replace(clue=clue)
        s, answers = (segment_train(self, s, n, mode=mode)
                      if torch.is_grad_enabled()
                      else segment(self, s, n, mode=mode))
        every = tuple(self.sm(answers[i]) for i in range(n))
        return s, every, self.halt.read(s.answer.detach())

    def act_step(self, s, x, grad: bool = True) -> tuple:
        """ :meth:`Recursion.act_step` in the mode :meth:`kernel_mode`
        picks for the carried state. """
        mode = self.kernel_mode(s.state)
        grad = grad and torch.is_grad_enabled()
        s = self.prefix(s, mode)
        with torch.set_grad_enabled(grad):
            s, every, halt = self.window(s, x, mode)
        return s, every, halt, every[-1]


class ActPool(act.ActPool):
    """ :class:`discopy.neural.act.ActPool` whose segment runs on the
    kernels when the net's :meth:`~Dispatch.kernel_mode` picks a mode. """
    def __init__(self, net: Act, stream, cfg: act.ActConfig):
        super().__init__(net, stream, cfg)
        self.mode = net.kernel_mode(self.x)

    def segment(self) -> tuple:
        """ :meth:`discopy.neural.act.ActPool.segment`; on the kernels, the
        window's backward clips every board's row of gradient relative to
        the per-board losses left on ``net.slot_ref``. """
        if self.mode is None:
            return super().segment()
        cfg, net = self.cfg, self.net
        s, every, halt = net.window(net.prefix(self.state, self.mode),
                                    self.x, self.mode)
        loss, ce, q, cell_ok, board = act.terms(
            every, halt, every[-1], self.x, self.target, self.given,
            cfg.halt_weight, cfg.halt_board_weight)
        net.slot_ref = board              # the kernel backward clips
        loss.backward()
        net.slot_ref = None
        return s, halt, loss, ce, q, cell_ok.all(-1), net.slot_clip_stats
