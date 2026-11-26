"""
CUDA segment kernels: a whole segment of the peer-graph recursion
(:class:`discopy.neural.cell.PeerSite` rounds and the answer refresh) for
one board per thread block, forward only (evaluation, detached prefixes) or
with its backward (the differentiated window), the backward returning one
row of parameter gradient per board for the per-board clip. Hopper (sm_90)
only; :func:`resolve` falls back to the PyTorch path elsewhere.

The nets that dispatch to them (:mod:`.dispatch`): the lattice nets
:class:`Net` and :class:`FactorNet`, the ACT net :class:`Act` and its
:class:`ActPool`, the library's with the kernel policy on top.
"""

from .autograd import segment, segment_train
from .build import resolve, supports
from .dispatch import Act, ActPool, FactorNet, Net

__all__ = ["segment", "segment_train", "resolve", "supports", "Net",
           "FactorNet", "Act", "ActPool"]
