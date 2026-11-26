"""
The parity harness: every recipe of the repository run end to end at a tiny
size on the CPU, and the bytes it writes fingerprinted.

    python -m parity capture --out parity/golden.json   # on the reference commit
    python -m parity check                              # on any later commit

A run is a list of steps, each one of the repository's own scripts with
its own flags (:mod:`parity.runs`). Every checkpoint tensor is hashed to
its bytes, every JSON the scripts write is kept with its timings dropped,
so two commits agree only if they train and test to the same bits. The
recipes are shrunk (few steps, small pools, small searches), never
changed, and run on their CPU path: fp32, eager, ``--kernel off``. The
CUDA segment kernel, the Triton round and bf16 autocast are checked nowhere
here; ``parity/test_model.py`` and ``test_training.py`` take the CUDA-only
branches of the PyTorch path on CPU tensors that say they are on CUDA, and
``test_model.py`` compares the dynamo graphs ``torch.compile`` traces.

The data directories come from ``$SUDOKU_DATA``, ``$MAZE_DATA`` and
``$QUEENS_DATA``; the CPU is forced and runs one thread, since a
reduction's bits depend on how it is split and the backward of a gather
accumulates in a different order on every run with more.
"""
