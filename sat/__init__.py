"""
Boolean satisfiability on the factor graph of a CNF formula: the ragged
batching and the G4SATBench generators (:mod:`.data`), the recursive
factor-graph GNN (:mod:`.model`), the residual lattice and its loss
(:mod:`.lattice`), the repair search that decodes it (:mod:`.solve`), the
segmented ACT recursion with its halt head (:mod:`.act`), and a training
and a test script per experiment -- with the lattice and the repair search
(:mod:`.train_lattice`, :mod:`.eval_lattice`), and without it: segmented
ACT, deterministic and best-of-k evaluation (:mod:`.train_act`,
:mod:`.eval_act`).
"""
