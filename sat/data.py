"""
The SAT datasets and the ragged CNF batch.

    export SAT_DATA=data            # the G4SATBench splits, written below it
    python -m sat.data --difficulty easy

builds the two random families of G4SATBench (Li et al., 2024) at the
sizes its paper uses, with its own generators ported here (the
``python-sat`` solver is the only extra dependency): ``sr``, whose clauses
are appended until a solver reports unsatisfiability, one flipped literal
of the last clause making the formula satisfiable again, and ``3-sat``,
drawn at the empirical critical ratio and kept when a solver finds it
satisfiable. Difficulty is the number of variables, drawn uniformly per
instance: ``[10, 40]`` easy, ``[40, 200]`` medium, ``[200, 400]`` (sr) or
``[200, 300]`` (3-sat) hard. Easy and medium hold 80,000 train, 10,000
valid and 10,000 test formulas; hard is test-only. Only the satisfiable
instances are written -- an unsatisfiable formula has no assignment to
predict, and both recipes are label-free, so no solution file is needed.

The data directory is ``--data``, else ``$SAT_DATA``, else
``data/g4satbench`` at the repository root, and a split is
``<root>/<difficulty>/<family>/<split>/sat`` -- G4SATBench's own layout,
so ``--data`` also accepts a tree its generators wrote. The trainers and
the evaluators build a missing split on first use.

Generation is the slow part of the easy experiment and a long job for the
others: the solver loop runs once per clause, so a formula of 200 to 400
variables costs thousands of incremental solver calls. Expect minutes for
an easy split, hours for medium (80,000 formulas of up to 200 variables)
and tens of GB on disk for both.

A batch is edge-indexed and flat (:class:`Batch`): instances of different
sizes concatenate into one graph, variables and clauses carry the index of
the instance they belong to, and every per-instance quantity is a
segmented reduction over those indices. Nothing in the model depends on a
variable count or a clause length, so one net runs on any formula.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

#: The two families, as G4SATBench names their directories.
FAMILIES = ("sr", "3-sat")
#: Variables per instance, per difficulty and family.
SIZES = {("sr", "easy"): (10, 40), ("sr", "medium"): (40, 200),
         ("sr", "hard"): (200, 400),
         ("3-sat", "easy"): (10, 40), ("3-sat", "medium"): (40, 200),
         ("3-sat", "hard"): (200, 300)}
#: Instances per split; hard is test-only.
COUNTS = {"easy": {"train": 80000, "valid": 10000, "test": 10000},
          "medium": {"train": 80000, "valid": 10000, "test": 10000},
          "hard": {"test": 10000}}
#: ``sr``'s clause-length distribution: ``1 + Geometric(p_geo)`` with
#: probability ``p_k_2``, else ``2 + Geometric(p_geo)``.
P_K_2, P_GEO = 0.3, 0.4


def family_of(name: str) -> str:
    """ ``"3sat"``, ``"threesat"`` and ``"3-sat"`` all name the 3-SAT family. """
    got = {"sr": "sr", "3sat": "3-sat", "3-sat": "3-sat",
           "threesat": "3-sat"}.get(name.lower())
    if got is None:
        raise ValueError(f"family must be one of {FAMILIES}, not {name!r}")
    return got


# -- the ragged batch --------------------------------------------------------

class Batch(NamedTuple):
    """ A batch of CNF instances as flat, edge-indexed tensors. """
    n_vars: int                #: V, total variables across the batch
    n_clauses: int             #: C, total clauses across the batch
    edge_var: torch.Tensor     #: (E,) long, the flat variable of each edge
    edge_clause: torch.Tensor  #: (E,) long, the flat clause of each edge
    edge_sign: torch.Tensor    #: (E,) long, 1 a positive literal, 0 negative
    v_batch: torch.Tensor      #: (V,) long, the instance of each variable
    c_batch: torch.Tensor      #: (C,) long, the instance of each clause
    n_instances: int


class Instance:
    """
    One formula packed into two int32 arrays: ``lits[j] = 2 * variable +
    sign`` in clause order, and ``offsets``, with clause ``i`` holding
    ``lits[offsets[i]:offsets[i + 1]]``. A pool of 80,000 medium formulas
    is resident for the whole run, so the packing matters: it costs 4
    bytes a literal against about 70 for the nested Python lists the
    DIMACS parser produces.
    """
    __slots__ = ("n_vars", "lits", "offsets")

    def __init__(self, n_vars: int, clauses):
        lengths = np.fromiter((len(c) for c in clauses), dtype=np.int64,
                              count=len(clauses))
        self.n_vars = int(n_vars)
        self.offsets = np.zeros(len(clauses) + 1, dtype=np.int32)
        np.cumsum(lengths, out=self.offsets[1:])
        self.lits = np.fromiter(
            (2 * v + sign for c in clauses for v, sign in c),
            dtype=np.int32, count=int(lengths.sum()))

    @property
    def n_clauses(self) -> int:
        return len(self.offsets) - 1

    @property
    def clauses(self) -> list:
        """ The clauses as lists of ``(variable, sign)``. """
        lits, offsets = self.lits.tolist(), self.offsets.tolist()
        return [[(lit >> 1, lit & 1) for lit in lits[offsets[i]:offsets[i + 1]]]
                for i in range(len(offsets) - 1)]


def collate(instances) -> Batch:
    """ A :class:`Batch` of :class:`Instance`s, concatenated in order. """
    n_vars = np.fromiter((i.n_vars for i in instances), dtype=np.int64,
                         count=len(instances))
    n_clauses = np.fromiter((i.n_clauses for i in instances), dtype=np.int64,
                            count=len(instances))
    n_lits = np.fromiter((len(i.lits) for i in instances), dtype=np.int64,
                         count=len(instances))
    var_offset = np.concatenate([[0], np.cumsum(n_vars)])
    clause_offset = np.concatenate([[0], np.cumsum(n_clauses)])
    lits = np.concatenate([i.lits for i in instances]).astype(np.int64)
    local_clause = np.concatenate(
        [np.repeat(np.arange(i.n_clauses), np.diff(i.offsets))
         for i in instances]).astype(np.int64)
    edge_clause = local_clause + np.repeat(clause_offset[:-1], n_lits)
    return Batch(
        n_vars=int(var_offset[-1]), n_clauses=int(clause_offset[-1]),
        edge_var=torch.from_numpy((lits >> 1)
                                  + np.repeat(var_offset[:-1], n_lits)),
        edge_clause=torch.from_numpy(edge_clause),
        edge_sign=torch.from_numpy(lits & 1),
        v_batch=torch.from_numpy(np.repeat(np.arange(len(instances)), n_vars)),
        c_batch=torch.from_numpy(
            np.repeat(np.arange(len(instances)), n_clauses)),
        n_instances=len(instances))


def to_device(batch: Batch, device) -> Batch:
    """ A :class:`Batch` with its tensors on ``device``. """
    return batch._replace(
        edge_var=batch.edge_var.to(device),
        edge_clause=batch.edge_clause.to(device),
        edge_sign=batch.edge_sign.to(device),
        v_batch=batch.v_batch.to(device),
        c_batch=batch.c_batch.to(device))


def sample(pool, n_instances: int, generator=None) -> Batch:
    """ A :class:`Batch` of ``n_instances`` formulas drawn from ``pool``
    with replacement. """
    index = torch.randint(0, len(pool), (n_instances,), generator=generator)
    return collate([pool[i] for i in index.tolist()])


def tile(batch: Batch, copies: int, device=None) -> Batch:
    """
    ``copies`` disjoint copies of ``batch`` in one :class:`Batch`,
    copy-major: copy ``c`` owns variables ``c * V ..`` and instances ``c *
    I ..``. The chains of the lattice search and the rollouts of the ACT
    recipe are both this -- one wide forward instead of ``copies`` narrow
    ones -- and the wiring is built once, since only what the copies carry
    differs.
    """
    index = torch.arange(copies, device=device).view(-1, 1)

    def spread(values, stride):
        return (index * stride + values.to(device).view(1, -1)).reshape(-1)

    return Batch(
        n_vars=copies * batch.n_vars, n_clauses=copies * batch.n_clauses,
        edge_var=spread(batch.edge_var, batch.n_vars),
        edge_clause=spread(batch.edge_clause, batch.n_clauses),
        edge_sign=batch.edge_sign.to(device).repeat(copies),
        v_batch=spread(batch.v_batch, batch.n_instances),
        c_batch=spread(batch.c_batch, batch.n_instances),
        n_instances=copies * batch.n_instances)


# -- DIMACS ------------------------------------------------------------------

def parse_dimacs(path) -> Instance:
    """
    One DIMACS ``.cnf`` file as an :class:`Instance`: variables 0-based,
    ``sign == 1`` a positive literal. Comments and the ``p`` header are
    skipped and a clause runs to its ``0`` terminator, so a clause split
    over several lines reads correctly.
    """
    n_vars, tokens = 0, []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line[0] in "c%":
                continue
            if line[0] == "p":
                n_vars = int(line.split()[2])
                continue
            tokens.extend(int(token) for token in line.split())
    clauses, clause = [], []
    for literal in tokens:
        if literal:
            clause.append((abs(literal) - 1, 1 if literal > 0 else 0))
        elif clause:
            clauses.append(clause)
            clause = []
    if clause:
        clauses.append(clause)
    return Instance(n_vars, clauses)


def write_dimacs(path, n_vars: int, clauses) -> None:
    """ ``clauses`` as DIMACS literals (``[3, -1, 7]``) at ``path``. """
    with open(path, "w") as handle:
        handle.write("p cnf %d %d\n" % (n_vars, len(clauses)))
        for clause in clauses:
            handle.write(" ".join(str(literal) for literal in clause) + " 0\n")


# -- the generators ----------------------------------------------------------

def connected(n_vars: int, clauses) -> bool:
    """
    Is the variable-incidence graph connected -- the acceptance test both
    G4SATBench generators apply. Union-find over each clause's variables,
    which is the connectivity its ``networkx`` clique does.
    """
    parent = list(range(n_vars))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for clause in clauses:
        first = find(abs(clause[0]) - 1)
        for literal in clause[1:]:
            root = find(abs(literal) - 1)
            if root != first:
                parent[root] = first
    return len({find(v) for v in range(n_vars)}) == 1


def fingerprint(clauses) -> frozenset:
    """ A formula's identity up to the order of its clauses and literals,
    for the generators' duplicate check. """
    return frozenset(frozenset(clause) for clause in clauses)


def deduplicate(clauses) -> list:
    """ The clauses in order, dropping a repeat of one already present. """
    seen, out = set(), []
    for clause in clauses:
        key = frozenset(clause)
        if key not in seen:
            seen.add(key)
            out.append(clause)
    return out


def solver():
    """ A fresh CaDiCaL, or the install hint if ``python-sat`` is missing. """
    try:
        from pysat.solvers import Cadical153
    except ImportError as error:        # pragma: no cover - install hint
        raise SystemExit(
            "generating the SAT datasets needs a solver: pip install "
            "python-sat (or point --data at a tree G4SATBench's own "
            "generators wrote)") from error
    return Cadical153()


def generate_sr(rng, min_n: int, max_n: int, seen: set) -> tuple:
    """
    One satisfiable ``sr`` formula, G4SATBench's construction: clauses of
    ``1 or 2 + Geometric(0.4)`` distinct variables with random polarities
    are added until the solver turns the formula unsatisfiable, and the
    first literal of that last clause is flipped, which makes it
    satisfiable again and leaves it one clause away from unsatisfiable.
    Rejected and redrawn while its incidence graph is disconnected or the
    formula is one ``seen`` before.
    """
    while True:
        n_vars = int(rng.integers(min_n, max_n + 1))
        engine, clauses, last = solver(), [], None
        try:
            while True:
                size = (1 if rng.random() < P_K_2 else 2) + rng.geometric(P_GEO)
                variables = rng.choice(n_vars, size=min(n_vars, int(size)),
                                       replace=False)
                last = [int(v + 1) if rng.random() < 0.5 else int(-(v + 1))
                        for v in variables]
                engine.add_clause(last)
                if not engine.solve():
                    break
                clauses.append(last)
        finally:
            engine.delete()
        clauses.append(last)                      # the unsatisfiable formula
        if not connected(n_vars, clauses):
            continue
        clauses = deduplicate(clauses)
        key = fingerprint(clauses)
        if key in seen:
            continue
        seen.add(key)
        clauses[-1] = [-last[0]] + last[1:]       # satisfiable again
        return n_vars, clauses


def generate_3sat(rng, min_n: int, max_n: int, seen: set) -> tuple:
    """
    One satisfiable ``3-sat`` formula at the threshold: ``m = 4.258 n +
    58.26 n ** (-2/3)`` distinct clauses of three distinct variables with
    random polarities, G4SATBench's empirical critical ratio. Redrawn
    while the incidence graph is disconnected, the formula was ``seen``
    before, or the solver reports it unsatisfiable.
    """
    while True:
        n_vars = int(rng.integers(min_n, max_n + 1))
        n_clauses = int(4.258 * n_vars + 58.26 * n_vars ** (-2 / 3.))
        clauses, drawn = [], set()
        while len(clauses) < n_clauses:
            variables = rng.choice(n_vars, size=3, replace=False)
            clause = [int(v + 1) if rng.random() < 0.5 else int(-(v + 1))
                      for v in variables]
            key = frozenset(clause)
            if key not in drawn:
                drawn.add(key)
                clauses.append(clause)
        if not connected(n_vars, clauses):
            continue
        key = fingerprint(clauses)
        if key in seen:
            continue
        engine = solver()
        try:
            engine.append_formula(clauses)
            satisfiable = engine.solve()
        finally:
            engine.delete()
        if satisfiable:
            seen.add(key)
            return n_vars, clauses


GENERATORS = {"sr": generate_sr, "3-sat": generate_3sat}


# -- building and loading ----------------------------------------------------

def root(directory=None) -> Path:
    """ ``directory``, else ``$SAT_DATA``, else ``data/g4satbench`` at the
    repository root. """
    if directory or os.environ.get("SAT_DATA"):
        return Path(directory or os.environ["SAT_DATA"])
    return Path(__file__).resolve().parents[1] / "data" / "g4satbench"


def split_dir(family: str, difficulty: str, split: str,
              directory=None) -> Path:
    """ The satisfiable half of one split, in G4SATBench's own layout. """
    return (root(directory) / difficulty / family_of(family) / split / "sat")


def build(family: str, difficulty: str, directory=None, counts=None,
          seed: int = 0, log=print) -> Path:
    """
    Write every split of ``family`` at ``difficulty``, plus the
    ``manifest.json`` whose presence marks the dataset complete. The
    generator is driven by one ``default_rng(seed)``, shared across the
    splits in ``train, valid, test`` order, and the duplicate check spans
    them, so no validation or test formula repeats a training one.
    """
    family = family_of(family)
    counts = COUNTS[difficulty] if counts is None else counts
    min_n, max_n = SIZES[family, difficulty]
    generate = GENERATORS[family]
    rng = np.random.default_rng(seed)
    seen: set = set()
    base = root(directory) / difficulty / family
    log(f"building {base}: {family} {difficulty}, {min_n}-{max_n} variables, "
        + ", ".join(f"{n} {name}" for name, n in counts.items()))
    for split, total in counts.items():
        out = base / split / "sat"
        out.mkdir(parents=True, exist_ok=True)
        for index in range(total):
            path = out / ("%.5d.cnf" % index)
            if not path.exists():
                write_dimacs(path, *generate(rng, min_n, max_n, seen))
            if (index + 1) % 1000 == 0 or index + 1 == total:
                log(f"    {split}: {index + 1}/{total}")
    manifest = {"family": family, "difficulty": difficulty, "seed": seed,
                "min_n": min_n, "max_n": max_n, "counts": dict(counts)}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return base


def load(family: str, difficulty: str, split: str, directory=None,
         n: int | None = None, seed: int = 0, log=print) -> list:
    """
    The pool of :class:`Instance`s of one split, building the dataset
    first if it is missing. ``n`` keeps a ``seed``-seeded random subset of
    that many formulas, chosen before parsing so the rest are never read.
    """
    where = split_dir(family, difficulty, split, directory)
    files = sorted(glob.glob(str(where / "*.cnf")))
    if not files:
        build(family, difficulty, directory, log=log)
        files = sorted(glob.glob(str(where / "*.cnf")))
    if not files:                       # pragma: no cover - unreachable build
        raise FileNotFoundError(f"no .cnf files under {where}")
    if n is not None and n < len(files):
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(files), generator=generator)[:n].tolist()
        files = [files[i] for i in sorted(keep)]
    return [parse_dimacs(path) for path in files]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the G4SATBench sr and 3-sat splits.")
    p.add_argument("--data", default=None,
                   help="data directory (default: $SAT_DATA or "
                        "data/g4satbench)")
    p.add_argument("--family", nargs="+", default=list(FAMILIES),
                   help="sr, 3sat (default: both)")
    p.add_argument("--difficulty", nargs="+", default=["easy"],
                   choices=sorted(COUNTS), help="default: easy")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the generator (default: 0)")
    args = p.parse_args(argv)
    for difficulty in args.difficulty:
        for family in args.family:
            base = root(args.data) / difficulty / family_of(family)
            if (base / "manifest.json").exists():
                print(f"{base}: exists", flush=True)
                continue
            build(family, difficulty, args.data, seed=args.seed,
                  log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
