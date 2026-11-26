"""
``python -m parity capture --out FILE [RUN ...]`` fingerprints the runs on
this commit; ``python -m parity check [--golden FILE] [RUN ...]`` reruns
them and lists every difference from the golden file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import fingerprint
from .runs import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THREADS = 1


def environment() -> dict:
    """
    The CPU only, on one thread, the data directories required: on more,
    the backward of the peer gather (``index_put_`` with accumulation) adds
    in an order that changes from run to run, in the original code too.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0",
               OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
    missing = [key for key in ("SUDOKU_DATA", "MAZE_DATA", "QUEENS_DATA")
               if key not in env]
    if missing:
        raise SystemExit(f"set {', '.join(missing)}: the parity runs read "
                         "the built datasets, they do not download them")
    return env


def data_paths() -> dict:
    """ Each data directory as the scripts write it, by its variable:
    ``str(Path($QUEENS_DATA))`` is ``<QUEENS_DATA>``. """
    env = environment()
    return {str(Path(env[key])): f"<{key}>"
            for key in ("SUDOKU_DATA", "MAZE_DATA", "QUEENS_DATA")}


def run(name: str, log) -> dict:
    """ Every step of ``RUNS[name]`` in a fresh directory, fingerprinted. """
    with tempfile.TemporaryDirectory(prefix=f"parity-{name}-") as tmp:
        out = Path(tmp)
        for step in RUNS[name]:
            spec = {"module": step.module, "threads": THREADS,
                    "argv": [arg.format(out=out) for arg in step.argv],
                    "patches": step.patches}
            tick = time.perf_counter()
            done = subprocess.run(
                [sys.executable, "-m", "parity.step", json.dumps(spec)],
                cwd=ROOT, env=environment(), capture_output=True, text=True)
            log(f"  {step.module} {' '.join(spec['argv'])}: "
                f"{time.perf_counter() - tick:.0f}s")
            if done.returncode:
                raise RuntimeError(f"{name}: {step.module} failed\n"
                                   f"{done.stdout[-4000:]}{done.stderr[-4000:]}")
        return fingerprint.directory(out, data_paths())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "check"))
    p.add_argument("runs", nargs="*", default=None,
                   help=f"a subset of {', '.join(RUNS)}")
    p.add_argument("--out", default=str(HERE / "golden.json"),
                   help="the file capture writes")
    p.add_argument("--golden", default=str(HERE / "golden.json"),
                   help="the file check compares against")
    args = p.parse_args(argv)
    names = args.runs or list(RUNS)

    def log(line):
        print(line, flush=True)

    if args.mode == "capture":
        path = Path(args.out)
        found = json.loads(path.read_text()) if path.exists() else {}
        for name in names:
            log(f"{name}:")
            found[name] = run(name, log)
            path.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
        log(f"wrote {path}")
        return 0
    golden = json.loads(Path(args.golden).read_text())
    failed = 0
    for name in names:
        log(f"{name}:")
        diffs = fingerprint.differences(golden[name], run(name, log))
        failed += bool(diffs)
        log("  same bits" if not diffs else "\n".join(
            f"  DIFFERENT {diff}" for diff in diffs[:40]))
    log(f"{len(names) - failed}/{len(names)} runs reproduce the golden bits")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
