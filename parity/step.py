"""
One step of a parity run, in its own process: patch module constants, then
call the script's ``main(argv)``.

    python -m parity.step '{"module": ..., "argv": [...], "patches": {...}}'

A patch ``"package.module.NAME": value`` shrinks a constant the script has
no flag for (a search's rows per model call), on the imported module, so
that every caller reading it at call time sees the small value.
"""

from __future__ import annotations

import importlib
import json
import sys

import torch


def main(spec: dict) -> int:
    torch.set_num_threads(spec.get("threads", 4))
    for path, value in spec.get("patches", {}).items():
        module, name = path.rsplit(".", 1)
        setattr(importlib.import_module(module), name, value)
    return importlib.import_module(spec["module"]).main(spec["argv"]) or 0


if __name__ == "__main__":
    raise SystemExit(main(json.loads(sys.argv[1])))
