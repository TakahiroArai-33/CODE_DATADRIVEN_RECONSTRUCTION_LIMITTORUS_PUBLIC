#!/usr/bin/env python3
"""Run Test.ipynb from Python with one sigma and a separate output directory.

The notebook's code cells are executed in their original order. Only the run
settings and notebook display calls are adapted for execution in a terminal.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise amplitude; zero disables noise.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for this run; existing directories are refused.")
    parser.add_argument("--t-end", type=float,
                        help="Override the notebook's final rescaled time epsilon*t.")
    parser.add_argument("--seed", type=int, help="Override the notebook's random seed.")
    parser.add_argument("--notebook", type=Path, default=HERE / "Test.ipynb",
                        help="Notebook to execute (default: SampleCode/Test.ipynb).")
    args = parser.parse_args(argv)
    if not math.isfinite(args.sigma) or args.sigma < 0:
        parser.error("--sigma must be nonnegative and finite")
    if args.t_end is not None and (not math.isfinite(args.t_end) or args.t_end <= 0):
        parser.error("--t-end must be positive and finite")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


class TerminalCells(ast.NodeTransformer):
    """Override top-level settings and suppress inline figure display only."""

    def __init__(self, overrides):
        self.overrides = overrides
        self.found = set()

    def visit_Module(self, node):
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                names = [target.id for target in statement.targets
                         if isinstance(target, ast.Name)]
                matched = self.overrides.keys() & set(names)
                if matched:
                    if len(statement.targets) != 1 or len(matched) != 1:
                        raise ValueError("Run settings must use separate, simple assignments")
                    name = matched.pop()
                    statement.value = ast.copy_location(
                        ast.Subscript(
                            value=ast.Name(id="__run_overrides__", ctx=ast.Load()),
                            slice=ast.Constant(name), ctx=ast.Load(),
                        ), statement.value,
                    )
                    self.found.add(name)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "plt" and call.func.attr == "show"):
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def prepare_cells(notebook, raw, overrides):
    """Validate and compile all cells before starting an expensive run."""
    document = json.loads(raw)
    transformer = TerminalCells(overrides)
    cells = []
    for index, cell in enumerate(document["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        lines = []
        for line in "".join(cell.get("source", [])).splitlines(keepends=True):
            stripped = line.strip()
            if stripped == "%matplotlib inline":
                lines.append("\n")
            elif stripped.startswith(("%", "!")):
                raise ValueError(f"Unsupported notebook magic in cell {index}: {stripped}")
            else:
                lines.append(line)
        source = "".join(lines)
        filename = f"{notebook}:cell-{index}"
        tree = transformer.visit(ast.parse(source, filename=filename))
        ast.fix_missing_locations(tree)
        label = next((line.strip() for line in lines
                      if line.strip() and not line.lstrip().startswith("#")), "empty cell")
        cells.append((index, label, compile(tree, filename, "exec")))
    missing = overrides.keys() - transformer.found
    if missing:
        raise ValueError(f"Notebook is missing run-setting assignments: {sorted(missing)}")
    return cells


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_metadata(path, metadata, namespace):
    for name in ("SIGMA", "T_END", "SEED", "DT", "EPSILON", "SAVE_INTERVAL"):
        if name in namespace:
            value = namespace[name]
            metadata[name.lower()] = value.item() if hasattr(value, "item") else value
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_notebook(args):
    notebook = args.notebook.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    raw = notebook.read_bytes()
    overrides = {"TEST_DIR": HERE, "OUTPUT_DIR": output,
                 "SIGMA": args.sigma, "loaddata": False}
    if args.t_end is not None:
        overrides["T_END"] = args.t_end
    if args.seed is not None:
        overrides["SEED"] = args.seed
    cells = prepare_cells(notebook, raw, overrides)
    # The creation is atomic: two processes cannot claim the same output path.
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "run_metadata.json"
    metadata = {
        "status": "running", "started_at": _utc_now(), "sigma": args.sigma,
        "notebook": str(notebook), "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "output_dir": str(output), "python": sys.executable, "command": sys.argv,
    }
    namespace = {"__name__": "__main__", "__file__": str(notebook),
                 "__run_overrides__": overrides}
    started = time.perf_counter()
    _write_metadata(metadata_path, metadata, namespace)
    print(f"Running {notebook.name}: sigma={args.sigma:g}; output={output}", flush=True)
    try:
        cache = output / ".cache"
        os.environ["XDG_CACHE_HOME"] = str(cache)
        os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
        os.environ["NUMBA_CACHE_DIR"] = str(cache / "numba")
        import matplotlib
        matplotlib.use("Agg")
        # The notebook imports local modules by name, regardless of launch cwd.
        for directory in (HERE, notebook.parent):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        import spde_cpu
        from tqdm import tqdm
        spde_cpu.tqdm = partial(tqdm, mininterval=30.0)

        for index, label, code in cells:
            metadata["current_cell"] = index
            _write_metadata(metadata_path, metadata, namespace)
            print(f"[cell {index}] {label[:100]}", flush=True)
            exec(code, namespace)
            _write_metadata(metadata_path, metadata, namespace)
        metadata["status"] = "completed"
        print(f"Completed: {output}", flush=True)
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        metadata["error"] = repr(exc)
        metadata["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 130 if metadata["status"] == "interrupted" else 1
    finally:
        metadata["finished_at"] = _utc_now()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        _write_metadata(metadata_path, metadata, namespace)
    return 0


def main(argv=None):
    args = parse_args(argv)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return run_notebook(args)
    except Exception as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
