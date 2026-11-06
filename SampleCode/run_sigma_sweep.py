#!/usr/bin/env python3
"""Run Test.ipynb for several noise amplitudes in separate Python processes."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", "--sigma", nargs="+", type=float, required=True,
                        help="Noise amplitudes, e.g. --sigmas 1e-6 1e-5.")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum concurrent runs (default: 2).")
    parser.add_argument("--output-root", type=Path,
                        help="New or empty folder for this sweep; defaults to a unique output_runs subfolder.")
    parser.add_argument("--notebook", type=Path, default=HERE / "Test.ipynb")
    parser.add_argument("--t-end", type=float, help="Override final rescaled time epsilon*t.")
    parser.add_argument("--seed", type=int, help="Override the notebook seed for every run.")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in args.sigmas):
        parser.error("--sigmas must be finite and nonnegative")
    if len(set(args.sigmas)) != len(args.sigmas):
        parser.error("--sigmas must not contain duplicate values")
    if args.t_end is not None and (not math.isfinite(args.t_end) or args.t_end <= 0):
        parser.error("--t-end must be positive and finite")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if not args.notebook.is_file():
        parser.error(f"Notebook not found: {args.notebook}")
    return args


def save_manifest(root, manifest):
    temporary = root / "sweep.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "sweep.json")


def main(argv=None):
    args = parse_args(argv)
    root = args.output_root
    if root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = HERE / "output_runs" / f"{stamp}_{uuid.uuid4().hex[:6]}"
    root = root.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output root must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir()
    # Both jobs execute an identical snapshot even if the working notebook is edited later.
    source = args.notebook.read_bytes()
    snapshot = root / "Test.executed.ipynb"
    snapshot.write_bytes(source)
    manifest = {
        "status": "running", "started_at": now(),
        "notebook": str(args.notebook.resolve()),
        "notebook_snapshot": str(snapshot),
        "notebook_sha256": hashlib.sha256(source).hexdigest(),
        "python": sys.executable, "jobs": [],
    }
    for sigma in args.sigmas:
        label = f"sigma_{sigma!r}"
        output = root / label
        command = [sys.executable, "-u", str(HERE / "run_test.py"),
                   "--notebook", str(snapshot), "--sigma", repr(sigma),
                   "--output-dir", str(output)]
        if args.t_end is not None:
            command.extend(["--t-end", repr(args.t_end)])
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        manifest["jobs"].append({
            "sigma": sigma, "status": "pending", "output_dir": str(output),
            "log": str(root / "logs" / f"{label}.log"), "command": command,
        })
    save_manifest(root, manifest)
    print(f"Results: {root}", flush=True)
    print(f"Running up to {min(args.jobs, len(args.sigmas))} jobs concurrently.", flush=True)
    pending = deque(manifest["jobs"])
    active = []

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        while pending or active:
            changed = False
            while pending and len(active) < args.jobs:
                job = pending.popleft()
                env = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[name] = "1"
                env["PYTHONUNBUFFERED"] = "1"
                log = Path(job["log"]).open("w")
                log.write(shlex.join(job["command"]) + "\n")
                log.flush()
                try:
                    process = subprocess.Popen(job["command"], cwd=HERE, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                except OSError as exc:
                    log.close()
                    job.update(status="failed", error=str(exc), returncode=-1, finished_at=now())
                else:
                    job.update(status="running", pid=process.pid, started_at=now())
                    active.append((job, process, log))
                    print(f"Started sigma={job['sigma']:g}, PID={process.pid}; log: {job['log']}", flush=True)
                changed = True
            for item in active[:]:
                job, process, log = item
                code = process.poll()
                if code is not None:
                    log.close()
                    job.update(status="completed" if code == 0 else "failed",
                               returncode=code, finished_at=now())
                    active.remove(item)
                    changed = True
                    print(f"sigma={job['sigma']:g}: {job['status']} (exit {code})", flush=True)
            if changed:
                save_manifest(root, manifest)
            if pending or active:
                time.sleep(1)
    except KeyboardInterrupt:
        for job in pending:
            job.update(status="cancelled", finished_at=now())
        for job, process, log in active:
            process.terminate()
        for job, process, log in active:
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            log.close()
            job.update(status="interrupted", returncode=code, finished_at=now())
        manifest.update(status="interrupted", finished_at=now())
        save_manifest(root, manifest)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    success = all(job["status"] == "completed" for job in manifest["jobs"])
    manifest.update(status="completed" if success else "failed", finished_at=now())
    save_manifest(root, manifest)
    print(f"Sweep {manifest['status']}: {root / 'sweep.json'}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
