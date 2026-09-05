#!/usr/bin/env python3



from pathlib import Path
from shutil import copytree
import sys, os 
import shutil


# Select the destination using USE_KRALEMANN from LocalCalculationProtocol_GPU.sh.
if os.getenv("USE_KRALEMANN") == "0":
    src = Path("Results")
    dst = Path("Results_LinearInterp")
elif os.getenv("USE_KRALEMANN") == "1":
    src = Path("Results")
    dst = Path("Results_KralemannMethod")
else:
    use_k = os.getenv("USE_KRALEMANN")
    print(f"Invalid USE_KRALEMANN value: {use_k!r} (expected 0 or 1)", file=sys.stderr)
    sys.exit(1)

# Append RUNDATE to the destination name.
run_date = os.getenv("RUNDATE")
dst = Path(f"{dst}_{run_date}")


if not src.exists():
    print(f"Source directory not found: {src}", file=sys.stderr)
    sys.exit(1)

if dst.exists():
    if dst.is_dir():
        shutil.rmtree(dst)
    else:
        dst.unlink()
    copytree(src, dst)
    print(f"Copied {src} → {dst}")
    sys.exit(0)

copytree(src, dst)

print("====== (DirCopy.py) Results were archived in the following directory. ======")
print(f"Copied {src} → {dst}")
