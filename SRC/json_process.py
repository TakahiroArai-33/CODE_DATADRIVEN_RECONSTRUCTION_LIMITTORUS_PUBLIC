#!/usr/bin/env python3
import json, os, sys

"""Return 0 if the final NPZ for a JSON configuration exists, otherwise 1."""

cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = json.load(f)

label = cfg["label"]
repeat = int(cfg["REPEAT"])
final_npz = os.path.join("CoupledModelSimulationDATA",
                         f"{label}_process_{repeat - 1}.npz")
if os.path.exists(final_npz):
    sys.exit(0)
sys.exit(1)
