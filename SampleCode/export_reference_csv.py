"""Export the full dense reference arrays to lossless-precision CSV files.

Matrices are streamed with NumPy because the spreadsheet authoring runtime
has no CSV-export API. No subsampling, rounding to display precision, or
spreadsheet cell limits are applied. Run from any working directory.
"""
from pathlib import Path
import pickle

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def main():
    def write(name, values, header=""):
        path = DATA / name
        temporary = path.with_suffix(".csv.tmp")
        np.savetxt(temporary, values, delimiter=",", fmt="%.17g", header=header, comments="")
        temporary.replace(path)
        restored = np.loadtxt(path, delimiter=",", skiprows=int(bool(header)), ndmin=2)
        np.testing.assert_array_equal(restored, np.asarray(values).reshape(restored.shape))
        print(f"{name}: {path.stat().st_size / 1024**2:.2f} MiB; all values verified", flush=True)

    with np.load(DATA / "MutualCouplingFunction_8193.npz") as coupling:
        for key in ("Gamma_s", "Gamma_t"):
            write(f"MutualCouplingFunction_8193_{key}.csv", coupling[key])
        write("MutualCouplingFunction_8193_axes.csv",
              np.column_stack((coupling["delta_theta"], coupling["delta_phi"])),
              header="delta_theta,delta_phi")
    with (DATA / "X0data_withPhi0.pickle").open("rb") as stream:
        x0 = pickle.load(stream)
    for key in ("U", "V"):
        write(f"X0_{key}.csv", x0[key])
    write("X0_axes.csv", np.column_stack((x0["t"], x0["rx"])),
          header="t,rx")
    metadata_path = DATA / "X0_metadata.csv"
    metadata_path.write_text(f"name,value\nT,{float(x0['T']):.17g}\n", encoding="utf-8")
    # Bmean is the original removed offset, separate from the centered By curve.
    with (DATA / "B_theory.pickle").open("rb") as stream:
        b = pickle.load(stream)
    b_metadata = DATA / "B_theory_metadata.csv"
    b_metadata.write_text(f"name,value\nBmean,{float(b['Bmean']):.17g}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
