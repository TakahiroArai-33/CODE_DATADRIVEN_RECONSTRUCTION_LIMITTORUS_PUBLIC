#!/usr/bin/env python3
"""ノイズあり結合2系を1回計算し、その1軌道からc・B・位相を推定する。"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, help="Analyze one saved coupled trajectory instead of simulating.")
    parser.add_argument("--t-end", type=float, default=7.0, help="Final slow time epsilon*t.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--save-interval", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--delta-phi", type=float, default=90.0)
    parser.add_argument("--delta-theta", type=float, default=2.5)
    parser.add_argument("--section", type=float, default=7.0)
    parser.add_argument("--x0", type=Path, default=DATA / "X0_axes.csv")
    parser.add_argument("--b-harmonics", type=int, default=10)
    parser.add_argument("--b-trim-cycles", type=int, default=1)
    parser.add_argument("--b-func", type=Path, default=None,
                        help="Optional fixed B reference; default estimates B from this trajectory.")
    parser.add_argument("--coupling", type=Path, default=DATA / "MutualCouplingFunction_8193_axes.csv")
    parser.add_argument("--domain-length", type=float, default=250.0, help="Domain length if saved trajectory has no rx.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    return parser.parse_args(argv)


def integer_steps(duration, dt, label):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"dt and {label} must be positive and finite")
    count = int(round(duration / dt))
    if count < 1 or not np.isclose(count * dt, duration, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} must be an integer multiple of dt={dt}")
    return count


def load_x0(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        if not path.name.endswith("_axes.csv"):
            raise ValueError("X0 CSV entry point must end in _axes.csv")
        stem = path.name[:-len("_axes.csv")]
        axes = np.genfromtxt(path, delimiter=",", names=True)
        if axes.dtype.names != ("t", "rx"):
            raise ValueError("X0 axes CSV must have header t,rx")
        return {"t": axes["t"], "rx": axes["rx"],
                "U": np.loadtxt(path.with_name(stem + "_U.csv"), delimiter=","),
                "V": np.loadtxt(path.with_name(stem + "_V.csv"), delimiter=",")}
    with path.open("rb") as stream:
        return pickle.load(stream)


def initial_states_from_phases(path, phi1, theta1, phi2, theta2):
    """Build X0(phi1, theta1), X0(phi2, theta2) on the nearest periodic grid."""
    data = load_x0(path)
    rx, ts = np.asarray(data["rx"]), np.asarray(data["t"])
    u, v = np.asarray(data["U"]), np.asarray(data["V"])
    if rx.ndim != 1 or ts.ndim != 1 or rx.size < 4 or ts.size < 3:
        raise ValueError("X0 requires one-dimensional rx/t with periodic endpoints")
    if u.shape != (ts.size, rx.size) or v.shape != u.shape:
        raise ValueError("X0 U/V shape must be (len(t), len(rx))")
    if not all(np.all(np.isfinite(a)) for a in (rx, ts, u, v, phi1, theta1, phi2, theta2)):
        raise ValueError("X0 and initial phases must be finite")
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(rx) <= 0):
        raise ValueError("X0 rx/t must increase strictly")
    if not np.allclose(np.diff(rx), rx[1] - rx[0]):
        raise ValueError("X0 requires a uniform spatial grid")
    if not np.allclose(u[:, 0], u[:, -1]) or not np.allclose(v[:, 0], v[:, -1]):
        raise ValueError("X0 spatial endpoint must duplicate the first point")
    length = float(np.ptp(rx))
    phi_grid = rx[:-1] - rx[0]
    theta_grid = 2 * np.pi * (ts[:-1] - ts[0]) / np.ptp(ts)
    def nearest(grid, target, period):
        return int(np.argmin(np.abs((grid - target + period / 2) % period - period / 2)))
    indices = [(nearest(phi_grid, phi, length), nearest(theta_grid, theta, 2 * np.pi))
               for phi, theta in ((phi1, theta1), (phi2, theta2))]
    initial = np.array([[np.roll(u[it, :-1], ix), np.roll(v[it, :-1], ix)]
                        for ix, it in indices], dtype=np.float64)
    (ix1, it1), (ix2, it2) = indices
    return initial, rx, {
        "phi1": float(phi_grid[ix1]), "theta1": float(theta_grid[it1]),
        "phi2": float(phi_grid[ix2]), "theta2": float(theta_grid[it2]),
        "delta_phi": float(phi_grid[ix1] - phi_grid[ix2]),
        "delta_theta": float(theta_grid[it1] - theta_grid[it2]),
        "lateral_index": ix1 - ix2, "vertical_index": it1 - it2,
        "lateral_index1": ix1, "vertical_index1": it1,
        "lateral_index2": ix2, "vertical_index2": it2,
        "reference_period": float(np.ptp(ts)),
    }


def initial_conditions(path, delta_phi=90.0, delta_theta=2.5):
    """Start the second system at X0(0, 0), retaining the CLI convention."""
    return initial_states_from_phases(path, delta_phi, delta_theta, 0.0, 0.0)


def load_trajectory(path, domain_length=250.0):
    """Load the moments from one coupled realization without simulating again."""
    with np.load(path, allow_pickle=False) as data:
        required = {"t", "U1_bar", "U2_bar", "A1_tilde", "A2_tilde"}
        if not required.issubset(data.files):
            raise ValueError(f"Trajectory needs {sorted(required)}; found {data.files}")
        moments = {k: data[k] for k in required | {"V1_bar", "V2_bar", "final_state"} if k in data}
        rx = data["rx"] if "rx" in data else None
    if rx is not None:
        if rx.ndim != 1 or rx.size < 2 or not np.all(np.isfinite(rx)) or np.any(np.diff(rx) <= 0):
            raise ValueError("Trajectory rx must be finite and strictly increasing")
        domain_length = float(np.ptp(rx))
    if not np.isfinite(domain_length) or domain_length <= 0:
        raise ValueError("domain_length must be finite and positive")
    return moments, rx, domain_length


def b_estimation_data(phase):
    return {key: value for key, value in phase.items()
            if key.startswith(("Bx", "By", "b_", "c_fit_variance"))
            or key in ("c1", "c2", "section", "domain_length")}


def b_estimation_summary(phase):
    if phase["b_method"] != "estimated":
        return {"method": "reference", "source": str(phase["b_source"])}
    return {"method": "estimated", "harmonics": int(phase["b_harmonics"]),
            "trim_cycles_per_end": int(phase["b_trim_cycles"]),
            "systems": {str(i): {"c": float(phase[f"c{i}"]),
                                 "c_fit_variance": float(phase[f"c_fit_variance{i}"]),
                                 "fit_rmse": float(phase[f"b_fit_rmse{i}"]),
                                 "fit_cycles": int(phase[f"b_fit_cycles{i}"])} for i in (1, 2)}}


def b_estimation_figure(phase):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for i, ax in enumerate(axes, start=1):
        theta, samples = phase[f"b_theta{i}"], phase[f"b_samples{i}"]
        stride = max(1, int(np.ceil(len(theta) / 5000)))
        ax.scatter(theta[::stride], samples[::stride], s=4, alpha=0.2,
                   color="tab:gray", label="Centered cycle samples")
        ax.plot(phase[f"Bx{i}"], phase[f"By{i}"], label="Estimated B", lw=1.8)
        ax.set(xlabel=r"$\Theta$", ylabel=r"$\widehat B(\Theta)$", xlim=(0, 2 * np.pi),
               title=f"System {i}: c={phase[f'c{i}']:.6g}, {phase[f'b_fit_cycles{i}']} cycles")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return fig


def comparison_figure(phase, theory, epsilon, length):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    elapsed = phase["t"] - phase["t"][0]
    x = epsilon * elapsed if epsilon > 0 else elapsed
    xlabel = r"$\epsilon(t-t_0)$" if epsilon > 0 else r"$t-t_0$"
    for i, (key, period, scale, label) in enumerate((
        ("delta_phi", length, length / 2, r"$\Delta\Phi/L$"),
        ("delta_theta", 2 * np.pi, np.pi, r"$\Delta\Theta/\pi$"),
    )):
        observed = phase[key]
        predicted = (theory[i] + period / 2) % period - period / 2
        for series, name, style in ((observed, "SPDE phase", "-"), (predicted, "Averaged theory", "--")):
            y = series / scale
            jumps = np.flatnonzero(np.abs(np.diff(y)) > 1) + 1
            axes[i, 0].plot(np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan), style, label=name)
        error = (observed - predicted + period / 2) % period - period / 2
        axes[i, 1].plot(x, error / scale, color="tab:green")
        axes[i, 0].set_ylabel(label)
        axes[i, 1].set_ylabel(label + " error")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    fig.suptitle("One noisy trajectory of two coupled Gray–Scott systems")
    return fig


def save_results(phase, theory, epsilon, length, output, parameters, simulation_seconds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors = {}
    for i, (key, period, scale) in enumerate((("delta_phi", length, length / 2), ("delta_theta", 2 * np.pi, np.pi))):
        error = (phase[key] - theory[i] + period / 2) % period - period / 2
        phase[key + "_error"] = error
        errors[key] = {"rmse": float(np.sqrt(np.mean(error**2))), "max_abs": float(np.max(np.abs(error))),
                       "normalized_rmse": float(np.sqrt(np.mean(error**2)) / scale)}
    if phase["b_method"] == "estimated":
        np.savez_compressed(output / "B_estimated.npz", **b_estimation_data(phase))
    np.savez_compressed(output / "phase_comparison.npz", **phase, theory_delta_phi=theory[0],
                        theory_delta_theta=theory[1], eps_t=epsilon * phase["t"])
    np.savetxt(output / "phase_comparison.csv",
               np.column_stack([phase["t"], epsilon * phase["t"], phase["delta_phi"], phase["delta_theta"],
                                theory[0], theory[1], phase["delta_phi_error"], phase["delta_theta_error"]]),
               delimiter=",", header="t,eps_t,delta_phi,delta_theta,theory_delta_phi,theory_delta_theta,error_phi,error_theta", comments="")
    summary = {"parameters": dict(parameters),
               "samples": int(phase["t"].size), "simulation_seconds": simulation_seconds,
               "comparison_interval": [float(phase["t"][0]), float(phase["t"][-1])],
               "b_estimation": b_estimation_summary(phase), "errors": errors,
               "comparison_note": "One realization of two coupled noisy systems; deterministic averaged difference ODE "
                                  "initialized at first measured phase pair. Not a pathwise stochastic prediction."}
    return summary


def main(argv=None):
    from phase_analysis import calculate_phases, solve_theory
    args = parse_args(argv)
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("epsilon must be nonnegative and finite")
    if args.trajectory is None and args.epsilon == 0:
        raise ValueError("epsilon must be positive when t-end specifies epsilon*t")
    if not args.coupling.is_file():
        raise FileNotFoundError(args.coupling)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulation_seconds = None
    if args.trajectory is None:
        from spde_cpu import simulate
        initial, rx, init_info = initial_conditions(args.x0, args.delta_phi, args.delta_theta)
        length = float(np.ptp(rx))
        physical_t_end = args.t_end / args.epsilon
        n_steps = integer_steps(physical_t_end, args.dt, "t-end / epsilon")
        stride = integer_steps(args.save_interval, args.dt, "save-interval")
        print(f"One coupled realization, shape={initial.shape}; {n_steps:,} CPU/Numba steps; sigma={args.sigma:g}", flush=True)
        started = time.perf_counter()
        moments = simulate(initial, rx, args.dt, n_steps, stride, epsilon=args.epsilon,
                           sigma=args.sigma, seed=args.seed)
        simulation_seconds = time.perf_counter() - started
        np.savez_compressed(args.output_dir / "trajectory.npz", **moments, rx=rx,
                            dt=args.dt, epsilon=args.epsilon, sigma=args.sigma, seed=args.seed)
        print(f"Simulation: {simulation_seconds:.2f} s; initial {init_info}", flush=True)
    else:
        moments, rx, length = load_trajectory(args.trajectory, args.domain_length)
        print(f"Analyzing one saved coupled trajectory: {args.trajectory}", flush=True)
    args.physical_t_end = float(moments["t"][-1])
    if args.b_func is not None and args.b_func.name in ("B_theory.csv", "B_theory.pickle") and not np.isclose(args.section, 7.0):
        raise ValueError("The copied B_theory reference is calibrated at section=7")
    phase = calculate_phases(moments, args.b_func, section=args.section, domain_length=length,
                             b_harmonics=args.b_harmonics, b_trim_cycles=args.b_trim_cycles)
    theory = solve_theory(phase["t"], [phase["delta_phi"][0], phase["delta_theta"][0]], args.coupling, args.epsilon, length)
    parameters = vars(args).copy()
    if args.trajectory is None:
        parameters["method"] = "euler"
    summary = save_results(phase, theory, args.epsilon, length, args.output_dir, parameters, simulation_seconds)
    cache = args.output_dir / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((cache / "matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = [("comparison.png", comparison_figure(phase, theory, args.epsilon, length))]
    if phase["b_method"] == "estimated":
        figures.append(("B_estimation.png", b_estimation_figure(phase)))
    for filename, fig in figures:
        fig.savefig(args.output_dir / filename, dpi=180)
        plt.close(fig)
    print("Estimates:", summary["b_estimation"])
    print("Errors:", summary["errors"])
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
