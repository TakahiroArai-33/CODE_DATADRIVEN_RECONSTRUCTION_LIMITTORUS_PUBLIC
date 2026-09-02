#!/usr/bin/env python3
# coding: utf-8

"""Inspect inferred phase equations.

The script saves only posterior-mean colormaps of the nonconstant terms. It
prints constant terms, errors against reference values, and selected Fourier
coefficients to standard output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RectBivariateSpline


if "ipykernel" in sys.modules:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
        ip.run_line_magic("config", "InlineBackend.figure_formats = {'jpeg', 'retina'}")


plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 11
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Figure_OUTPUT"
PHASE_EQUATION_DIR = ROOT / "ReferenceData" / "PhaseEquation"
PHASE_SUMMARY_FILE = ROOT / "ReferenceData" / "phase_dot_summary.csv"
TRUE_COUPLING_FILE = PHASE_EQUATION_DIR / "CouplingFunction_8193.npz"

L = 125.0

# Fourier modes reported for comparison with reference values.
# (m_s, m_t) correspond to the Delta Phi and Delta Theta directions.
FOURIER_MODES_TO_PRINT = (
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
)

BAYES_FILES = (
    ROOT / "Results/BayesianInference/bayes_eps1e-06_sig1e-06.npz",
)

EPS_PATTERN = re.compile(r"(?:eps|epsilon)[=_-]?([-+0-9.eE]+)", re.IGNORECASE)
SIG_PATTERN = re.compile(r"(?:sig|sigma)[=_-]?([-+0-9.eE]+)", re.IGNORECASE)


def parse_eps_sigma(path: Path) -> tuple[float, float]:
    """Parse epsilon and sigma from a Bayesian-output filename."""
    eps_match = EPS_PATTERN.search(path.stem)
    sig_match = SIG_PATTERN.search(path.stem)
    if eps_match is None or sig_match is None:
        raise ValueError(f"Could not parse epsilon and sigma from filename: {path}")
    return float(eps_match.group(1)), float(sig_match.group(1))


def nonconstant_file(path: Path) -> Path:
    return path.with_name(f"{path.stem}_nonconst{path.suffix}")


def load_bayes_data(path: Path) -> dict[str, np.ndarray]:
    """Load the full-phase-equation Bayesian output."""
    with np.load(path) as data:
        result = {
            "param_a1": np.asarray(data["param_a1"], dtype=np.complex128).copy(),
            "param_a2": np.asarray(data["param_a2"], dtype=np.complex128).copy(),
            "m_profile": np.asarray(data["m_profile"], dtype=int).copy(),
        }
        if "error1" in data.files:
            result["error1"] = np.asarray(data["error1"]).copy()
        if "error2" in data.files:
            result["error2"] = np.asarray(data["error2"]).copy()
    return result


def load_nonconstant_means(path: Path) -> dict[str, np.ndarray]:
    """Load posterior means for the nonconstant terms."""
    with np.load(path) as data:
        return {
            "x": np.asarray(data["Delta_phi"]).copy(),
            "y": np.asarray(data["Delta_theta"]).copy(),
            "Phi1": np.asarray(data["ZZ1_s_mean"]).copy(),
            "Phi2": np.asarray(data["ZZ2_s_mean"]).copy(),
            "Theta1": np.asarray(data["ZZ1_t_mean"]).copy(),
            "Theta2": np.asarray(data["ZZ2_t_mean"]).copy(),
        }


def _center_periodic_axis(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a periodic grid from [0, period] to [-period/2, period/2]."""
    values_without_endpoint = np.asarray(values[:-1], dtype=float)
    centered = np.where(
        values_without_endpoint >= period / 2.0,
        values_without_endpoint - period,
        values_without_endpoint,
    )
    order = np.argsort(centered)
    sorted_values = centered[order]
    extended_values = np.concatenate((sorted_values, [sorted_values[0] + period]))
    return order, extended_values


def _periodic_mean(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    integral_x = np.trapz(z, x=x, axis=1)
    integral_xy = np.trapz(integral_x, x=y)
    return float(integral_xy / (np.ptp(x) * np.ptp(y)))


def load_true_nonconstant(
    epsilon: float,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate reference coupling functions and remove constant terms."""
    with np.load(TRUE_COUPLING_FILE) as data:
        delta_phi = np.asarray(data["delta_phi"], dtype=float)
        delta_theta = np.asarray(data["delta_theta"], dtype=float)
        gamma_s = epsilon * np.asarray(data["Gamma_s"], dtype=float)
        gamma_t = epsilon * np.asarray(data["Gamma_t"], dtype=float)

    phi_order, phi_grid = _center_periodic_axis(delta_phi, np.ptp(delta_phi))
    theta_order, theta_grid = _center_periodic_axis(delta_theta, np.ptp(delta_theta))

    def arrange_and_interpolate(gamma: np.ndarray) -> np.ndarray:
        arranged = gamma[:-1, :-1][theta_order][:, phi_order]
        arranged = np.concatenate((arranged, arranged[:, :1]), axis=1)
        arranged = np.concatenate((arranged, arranged[:1, :]), axis=0)
        interpolator = RectBivariateSpline(
            theta_grid,
            phi_grid,
            arranged,
            kx=1,
            ky=1,
        )
        interpolated = interpolator(target_y, target_x, grid=True)
        return interpolated - _periodic_mean(interpolated, target_x, target_y)

    return {
        "Phi": arrange_and_interpolate(gamma_s),
        "Theta": arrange_and_interpolate(gamma_t),
    }


def load_true_fourier_coefficients(
    epsilon: float,
) -> tuple[dict[tuple[int, int], complex], dict[tuple[int, int], complex]]:
    """Load reference Fourier coefficients indexed by (m_s, m_t)."""
    index_mt = np.loadtxt(PHASE_EQUATION_DIR / "index_m.csv", delimiter=",", dtype=int)
    index_ms = np.loadtxt(PHASE_EQUATION_DIR / "index_n.csv", delimiter=",", dtype=int)
    coef_s = np.loadtxt(PHASE_EQUATION_DIR / "coef_s_real.csv", delimiter=",")
    coef_s = coef_s + 1j * np.loadtxt(
        PHASE_EQUATION_DIR / "coef_s_imag.csv", delimiter=","
    )
    coef_t = np.loadtxt(PHASE_EQUATION_DIR / "coef_t_real.csv", delimiter=",")
    coef_t = coef_t + 1j * np.loadtxt(
        PHASE_EQUATION_DIR / "coef_t_imag.csv", delimiter=","
    )

    true_s: dict[tuple[int, int], complex] = {}
    true_t: dict[tuple[int, int], complex] = {}
    for ms, mt, value_s, value_t in zip(
        index_ms.ravel(),
        index_mt.ravel(),
        coef_s.ravel(),
        coef_t.ravel(),
    ):
        mode = (int(ms), int(mt))
        true_s[mode] = complex(epsilon * value_s)
        true_t[mode] = complex(epsilon * value_t)
    return true_s, true_t


def _format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.8e}{value.imag:+.8e}j"


def _constant_estimates(bayes: dict[str, np.ndarray]) -> dict[str, complex]:
    m_profile = bayes["m_profile"]
    index_by_mode = {
        (int(ms), int(mt)): index
        for index, (ms, mt) in enumerate(m_profile.tolist())
    }
    zero_index = index_by_mode[(0, 0)]
    block_size = len(m_profile)
    a1 = bayes["param_a1"]
    a2 = bayes["param_a2"]
    return {
        "Phi1": complex(a1[zero_index]),
        "Phi2": complex(a2[zero_index]),
        "Theta1": complex(a1[zero_index + block_size]),
        "Theta2": complex(a2[zero_index + block_size]),
    }


def _measured_constant_references(
    sigma: float,
    coupling_s_00: complex,
    coupling_t_00: complex,
) -> dict[str, float] | None:
    data = np.atleast_1d(
        np.genfromtxt(PHASE_SUMMARY_FILE, delimiter=",", names=True, encoding="utf-8")
    )
    mask = np.isclose(np.asarray(data["sigma"], dtype=float), sigma, rtol=1.0e-10, atol=0.0)
    if "eps" in data.dtype.names:
        mask &= np.isclose(np.asarray(data["eps"], dtype=float), 0.0, rtol=0.0, atol=0.0)
    selected = data[mask]
    if selected.size == 0:
        return None

    def mean(field: str) -> float:
        return float(np.mean(np.asarray(selected[field], dtype=float)))

    return {
        "Phi1": mean("mean_dot_Phi1") + coupling_s_00.real,
        "Phi2": mean("mean_dot_Phi2") + coupling_s_00.real,
        "Theta1": mean("mean_dot_Theta1") + coupling_t_00.real,
        "Theta2": mean("mean_dot_Theta2") + coupling_t_00.real,
    }


def print_constant_comparison(
    bayes: dict[str, np.ndarray],
    sigma: float,
    true_s: dict[tuple[int, int], complex],
    true_t: dict[tuple[int, int], complex],
) -> None:
    estimates = _constant_estimates(bayes)
    references = _measured_constant_references(sigma, true_s[(0, 0)], true_t[(0, 0)])

    print("\n[Constant terms: estimate vs measured reference]")
    print(
        "reference = phase_dot_summary.csv (eps=0) + "
        "true coupling Fourier coefficient (0,0)"
    )
    if references is None:
        print(f"sigma={sigma:.8e}: matching measured data was not found")
        for label, estimate in estimates.items():
            print(f"  {label:<6s} estimate={_format_complex(estimate)} measured=N/A")
        return

    for label in ("Phi1", "Phi2", "Theta1", "Theta2"):
        print(
            f"  {label:<6s} estimate={_format_complex(estimates[label])} "
            f"measured={references[label]:+.8e}"
        )


def _error_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    difference = np.asarray(estimate) - np.asarray(truth)
    area = np.ptp(x) * np.ptp(y)

    def normalized_l2(values: np.ndarray) -> float:
        integral_x = np.trapz(np.abs(values) ** 2, x=x, axis=1)
        integral_xy = np.trapz(integral_x, x=y)
        return float(np.sqrt(integral_xy / area))

    absolute_l2 = normalized_l2(difference)
    truth_l2 = normalized_l2(truth)
    relative_l2 = absolute_l2 / truth_l2 if truth_l2 > 0.0 else np.nan
    max_absolute = float(np.max(np.abs(difference)))
    return absolute_l2, relative_l2, max_absolute


def print_phase_equation_errors(
    means: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    bayes: dict[str, np.ndarray],
) -> None:
    print("\n[Nonconstant phase-equation errors]")
    print("component  L2(error)       relative-L2     max|error|")
    truth_by_component = {
        "Phi1": truth["Phi"],
        "Phi2": truth["Phi"],
        "Theta1": truth["Theta"],
        "Theta2": truth["Theta"],
    }
    for label in ("Phi1", "Phi2", "Theta1", "Theta2"):
        absolute_l2, relative_l2, max_absolute = _error_metrics(
            means[label],
            truth_by_component[label],
            means["x"],
            means["y"],
        )
        print(
            f"{label:<9s} {absolute_l2: .8e}  "
            f"{relative_l2: .8e}  {max_absolute: .8e}"
        )

    if "error1" in bayes or "error2" in bayes:
        print("\n[Bayes iteration errors stored in the result]")
        for key in ("error1", "error2"):
            if key in bayes:
                values = np.asarray(bayes[key]).ravel()
                print(f"  {key}={np.array2string(values, precision=8, separator=', ')}")


def print_fourier_coefficients(
    bayes: dict[str, np.ndarray],
    true_s: dict[tuple[int, int], complex],
    true_t: dict[tuple[int, int], complex],
    modes: Iterable[tuple[int, int]] = FOURIER_MODES_TO_PRINT,
) -> None:
    m_profile = bayes["m_profile"]
    index_by_mode = {
        (int(ms), int(mt)): index
        for index, (ms, mt) in enumerate(m_profile.tolist())
    }
    block_size = len(m_profile)
    a1 = bayes["param_a1"]
    a2 = bayes["param_a2"]

    print("\n[Absolute Fourier coefficients: estimate vs truth]")
    print("mode=(m_s,m_t); epsilon is included in the true coefficients")
    for component, offset, truth in (
        ("s", 0, true_s),
        ("t", block_size, true_t),
    ):
        print(f"  component={component}")
        for mode in modes:
            index = index_by_mode.get(mode)
            true_value = truth.get(mode)
            if index is None or true_value is None:
                print(f"    mode={mode!s:<8s} |estimate1|=N/A |estimate2|=N/A |true|=N/A")
                continue
            estimate1 = float(np.abs(a1[index + offset]))
            estimate2 = float(np.abs(a2[index + offset]))
            true_absolute = float(np.abs(true_value))
            print(
                f"    mode={mode!s:<8s} "
                f"|estimate1|={estimate1:.8e} "
                f"|estimate2|={estimate2:.8e} "
                f"|true|={true_absolute:.8e}"
            )


def _add_colorbar(fig, ax, image) -> None:
    cax = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="center left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.locator = mticker.MaxNLocator(nbins=4)
    colorbar.formatter = mticker.ScalarFormatter(useMathText=True)
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()


def save_mean_colormap(
    means: dict[str, np.ndarray],
    sigma: float,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a 2-by-2 colormap of the nonconstant posterior means."""
    x_mesh, y_mesh = np.meshgrid(means["x"] / L, means["y"] / np.pi, indexing="xy")
    fig, axes = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
    fig.subplots_adjust(wspace=0.525, hspace=0.65, right=0.85, bottom=0.175)
    fig.suptitle(rf"MAP Estimate: $\sigma^2 = {sigma**2:.1e}$", fontsize=14)

    panels = (
        (axes[0, 0], means["Phi1"], -2.0e-4, 2.0e-4,
         r"$\epsilon \hat{\Gamma}^\mathrm{s}_1 (\Delta \Phi, \Delta \Theta)$"),
        (axes[0, 1], means["Phi2"], -2.0e-4, 2.0e-4,
         r"$\epsilon \hat{\Gamma}^\mathrm{s}_2 (\Delta \Phi, \Delta \Theta)$"),
        (axes[1, 0], means["Theta1"], -4.0e-5, 4.0e-5,
         r"$\epsilon \hat{\Gamma}^\mathrm{t}_1 (\Delta \Phi, \Delta \Theta)$"),
        (axes[1, 1], means["Theta2"], -4.0e-5, 4.0e-5,
         r"$\epsilon \hat{\Gamma}^\mathrm{t}_2 (\Delta \Phi, \Delta \Theta)$"),
    )
    for ax, values, vmin, vmax, title in panels:
        image = ax.pcolormesh(
            x_mesh,
            y_mesh,
            values,
            cmap="seismic",
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        _add_colorbar(fig, ax, image)
        ax.set_xlabel(r"$\Delta \Phi / L$")
        ax.set_ylabel(r"$\Delta \Theta / \pi$")
        ax.set_aspect(np.ptp(x_mesh) / np.ptp(y_mesh))
        ax.set_title(title)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plot.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def process_file(file_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    file_path = Path(file_path)
    nonconstant_path = nonconstant_file(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Bayes file not found: {file_path}")
    if not nonconstant_path.is_file():
        raise FileNotFoundError(f"Nonconstant Bayes file not found: {nonconstant_path}")

    epsilon, sigma = parse_eps_sigma(file_path)
    bayes = load_bayes_data(file_path)
    means = load_nonconstant_means(nonconstant_path)
    truth = load_true_nonconstant(epsilon, means["x"], means["y"])
    true_s, true_t = load_true_fourier_coefficients(epsilon)

    print("\n" + "=" * 88)
    print(f"source={file_path}")
    print(f"epsilon={epsilon:.8e} sigma={sigma:.8e} sigma^2={sigma**2:.8e}")
    print_constant_comparison(bayes, sigma, true_s, true_t)
    print_phase_equation_errors(means, truth, bayes)
    print_fourier_coefficients(bayes, true_s, true_t)

    output_path = save_mean_colormap(means, sigma, output_dir=output_dir)
    print(f"\nsaved mean colormap: {output_path}")
    return output_path


def main() -> None:
    for file_path in BAYES_FILES:
        process_file(file_path)


if __name__ == "__main__":
    main()
