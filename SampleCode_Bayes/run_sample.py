#!/usr/bin/env python3
# coding: utf-8

# %% 読み込み
"""位相方程式を読み込み、CPUでSDE生成・MAP推定を実行するサンプル。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib

try:
    from IPython import get_ipython
    IPYTHON = get_ipython()
except ImportError:
    IPYTHON = None
if IPYTHON is None:
    matplotlib.use("Agg")
else:
    IPYTHON.run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt

if "__file__" in globals() and Path(__file__).name == "run_sample.py":
    SAMPLE_DIR = Path(__file__).resolve().parent
else:
    SAMPLE_DIR = next(
        (root / relative for root in (Path.cwd(), *Path.cwd().parents)
         for relative in (Path("."), Path("推定サンプルプログラム"))
         if all((root / relative / name).is_file()
                for name in ("run_sample.py", "phase_sde.py", "bayes_cpu.py"))),
        None,
    )
    if SAMPLE_DIR is None:
        raise FileNotFoundError("サンプルのフォルダ内、またはリポジトリ内を作業ディレクトリにしてください。")
sys.path.insert(0, str(SAMPLE_DIR))
from phase_sde import load_phase_equation, simulate_dataset, prepare_estimation_data
from bayes_cpu import fit, evaluate_drift


# %% 設定（変更後に最後の実行セルを実行）
INPUT_FILE = SAMPLE_DIR / "InputData" / "phase_equation.npz"
OUTPUT_DIR = SAMPLE_DIR / "output"
STAGE = "all"  # "all", "generate", "estimate"
EPSILON = 1.0e-6
SIGMA = 1.0e-6
DT = 0.05
T_END = 1000.0
SAVE_EVERY = 20  # 保存間隔 Delta_t = DT * SAVE_EVERY = 1.0
N_PHI, N_THETA = 16, 8  # 空間方向は 2*MS+1 より多い初期位相を用意する。
MS, MT = 4, 2  # CPUで試せる小規模設定。論文の次数は 20, 5。
BINS = 16
MAX_PER_CELL = 100
MAX_ITERATIONS = 50
RTOL = 1.0e-5
PRIOR_PRECISION = 1.0e-6
NOISE_FLOOR = 1.0e-14
SEED = 415643
OVERWRITE = True  # セル再実行時の、このサンプルの出力のみ。


# %% 実行関数
DATA_KEYS = (
    "Phi1_ast", "Phi2_ast", "Theta1_ast", "Theta2_ast",
    "dot_Phi1", "dot_Phi2", "dot_Theta1", "dot_Theta2",
)


def save_dataset(path, dataset):
    payload = {key: value for key, value in dataset.items() if key != "metadata"}
    payload["metadata_json"] = np.asarray(json.dumps(dataset.get("metadata", {}), ensure_ascii=False))
    np.savez_compressed(path, **payload)


def plot_comparison(model, results, epsilon, output_dir):
    """系1の非定数部分を、読み込んだ方程式と比較する。"""
    phi, theta = np.meshgrid(model["phi_grid"], model["theta_grid"], indexing="xy")
    predicted = evaluate_drift(results[0], phi, theta)
    truth = np.stack((model["c"] + epsilon * model["gamma_s"],
                      model["omega"] + epsilon * model["gamma_t"]), axis=-1)
    # 周期端点を二重に数えず、定数項と非定数項を分ける。
    truth = truth[:-1, :-1]
    predicted = predicted[:-1, :-1]
    truth_centered = truth - np.mean(truth, axis=(0, 1))
    predicted_centered = predicted - np.mean(predicted, axis=(0, 1))
    difference = predicted_centered - truth_centered
    metrics = {}
    for component, name in enumerate(("spatial", "temporal")):
        reference_norm = np.linalg.norm(truth_centered[..., component])
        metrics[name] = {
            "true_constant": float(np.mean(truth[..., component])),
            "estimated_constant": float(np.mean(predicted[..., component])),
            "nonconstant_relative_l2_error": float(np.linalg.norm(difference[..., component]) / reference_norm)
            if reference_norm > 0 else None,
        }

    plt.rcParams.update({"font.size": 9, "font.family": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.1), layout="constrained")
    for row, name in enumerate(("Spatial", "Temporal")):
        limit = max(float(np.max(np.abs(truth_centered[..., row]))),
                    float(np.max(np.abs(predicted_centered[..., row]))), 1e-15)
        error_limit = max(float(np.max(np.abs(difference[..., row]))), 1e-15)
        for col, (title, values, bound) in enumerate((
            ("Input equation", truth_centered[..., row], limit),
            ("CPU estimate", predicted_centered[..., row], limit),
            ("Estimate - input", difference[..., row], error_limit),
        )):
            ax = axes[row, col]
            mesh = ax.pcolormesh(phi[:-1, :-1] / model["L"], theta[:-1, :-1] / np.pi,
                                 values, cmap="RdBu_r", vmin=-bound, vmax=bound, rasterized=True)
            ax.set(title=title if row == 0 else None, xlabel=r"$\Delta\Phi/L$")
            ax.set_ylabel(name + "\n" + r"$\Delta\Theta/\pi$" if col == 0 else r"$\Delta\Theta/\pi$")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            colorbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
            colorbar.formatter.set_powerlimits((0, 0))
            colorbar.update_ticks()
    fig.savefig(output_dir / "phase_equation_comparison.png", dpi=200)
    fig.savefig(output_dir / "phase_equation_comparison.pdf")
    if IPYTHON is not None:
        plt.show()
    plt.close(fig)
    return metrics, phi, theta, truth, predicted


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "generate", "estimate"), default=STAGE)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--n-phi", type=int, default=N_PHI)
    parser.add_argument("--n-theta", type=int, default=N_THETA)
    parser.add_argument("--ms", type=int, default=MS)
    parser.add_argument("--mt", type=int, default=MT)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--max-per-cell", type=int, default=MAX_PER_CELL)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--prior-precision", type=float, default=PRIOR_PRECISION)
    parser.add_argument("--noise-floor", type=float, default=NOISE_FLOOR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args):
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    model_path = args.input_file.expanduser().resolve()
    model = load_phase_equation(model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    stage_names = []
    if args.stage in ("all", "generate"):
        stage_names += ["sde_data.npz", "estimation_data.npz"]
    if args.stage in ("all", "estimate"):
        stage_names += ["estimated_phase_equation.npz", "summary.json",
                        "phase_equation_comparison.png", "phase_equation_comparison.pdf"]
    if not args.overwrite:
        existing = [output_dir / name for name in stage_names if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(f"出力済みです: {existing[0]}。別の --output-dir または --overwrite を指定してください。")
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    if args.stage in ("all", "generate"):
        print("[1/3] CPU: 位相方程式からSDEデータを生成", flush=True)
        tick = time.perf_counter()
        dataset = simulate_dataset(
            model, epsilon=args.epsilon, sigma=args.sigma, dt=args.dt,
            t_end=args.t_end, save_every=args.save_every,
            n_phi=args.n_phi, n_theta=args.n_theta, seed=args.seed,
        )
        timings["simulation_seconds"] = time.perf_counter() - tick
        dataset["equation_sha256"] = np.asarray(model_sha)
        save_dataset(output_dir / "sde_data.npz", dataset)
        print("[2/3] 中点・差分を作成し、位相差平面から推定サンプルを選択", flush=True)
        tick = time.perf_counter()
        selected = prepare_estimation_data(dataset, bins=args.bins,
                                           max_per_cell=args.max_per_cell, seed=args.seed)
        timings["preprocessing_seconds"] = time.perf_counter() - tick
        raw_points = int(np.prod(dataset["trajectory"].shape[:2]))
        pair_count = int(dataset["trajectory"].shape[0] * (dataset["trajectory"].shape[1] - 1))
        selected.update({"Delta_t": dataset["Delta_t"], "sigma": args.sigma,
                         "epsilon": args.epsilon, "L": model["L"],
                         "n_saved_timepoints": raw_points,
                         "n_increments_before_selection": pair_count,
                         "generation_configuration_json": json.dumps(
                             {key: getattr(args, key) for key in (
                                 "epsilon", "sigma", "dt", "t_end", "save_every", "n_phi",
                                 "n_theta", "bins", "max_per_cell", "seed")}),
                         "equation_sha256": np.asarray(model_sha)})
        np.savez_compressed(output_dir / "estimation_data.npz", **selected)
    else:
        with np.load(output_dir / "estimation_data.npz", allow_pickle=False) as saved:
            selected = {key: saved[key] for key in saved.files}
        if str(selected["equation_sha256"]) != model_sha:
            raise ValueError("保存データの生成に使った方程式と --input-file が異なります。")
        raw_points = int(selected["n_saved_timepoints"])
        pair_count = int(selected["n_increments_before_selection"])
    n_samples = int(selected["Phi1_ast"].size)
    payload_bytes = sum(selected[key].nbytes for key in DATA_KEYS)
    print(f"保存時点数={raw_points:,}; 増分数={pair_count:,}; 推定サンプル={n_samples:,}; "
          f"推定入力8配列={payload_bytes / 1e6:.3f} MB", flush=True)
    if args.stage == "generate":
        print(f"生成完了: {output_dir}", flush=True)
        return {"output_dir": output_dir, "n_samples": n_samples}

    print("[3/3] 固定した事前分布でCPU MAP推定（系1・系2）", flush=True)
    results = []
    for oscillator in (1, 2):
        tick = time.perf_counter()
        result = fit(selected, float(selected["Delta_t"]), L=float(selected["L"]),
                     Ms=args.ms, Mt=args.mt, oscillator=oscillator,
                     max_iterations=args.max_iterations, rtol=args.rtol,
                     prior_precision=args.prior_precision, noise_floor=args.noise_floor)
        timings[f"fit_oscillator_{oscillator}_seconds"] = time.perf_counter() - tick
        results.append(result)
        print(f"系{oscillator}: {result['iterations']}回、収束={result['converged']}", flush=True)

    metrics, phi, theta, truth, predicted = plot_comparison(
        model, results, float(selected["epsilon"]), output_dir)
    estimated = {"delta_phi": phi[0], "delta_theta": theta[:, 0],
                 "modes": results[0]["modes"], "L": float(selected["L"]),
                 "epsilon": float(selected["epsilon"]), "sigma": float(selected["sigma"])}
    for oscillator, result in enumerate(results, 1):
        for key in ("coefficients", "precision", "E", "errors", "noise_errors",
                    "noise_for_last_coefficient_update", "prior_mean", "prior_precision",
                    "converged", "iterations", "noise_floor_applications"):
            estimated[f"{key}_{oscillator}"] = result[key]
        estimated[f"drift_{oscillator}"] = evaluate_drift(result, phi, theta)
    np.savez_compressed(output_dir / "estimated_phase_equation.npz", **estimated)
    timings["total_seconds"] = time.perf_counter() - started
    summary = {
        "input_equation": str(model_path), "equation_sha256": model_sha,
        "command_configuration": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
        "generation_configuration": json.loads(str(selected.get("generation_configuration_json", "{}"))),
        "n_saved_timepoints": raw_points, "n_increments_before_selection": pair_count,
        "n_estimation_samples": n_samples, "estimation_array_bytes": payload_bytes,
        "actual_Delta_t": float(selected["Delta_t"]),
        "actual_sigma": float(selected["sigma"]),
        "actual_epsilon": float(selected["epsilon"]),
        "prior_mean": "zero (fixed)", "prior_precision": args.prior_precision,
        "prior_rhs": "Xi_prior @ a_prior (fixed for all iterations)",
        "precision_update": "Xi_prior + likelihood_precision (no repeated data accumulation)",
        "converged": [bool(result["converged"]) for result in results],
        "iterations": [int(result["iterations"]) for result in results],
        "final_coefficient_relative_changes": [float(result["errors"][-1]) for result in results],
        "final_covariance_relative_changes": [float(result["noise_errors"][-1]) for result in results],
        "noise_floor_applications": [int(result["noise_floor_applications"]) for result in results],
        "input_noise_covariance": (float(selected["sigma"])**2 * model["C"]).tolist(),
        "estimated_noise_covariance": [result["E"].tolist() for result in results],
        "oscillator_1_grid_metrics": metrics, "timings": timings,
        "note": "CPU demonstration with a small Fourier truncation; not a reproduction of paper estimates.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"完了: {output_dir}（合計 {timings['total_seconds']:.2f} 秒）", flush=True)
    return {"summary": summary, "results": results}


def main(argv=None):
    return run(parse_args(argv))


# %% 実行
if __name__ == "__main__":
    RESULT = main((["--overwrite"] if OVERWRITE else []) if IPYTHON is not None else None)
