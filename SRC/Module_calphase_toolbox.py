import numpy as np
import cupy as cp
import sys, os
import pickle
from numpy import linalg as LA
import gc, os, re, sys
from collections import defaultdict
from typing import Tuple, List, Callable
from scipy.interpolate import interp1d
from functions.intersect_time import intersect_time
import Module_get_envname
import Module_Kralemann_GPU 
from scipy.signal import hilbert
from cupyx.scipy.signal import hilbert as hilbert_gpu



SECTION = Module_get_envname.SECTION



def find_crossing_time(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return times when ``x(t)`` crosses ``SECTION`` upward."""
    return intersect_time(t, x, section=SECTION)



def calculate_Theta(x: np.ndarray, t: np.ndarray, return_mask: bool=False) -> np.ndarray:
    """
    Linearly interpolate temporal phase between upward section crossings.

    If ``return_mask`` is true, also return the mask that selects the samples
    covered by complete crossing intervals.
    """
    t_cross = find_crossing_time(x, t)

    x = t_cross 
    y = 2.0 * np.pi * np.arange(len(t_cross)) 
    func_interp_Theta = interp1d(x, y, kind="linear")
    mask = (t >= t_cross[0]) & (t < t_cross[-1])
    _t = t[mask]
    Theta = func_interp_Theta(_t)
    
    if return_mask:
        mask_array = mask
        return np.mod(Theta.astype(float), 2.0*np.pi), mask_array
    else:
        return np.mod(Theta.astype(float), 2.0*np.pi)



def calculate_Theta_Kralemann(x: np.ndarray, t: np.ndarray, 
                              dump_length: float = 1000.0, 
                              M: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate phase with the Kralemann transform.

    ``dump_length`` is removed from both ends to suppress Hilbert-transform
    edge effects. The result is shifted so section crossings lie near zero.
    ``M`` is the Fourier truncation order.
    """

    t_gpu = cp.asarray(t)
    x_gpu = cp.asarray(x)

    # Remove both ends of the time series.
    s = t[0] + dump_length
    e = t[-1] - dump_length
    mask_dump = (s <= t_gpu) & (t_gpu < e)

    t_masked_gpu = t_gpu[mask_dump]

    x_masked_gpu = x_gpu[mask_dump]
    t_cross = find_crossing_time(x_masked_gpu.get(), t_masked_gpu.get())

    # Obtain the protophase from the analytic signal.
    analytic_signal_gpu = hilbert_gpu(x_gpu)
    analytic_signal_gpu = analytic_signal_gpu[mask_dump]
    analytic_signal_gpu -= cp.mean(analytic_signal_gpu)
    proto_phase_gpu = cp.mod(cp.angle(analytic_signal_gpu), 2.0*np.pi) 


    pt = Module_Kralemann_GPU.PhaseTransform(M=M)
    Sm_array = pt.return_Sm(proto_phase_gpu)
    Theta_gpu = pt.kralemann_phase_transform(proto_phase_gpu, Sm_array)
    Theta_gpu = cp.mod(Theta_gpu, 2.0*np.pi)

    Theta_np = Theta_gpu.get()

    # Shift the circular mean phase at section crossings to zero.
    _x = t_masked_gpu.get()
    _y = np.unwrap(Theta_np)
    xp = t_cross
    func_Theta = interp1d(_x, _y, kind="linear")
    Theta_tc = func_Theta(xp)
    shift = np.angle(np.mean(np.exp(1j * Theta_tc)))

    Theta = np.mod(Theta_np - shift, 2.0*np.pi)

    mask_dump_np = mask_dump.get()
    return Theta.astype(float), mask_dump_np





def calculate_c(A_tilde_list: List[np.ndarray], x_list: List[np.ndarray], t_list: List[np.ndarray], params: dict) -> Tuple[float, float, List[np.ndarray]]:
    """
    Estimate translation speed from phase increments at section crossings.

    Each list element represents one trajectory. The regression model is

        (L / pi) * diff(arg(A_tilde(t_n))) = c * diff(t_n) + eta_n.

    Returns the maximum-likelihood speed, residual variance, and residuals
    split by trajectory.
    """
    L = params["L"]


    yi_list = []
    xi_list = []
    for A_tilde, x, t in zip(A_tilde_list, x_list, t_list):
        tn = find_crossing_time(x, t)

        argA = np.unwrap(np.angle(A_tilde))

        func_Atilde = interp1d(t, argA, kind="linear")
        argA_tn = func_Atilde(tn)
    
        d_argA = np.diff(argA_tn)
        Delta_T   = np.diff(tn)

        yi = (L / np.pi) * d_argA
        xi = Delta_T

        yi_list.append(yi)
        xi_list.append(xi)

    yi = np.concatenate(yi_list).astype(float)
    xi = np.concatenate(xi_list).astype(float)

    c_hat = np.sum(xi * yi) / np.sum(xi**2)

    eta_i = yi - c_hat * xi
    sigma2_hat = np.sum(eta_i**2) / len(yi)
    

    residuals = []
    for xi, yi in zip(xi_list, yi_list):
        eta_i = yi - c_hat * xi
        residuals.append(eta_i.astype(float))

    return float(c_hat), float(sigma2_hat), residuals


def calculate_b(A_tilde, Theta, t, params):
    """
    Construct ``b_i`` on complete interior phase laps without resampling.

    A_tilde : array-like, shape (N,) [complex]
    Theta   : array-like, shape (N,) [rad]
    t       : array-like, shape (N,) [float]

    The input arrays describe one trajectory and must have equal length.
    The first and last laps are discarded. For each remaining lap, the
    phase-average is subtracted from ``y = (L/pi)*arg(A_tilde) - c*t``.

    Returns
    -------
    out : list of dict
        Per-lap time, wrapped phase, centered ``b``, and phase average.
    """

    L = params["L"]
    c = params["c"]

    argA = np.unwrap(np.angle(A_tilde))
    y = (L/np.pi) * argA - c * t

    # Assign a lap number to each unwrapped phase sample.
    Th_u    = np.unwrap(Theta)
    lap_idx = np.floor((Th_u - Th_u[0]) / (2*np.pi)).astype(int)
    
    laps = np.arange(lap_idx.min(), lap_idx.max()+1)
    if laps.size <= 2:
        return []

    # Exclude the incomplete edge laps.
    core_laps = laps[1:-1]

    out = []

    unique_laps, lap_counts = np.unique(lap_idx, return_counts=True)
    count_map = dict(zip(unique_laps, lap_counts))
    list1 = np.array([count_map[i] for i in core_laps], dtype=int) 
    n_max = np.max(list1)
    n_min = np.min(list1)
    print(f"[calculate_b] lap count: min={n_min}, max={n_max}, mean={list1.mean()}", flush=True)

    # Report violations of the monotonic-phase assumption.
    dTh = np.diff(Th_u.astype(float))
    if not np.all(dTh > 0):
        print(f"[calculate_b] Warning: Th_u is not strictly increasing! min(dTh)={dTh.min()}, max(dTh)={dTh.max()}", flush=True)
    else:
        print(f"[calculate_b] Th_u is strictly increasing.", flush=True)

    y_interp = interp1d(Th_u, y, kind="linear")

    # Build a common quadrature grid and slices for each lap.
    grid = np.linspace(0.0, 2.0*np.pi, 2*10+1)
    lap_edges = np.concatenate(([0], np.flatnonzero(np.diff(lap_idx)) + 1, [lap_idx.size]))
    lap_slices = {lap: slice(lap_edges[idx], lap_edges[idx+1]) for idx, lap in enumerate(laps)}

    for i in core_laps:
        span = lap_slices[i]
        start = span.start
        end = span.stop
        n = Th_u[start] // (2.0*np.pi)

        xp = grid + 2.0*np.pi * n
        yp = y_interp(xp)
        y_mean = np.trapz(yp, xp) / (2.0*np.pi)

        y_i = y[span]
        b_i = y_i - y_mean

        out.append({
            'lap':   int(i),
            't':     t[span],
            'theta': np.mod(Th_u[span], 2*np.pi),
            'b':     b_i,
            'y_mean': float(y_mean),
        })
    return out


def concatenate_theta_b(out):
    """
    Flatten the list of per-lap dictionaries created in calculate_b into
    two 1-D numpy arrays: theta values and their corresponding b values.

    Parameters
    ----------
    out : list of dict
        Output from calculate_b. Each dict has keys 'theta' and 'b'.

    Returns
    -------
    theta_concat : ndarray, shape (N,)
    b_concat     : ndarray, shape (N,)
    """
    if not out:
        return np.array([], dtype=float), np.array([], dtype=float)

    theta_concat = np.concatenate([entry['theta'] for entry in out]).astype(float)
    b_concat = np.concatenate([entry['b'] for entry in out]).astype(float)
    return theta_concat, b_concat





def fit_B_function(Theta, B, M=10, l2=0.0, kind="linear", grid_points=2**12+1) -> dict:
    """
    Fit ``B(Theta)`` with a zero-constant Fourier series by least squares.

    The fitted curve is shifted to have zero uniform mean and sampled on a
    periodic grid.

    Parameters
    ----------
    Theta : array-like [rad], shape (N,)
    B     : array-like, shape (N,)
    M     : Fourier truncation order
    l2    : ridge penalty; zero disables regularization
    kind  : retained interpolation-method parameter
    grid_points : number of samples on the returned uniform grid

    Returns
    -------
    model : dict
        Uniform samples ``Bx`` and ``By``, Fourier coefficients ``alpha`` and
        ``beta``, truncation order ``M``, and the removed mean ``offset``.
    """
    # Normalize angles to [0, 2π) to respect periodicity of the basis.
    th = np.mod(np.asarray(Theta, float), 2*np.pi)
    y  = np.asarray(B, float)

    # Build Fourier design matrix with harmonics up to order M.
    cos_block = np.cos(np.outer(th, np.arange(1, M+1)))
    sin_block = np.sin(np.outer(th, np.arange(1, M+1)))
    X = np.column_stack([cos_block, sin_block]).astype(float)

    if l2 > 0:
        # Solve a ridge-regularized normal equation when λ > 0.
        XtX = X.T @ X
        XtX.flat[::XtX.shape[0]+1] += l2
        coef = np.linalg.solve(XtX, X.T @ y)
    else:
        # Otherwise fall back to the plain least-squares solution.
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    alpha = coef[:M]
    beta  = coef[M:]

    def _B(theta):
        theta = np.asarray(theta, float)
        out = np.zeros_like(theta, dtype=float)
        for m in range(1, M+1):
            out += alpha[m-1]*np.cos(m*theta) + beta[m-1]*np.sin(m*theta)
        return out

    # Subtract the uniform mean so Bhat integrates to zero.
    grid = np.linspace(0, 2*np.pi, 2**12+1)
    offset = np.trapz(_B(grid), grid) / (2.0*np.pi)

    def Bhat(theta):
        return _B(theta) - offset

    # Sample on a dense uniform grid (including 2π) to build a periodic interpolator.
    xx = np.linspace(0, 2*np.pi, grid_points)
    yy = Bhat(xx)

    Bx = xx
    By = yy

    return {"Bx": Bx, "By": By,
            "alpha": alpha, "beta": beta, "M": M, "offset": float(offset)}



def calculate_Phi(A_tilde:np.ndarray, Theta:np.ndarray, 
                  Bfunc:Callable[[np.ndarray], np.ndarray],
                  params:dict) -> np.ndarray:
    """Compute ``Phi(t) = (L/pi)*arg(A_tilde(t)) - B(Theta(t))``."""

    L = params["L"]
    # c = params["c"]

    argA = np.mod(np.angle(A_tilde), 2.0*np.pi)
    B = Bfunc(Theta)
    Phi = (L/np.pi) * argA - B
    Phi = np.mod(Phi, 2.0*L)

    # Map [0, 2L) to [-L, L).
    Phi[Phi >= L] -= 2.0*L
    return Phi




def get_common_time(t1: np.ndarray, t2: np.ndarray):
    """Return masks selecting the overlapping interval of two time grids."""
    s = np.max([t1.min(), t2.min()])
    e = np.min([t1.max(), t2.max()])
    mask1 = (s <= t1) & (t1 <= e)
    mask2 = (s <= t2) & (t2 <= e)
    return mask1, mask2
