#!/usr/bin/env python3
# coding: utf-8


"""GPU simulation of a pair of coupled Gray-Scott systems."""


import numpy as np
import cupy as cp
import gc, os, sys, time
from typing import Literal, Tuple, Optional
from functions.time_deco import log_execution_time
from Module_GrayScottModel_GPU import GrayScottModel


# Shared block size for the SPDE and random-number kernels.
# threads_per_block = 512
threads_per_block = 256
# threads_per_block = 128


#* Profiling flag (set PROFILE_SPDE=1 in env to enable)
PROFILE_SPDE = bool(int(os.getenv("PROFILE_SPDE", "0")))


#* --------------------------- RNG Utilities ------------------------------------- *#
# Generate Gaussian noise inside the SPDE kernel with xoroshiro128+ and Box-Muller.
_rng_seed: int = 0
_xor_state0: cp.ndarray | None = None
_xor_state1: cp.ndarray | None = None
_xor_capacity: int = 0
_xor_seed_cached: int | None = None


def _ensure_rng_states(n: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Ensure xoroshiro state arrays exist for at least `n` threads and match the seed."""
    global _xor_state0, _xor_state1, _xor_capacity, _xor_seed_cached

    if (
        _xor_state0 is None
        or _xor_capacity < n
        or _xor_seed_cached != _rng_seed
    ):
        new_capacity = max(n, _xor_capacity * 2 if _xor_capacity else n)
        host_rng = np.random.default_rng(_rng_seed)
        host_state0 = host_rng.integers(0, np.iinfo(np.uint64).max, size=new_capacity, dtype=np.uint64)
        host_state1 = host_rng.integers(0, np.iinfo(np.uint64).max, size=new_capacity, dtype=np.uint64)
        _xor_state0 = cp.asarray(host_state0)
        _xor_state1 = cp.asarray(host_state1)
        _xor_capacity = new_capacity
        _xor_seed_cached = _rng_seed

    return _xor_state0[:n], _xor_state1[:n]


def set_rng_seed(seed: int | None) -> None:
    """Expose RNG seed control for external scripts (xoroshiro128+ helper)."""
    global _rng_seed, _xor_seed_cached
    _rng_seed = 0 if seed is None else int(seed)
    _xor_seed_cached = None


#* --------------------------- CUDA Kernels ------------------------------------- *#
coupled_pde_kernel = cp.RawKernel(r'''
extern "C" __device__ inline unsigned long long rotl64(const unsigned long long x, const int k) {
    return (x << k) | (x >> (64 - k));
}

extern "C" __device__ inline unsigned long long xoroshiro128p_next(unsigned long long &s0, unsigned long long &s1) {
    const unsigned long long result = s0 + s1;
    s1 ^= s0;
    s0 = rotl64(s0, 55) ^ s1 ^ (s1 << 14);
    s1 = rotl64(s1, 36);
    return result;
}

extern "C" __device__ inline double uniform01(unsigned long long &s0, unsigned long long &s1) {
    const unsigned long long bits = xoroshiro128p_next(s0, s1) >> 11;
    return (double)bits * (1.0 / 9007199254740992.0);
}

extern "C" __device__ inline void box_muller2(unsigned long long &s0, unsigned long long &s1, double &out0, double &out1) {
    double u1 = uniform01(s0, s1);
    if (u1 <= 1e-16) {
        u1 = 1e-16;
    }
    const double u2 = uniform01(s0, s1);
    const double radius = sqrt(-2.0 * log(u1));
    const double theta = 6.28318530717958647692 * u2;
    out0 = radius * cos(theta);
    out1 = radius * sin(theta);
}

extern "C" __global__
void compute_pde_rng(
    double* __restrict__ U1,
    double* __restrict__ V1,
    double* __restrict__ U2,
    double* __restrict__ V2,
    unsigned long long* __restrict__ state0,
    unsigned long long* __restrict__ state1,
    double epsilon,
    double noise_std,
    double Du,
    double Dv,
    double h,
    double c,
    double f,
    double k,
    int N,
    double dt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Generate four independent standard-normal values per thread.
    unsigned long long s0 = state0[i];
    unsigned long long s1 = state1[i];
    double z0, z1, z2, z3;
    box_muller2(s0, s1, z0, z1);
    box_muller2(s0, s1, z2, z3);
    state0[i] = s0;
    state1[i] = s1;

    const double dW_u1 = noise_std * z0;
    const double dW_v1 = noise_std * z1;
    const double dW_u2 = noise_std * z2;
    const double dW_v2 = noise_std * z3;

    const int ip = (i + 1) % N;
    const int im = (i - 1 + N) % N;

    const double u1 = U1[i];
    const double v1 = V1[i];
    const double u2 = U2[i];
    const double v2 = V2[i];

    const double u1_ip = U1[ip];
    const double u1_im = U1[im];
    const double v1_ip = V1[ip];
    const double v1_im = V1[im];
    const double u2_ip = U2[ip];
    const double u2_im = U2[im];
    const double v2_ip = V2[ip];
    const double v2_im = V2[im];

    const double inv_h2 = 1.0 / (h * h);
    const double inv_2h = 0.5 / h;

    const double Fu1 = u1 * u1 * v1 - (f + k) * u1;
    const double Fv1 = -u1 * u1 * v1 + f * (1.0 - v1);
    const double Fu2 = u2 * u2 * v2 - (f + k) * u2;
    const double Fv2 = -u2 * u2 * v2 + f * (1.0 - v2);

    const double diff_u1 = Du * (u1_ip - 2.0 * u1 + u1_im) * inv_h2;
    const double diff_v1 = Dv * (v1_ip - 2.0 * v1 + v1_im) * inv_h2;
    const double diff_u2 = Du * (u2_ip - 2.0 * u2 + u2_im) * inv_h2;
    const double diff_v2 = Dv * (v2_ip - 2.0 * v2 + v2_im) * inv_h2;

    const double grad_u1 = c * (u1_ip - u1_im) * inv_2h;
    const double grad_v1 = c * (v1_ip - v1_im) * inv_2h;
    const double grad_u2 = c * (u2_ip - u2_im) * inv_2h;
    const double grad_v2 = c * (v2_ip - v2_im) * inv_2h;

    const double G_u1 = epsilon * (u2 - u1);
    const double G_v1 = epsilon * (v2 - v1);
    const double G_u2 = epsilon * (u1 - u2);
    const double G_v2 = epsilon * (v1 - v2);

    U1[i] = u1 + dt * (Fu1 + diff_u1 + grad_u1 + G_u1) + dW_u1;
    V1[i] = v1 + dt * (Fv1 + diff_v1 + grad_v1 + G_v1) + dW_v1;
    U2[i] = u2 + dt * (Fu2 + diff_u2 + grad_u2 + G_u2) + dW_u2;
    V2[i] = v2 + dt * (Fv2 + diff_v2 + grad_v2 + G_v2) + dW_v2;
}
''', 'compute_pde_rng')



integral_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_moments(
    const double* __restrict__ U1,
    const double* __restrict__ V1,
    const double* __restrict__ U2,
    const double* __restrict__ V2,
    const double* __restrict__ phase_re,
    const double* __restrict__ phase_im,
    double h, int N,
    double* __restrict__ out_U1,
    double* __restrict__ out_V1,
    double* __restrict__ out_U2,
    double* __restrict__ out_V2,
    double* __restrict__ out_A1_re,
    double* __restrict__ out_A1_im,
    double* __restrict__ out_A2_re,
    double* __restrict__ out_A2_im
){
    // Store block-level partial sums in shared memory.
    extern __shared__ double sdata[];
    double* sU1  = sdata;
    double* sV1  = sU1  + blockDim.x;
    double* sU2  = sV1  + blockDim.x;
    double* sV2  = sU2  + blockDim.x;
    double* sA1r = sV2  + blockDim.x;
    double* sA1i = sA1r + blockDim.x;
    double* sA2r = sA1i + blockDim.x;
    double* sA2i = sA2r + blockDim.x;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load field and phase values; out-of-range entries remain zero.
    double u1 = 0.0, v1 = 0.0, u2 = 0.0, v2 = 0.0, phire = 0.0, phiim = 0.0;
    if (gid < N){
        u1    = U1[gid];
        v1    = V1[gid];
        u2    = U2[gid];
        v2    = V2[gid];
        phire = phase_re[gid];
        phiim = phase_im[gid];
    }

    // Unit weights implement the periodic trapezoidal rule because x=L is omitted.
    sU1[threadIdx.x]  = u1;
    sV1[threadIdx.x]  = v1;
    sU2[threadIdx.x]  = u2;
    sV2[threadIdx.x]  = v2;
    sA1r[threadIdx.x] = u1 * phire;
    sA1i[threadIdx.x] = u1 * phiim;
    sA2r[threadIdx.x] = u2 * phire;
    sA2i[threadIdx.x] = u2 * phiim;
    __syncthreads();

    // Standard within-block reduction.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if (threadIdx.x < offset){
            sU1[threadIdx.x]  += sU1[threadIdx.x + offset];
            sV1[threadIdx.x]  += sV1[threadIdx.x + offset];
            sU2[threadIdx.x]  += sU2[threadIdx.x + offset];
            sV2[threadIdx.x]  += sV2[threadIdx.x + offset];
            sA1r[threadIdx.x] += sA1r[threadIdx.x + offset];
            sA1i[threadIdx.x] += sA1i[threadIdx.x + offset];
            sA2r[threadIdx.x] += sA2r[threadIdx.x + offset];
            sA2i[threadIdx.x] += sA2i[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Accumulate the partial sum from each block.
    if (threadIdx.x == 0){
        double scale = h;
        atomicAdd(out_U1,    scale * sU1[0]);
        atomicAdd(out_V1,    scale * sV1[0]);
        atomicAdd(out_U2,    scale * sU2[0]);
        atomicAdd(out_V2,    scale * sV2[0]);
        atomicAdd(out_A1_re, scale * sA1r[0]);
        atomicAdd(out_A1_im, scale * sA1i[0]);
        atomicAdd(out_A2_re, scale * sA2r[0]);
        atomicAdd(out_A2_im, scale * sA2i[0]);
    }
}
''', 'compute_moments')


dot_partial_kernel = cp.RawKernel(r'''
extern "C" __global__
void dot_partial(const double* __restrict__ a,
                 const double* __restrict__ b,
                 double* __restrict__ partial,
                 double scale,
                 int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    double val = 0.0;
    if (gid < N) {
        val = a[gid] * b[gid] * scale;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
''', 'dot_partial')


dot_reduce_kernel = cp.RawKernel(r'''
extern "C" __global__
void reduce_partial(const double* __restrict__ partial,
                    double* __restrict__ out,
                    int num_partials)
{
    double sum = 0.0;
    for (int i = 0; i < num_partials; ++i) {
        sum += partial[i];
    }
    out[0] = sum;
}
''', 'reduce_partial')


dot_kernel = cp.RawKernel(r'''
extern "C" __global__
void dot_reduce(const double* __restrict__ a,
                const double* __restrict__ b,
                double* __restrict__ out,
                double scale,
                int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    double val = 0.0;
    if (gid < N) {
        val = a[gid] * b[gid] * scale;
    }
    sdata[tid] = val;
    __syncthreads();

    // Within-block reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}
''', 'dot_reduce')







#* --------------------------- Time Integration --------------------------------- *#
def Euler_Maruyama(X1_init: Tuple[cp.ndarray, cp.ndarray], X2_init: Tuple[cp.ndarray, cp.ndarray],
                rx: np.ndarray,
                epsilon:float, sigma:float, D: cp.ndarray, h: float, c: float, params: tuple, 
                 dt: float, iteration: int,
                 log_dt: Optional[float] = None,
                 start_time: float = 0.0):
    """
    Advance the coupled SPDE with Euler-Maruyama on the GPU.

    Parameters
    ----------
    rx : np.ndarray
        One-dimensional spatial coordinates with the duplicated endpoint removed.
    log_dt : float | None, optional
        Sampling interval for spatial moments. ``None`` disables sampling.
    """

    if not isinstance(params, tuple) or len(params) != 2:
        raise ValueError("params must be a tuple of two float values (f, k)")

    # 
    U1, V1 = X1_init
    U2, V2 = X2_init

    f, k = params
    f, k = cp.float64(f), cp.float64(k)

    h_scalar = float(h)
    dt_scalar = float(dt)
    sigma_scalar = float(sigma)
    
    # Block noise with a block width equal to h.
    noise_std = cp.float64(sigma_scalar * np.sqrt(dt_scalar))

    epsilon = cp.float64(epsilon)
    Du = cp.float64(D[0, 0])
    Dv = cp.float64(D[1, 1])
    c = cp.float64(c)
    h = cp.float64(h_scalar)

    # spde_euler removes the duplicated periodic endpoint.
    N = U1.size

    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    #* RNG states are mutated in-place inside the PDE kernel; get views sized for N
    state0, state1 = _ensure_rng_states(N)

    collect_moments = (log_dt is not None)
    if collect_moments and log_dt <= 0.0:
        raise ValueError("log_dt must be positive or None")
    if collect_moments and not (dt < log_dt):
        raise ValueError("log_dt must be strictly greater than dt to avoid skipped logs")

    moment_elapsed_ms = 0.0
    moment_calls = 0
    if collect_moments:
        L = float((np.ptp(rx) + h) / 2.0)
        rx_host = np.asarray(rx, dtype=np.float64).ravel()
        grids = cp.asarray(rx_host, dtype=cp.float64)
        phase = cp.exp(1j * cp.pi * grids / L)
        phase_re = phase.real
        phase_im = phase.imag
        ones = cp.ones(N, dtype=cp.float64)


        U1_sum = cp.zeros((), dtype=cp.float64)
        V1_sum = cp.zeros((), dtype=cp.float64)
        U2_sum = cp.zeros((), dtype=cp.float64)
        V2_sum = cp.zeros((), dtype=cp.float64)
        A1_re  = cp.zeros((), dtype=cp.float64)
        A1_im  = cp.zeros((), dtype=cp.float64)
        A2_re  = cp.zeros((), dtype=cp.float64)
        A2_im  = cp.zeros((), dtype=cp.float64)
        re_buf = cp.zeros((), dtype=cp.float64)
        im_buf = cp.zeros((), dtype=cp.float64)
        dot_out = cp.zeros((), dtype=cp.float64)
        partial = cp.empty(blocks_per_grid, dtype=cp.float64)

        shared = threads_per_block * cp.dtype(cp.float64).itemsize

        moments_time: list[float] = []
        U1_bar_log_gpu: list[cp.ndarray] = []
        U2_bar_log_gpu: list[cp.ndarray] = []
        V1_bar_log_gpu: list[cp.ndarray] = []
        V2_bar_log_gpu: list[cp.ndarray] = []
        A1_tilde_re_gpu: list[cp.ndarray] = []
        A1_tilde_im_gpu: list[cp.ndarray] = []
        A2_tilde_re_gpu: list[cp.ndarray] = []
        A2_tilde_im_gpu: list[cp.ndarray] = []

        if PROFILE_SPDE:
            #! profile: events for Atilde/Xbar logging
            moment_event_start = cp.cuda.Event()
            moment_event_stop = cp.cuda.Event()

        from math import floor, ceil, isclose
        steps_per_log = int(round(float(log_dt) / float(dt)))
        if steps_per_log <= 0 or not isclose(steps_per_log * float(dt), float(log_dt), rel_tol=0.0, abs_tol=1e-12 * float(log_dt) + 1e-15):
            raise ValueError("log_dt must be an integer multiple of dt")
        current_time = float(start_time)
        next_log_time = (floor(current_time / log_dt) + 1.0) * log_dt
        if next_log_time <= current_time + 1e-12:
            next_log_time += log_dt
        steps_to_next = max(1, int(ceil((next_log_time - current_time - 1e-12) / dt)))
        next_log_step = steps_to_next
    else:
        phase = phase_re = phase_im = None
        U1_sum = V1_sum = U2_sum = V2_sum = A1_re = A1_im = A2_re = A2_im = None
        shared = 0
        moments_time = U1_bar_log_gpu = U2_bar_log_gpu = V1_bar_log_gpu = V2_bar_log_gpu = A1_tilde_re_gpu = A1_tilde_im_gpu = A2_tilde_re_gpu = A2_tilde_im_gpu = None
        current_time = float(start_time)
        next_log_time = None
        steps_per_log = None
        next_log_step = None

    #! profile: setup CUDA events when profiling is enabled
    if PROFILE_SPDE:
        kernel_start = cp.cuda.Event()
        kernel_stop = cp.cuda.Event()
        kernel_elapsed_ms = 0.0


    def accumulate(a, b, scale, dest):
        partial.fill(0)
        dot_out.fill(0)
        dot_partial_kernel((blocks_per_grid,), (threads_per_block,),
            (a, b, partial, scale, N),
            shared_mem=shared)
        dot_reduce_kernel((1,), (1,), (partial, dot_out, blocks_per_grid))
        dest[...] = dot_out

    for step in range(1, iteration + 1):
        if PROFILE_SPDE:
            kernel_start.record()
        # Advance one SPDE step, including noise generation.
        coupled_pde_kernel((blocks_per_grid,), (threads_per_block,),
            (U1, V1, U2, V2, state0, state1,
             epsilon, noise_std, Du, Dv, h, c, f, k, N, dt))
        if PROFILE_SPDE:
            kernel_stop.record()

        if PROFILE_SPDE:
            kernel_stop.synchronize()
            kernel_elapsed_ms += cp.cuda.get_elapsed_time(kernel_start, kernel_stop)
        current_time += float(dt)

        if collect_moments:
            if step == next_log_step:
                if PROFILE_SPDE:
                    moment_event_start.record()
                U1_sum.fill(0)
                V1_sum.fill(0)
                U2_sum.fill(0)
                V2_sum.fill(0)
                A1_re.fill(0)
                A1_im.fill(0)
                A2_re.fill(0)
                A2_im.fill(0)

                scale = h
                # Spatial integrals of the four fields.
                cp.dot(U1, ones, out=U1_sum)
                U1_sum *= scale

                cp.dot(V1, ones, out=V1_sum)
                V1_sum *= scale

                cp.dot(U2, ones, out=U2_sum)
                U2_sum *= scale

                cp.dot(V2, ones, out=V2_sum)
                V2_sum *= scale

                # Real and imaginary components of the complex amplitudes.
                cp.dot(U1, phase_re, out=re_buf)
                re_buf *= scale
                A1_re[...] = re_buf

                cp.dot(U1, phase_im, out=im_buf)
                im_buf *= scale
                A1_im[...] = im_buf

                cp.dot(U2, phase_re, out=re_buf)
                re_buf *= scale
                A2_re[...] = re_buf

                cp.dot(U2, phase_im, out=im_buf)
                im_buf *= scale
                A2_im[...] = im_buf

                # Retain samples on the GPU until this integration segment ends.
                moments_time.append(current_time)
                U1_bar_log_gpu.append(U1_sum.copy())
                U2_bar_log_gpu.append(U2_sum.copy())
                V1_bar_log_gpu.append(V1_sum.copy())
                V2_bar_log_gpu.append(V2_sum.copy())
                A1_tilde_re_gpu.append(A1_re.copy())
                A1_tilde_im_gpu.append(A1_im.copy())
                A2_tilde_re_gpu.append(A2_re.copy())
                A2_tilde_im_gpu.append(A2_im.copy())



                next_log_time += log_dt
                next_log_step += steps_per_log

                if PROFILE_SPDE:
                    moment_event_stop.record()
                    moment_event_stop.synchronize()
                    moment_elapsed_ms += cp.cuda.get_elapsed_time(moment_event_start, moment_event_stop)
                    moment_calls += 1

    diagnostics = None
    if collect_moments:
        if moments_time:

            # Transfer the accumulated diagnostics to the CPU.
            U1_bar = cp.stack(U1_bar_log_gpu, axis=0).get()
            V1_bar = cp.stack(V1_bar_log_gpu, axis=0).get()
            U2_bar = cp.stack(U2_bar_log_gpu, axis=0).get()
            V2_bar = cp.stack(V2_bar_log_gpu, axis=0).get()
            A1_re = cp.stack(A1_tilde_re_gpu, axis=0).get()
            A1_im = cp.stack(A1_tilde_im_gpu, axis=0).get()
            A2_re = cp.stack(A2_tilde_re_gpu, axis=0).get()
            A2_im = cp.stack(A2_tilde_im_gpu, axis=0).get()
            diagnostics = {
                "time": np.asarray(moments_time, dtype=np.float64),
                "U1_bar": U1_bar.astype(np.float64, copy=False),
                "V1_bar": V1_bar.astype(np.float64, copy=False),
                "U2_bar": U2_bar.astype(np.float64, copy=False),
                "V2_bar": V2_bar.astype(np.float64, copy=False),
                "A1_tilde": (A1_re + 1j * A1_im).astype(np.complex128, copy=False),
                "A2_tilde": (A2_re + 1j * A2_im).astype(np.complex128, copy=False),
            }
        else:
            diagnostics = {
                "time": np.empty(0, dtype=np.float64),
                "U1_bar": np.empty(0, dtype=np.float64),
                "V1_bar": np.empty(0, dtype=np.float64),
                "U2_bar": np.empty(0, dtype=np.float64),
                "V2_bar": np.empty(0, dtype=np.float64),
                "A1_tilde": np.empty(0, dtype=np.complex128),
                "A2_tilde": np.empty(0, dtype=np.complex128),
            }

    if PROFILE_SPDE and iteration > 0:
        total_ms = kernel_elapsed_ms + moment_elapsed_ms
        profile_msg = (f"[profile] spde+noise: {kernel_elapsed_ms/iteration:.3f} ms/iter")
        if collect_moments and moment_calls > 0:
            profile_msg += (f", moments: {moment_elapsed_ms/moment_calls:.3f} ms/call"
                            f" (~{moment_elapsed_ms/iteration:.3f} ms/iter)")
        profile_msg += f", total: {total_ms/iteration:.3f} ms/iter"
        print(profile_msg, flush=True)

    # Return CPU arrays to the caller.
    return (cp.asnumpy(U1), cp.asnumpy(V1)), (cp.asnumpy(U2), cp.asnumpy(V2)), diagnostics







class Coupled_GrayScottModel(GrayScottModel):
    """ 
    Equation (2) in Yadome PRE 2011
        ut = ∆u + u^2v - (F + k)u,
        vt = d*∆vxx - u^2v + F (1 - v).

    The parameter values are set to F = 0.018,k = 0.052, for which the kinetics part of Eq. (2) is excitable (Fig. 2). Periodic boundary conditions are used for Eq. (2), and the system size L = 250 is sufficiently larger than the typical size of a pulse solution. The spatial grid size is 512 and the time increment is 0.1.
                    [Yadome PRE (2011)]

    where u and v are the concentrations of the two species, 
    D_u and D_v are the diffusion coefficients, 
    F is the feed rate, and k is the kill rate.
    """

    def __init__(self, paramdict=None, display=True):
        super().__init__(paramdict=paramdict, display=display)

        # Apply coupling and noise overrides.
        if type(paramdict) is dict:
                if 'epsilon' in paramdict:
                    self.epsilon = np.float64(paramdict['epsilon'])

                if "sigma" in paramdict:
                    self.sigma = np.float64(paramdict['sigma'])


        else:
            pass

        if display == True:
            print("epsilon: {:0.4e}".format(self.epsilon))
            print("sigma: {:0.4e}".format(self.sigma))






    def spde_euler(self, init_x1: tuple[np.ndarray, np.ndarray], 
                        init_x2: tuple[np.ndarray, np.ndarray],
                        dt: float, iteration: int,
                        c:float=float(0.0),
                        log_dt: Optional[float] = None,
                        start_time: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """ 
        Solve the coupled SPDE with Euler-Maruyama.

        Parameters:
        init_x : tuple of 1d-ndarray [u, v] 
        tspan : 1d-ndarray
        step : int, optional
            Save results every `step` iterations (default=10)

        Parameters
        ----------
        log_dt : float | None, optional
            Moment-sampling interval. ``None`` disables diagnostic logging.

        Returns
        -------
        (U1, V1), (U2, V2): tuple[np.ndarray, np.ndarray]
            Fields at the final time.
        diagnostics : dict | None
            Sampled means and complex amplitudes when ``log_dt`` is provided.
        """

        if (init_x1[0].size == self.gridnum) and (init_x1[1].size == self.gridnum) and \
              (init_x2[0].size == self.gridnum) and (init_x2[1].size == self.gridnum):
            pass
        else:
            print("Error: init_x1 and init_x2 must be 1d-ndarray with size of gridnum")
            return None
        

        u1_init, v1_init = init_x1
        u2_init, v2_init = init_x2
        D = self.D
        h = self.h
        epsilon = self.epsilon
        sigma = self.sigma
        params = self.params
        c = cp.float64(c)

        # Remove the duplicated periodic endpoint before GPU integration.
        u1_init = u1_init[:-1]
        v1_init = v1_init[:-1]
        u2_init = u2_init[:-1]
        v2_init = v2_init[:-1]
        rx_grid = self.rx[:-1]

        # Transfer the initial fields to the GPU.
        u1_init, v1_init = cp.asarray(u1_init), cp.asarray(v1_init)
        u2_init, v2_init = cp.asarray(u2_init), cp.asarray(v2_init)

        (U1, V1), (U2, V2), diagnostics = Euler_Maruyama((u1_init, v1_init),
                                   (u2_init, v2_init),
                                    rx_grid,
                                    epsilon, sigma, D, h, c, params, dt, iteration,
                                    log_dt=log_dt,
                                    start_time=start_time)
        
        # Restore the duplicated periodic endpoint.
        U1 = np.concatenate((U1, U1[:1]))
        V1 = np.concatenate((V1, V1[:1]))
        U2 = np.concatenate((U2, U2[:1]))
        V2 = np.concatenate((V2, V2[:1]))
        return (U1, V1), (U2, V2), diagnostics



# def get_grid_property():
#     a = np.load(loadfilename)
#     T = np.ptp(a["t"])
#     dtheta = (2.0 * np.pi) * (a["t"][1] - a["t"][0]) / T 
#     return {
#         "shape": (a["t"].size, a["rx"].size),
#         "dtheta": float(dtheta),
#         "dx": float(a["rx"][1] - a["rx"][0]),
#             }



def get_init_condition(lateral_diff:int = 0, vertical_diff:int = 0):
    """ 
    Return two initial states separated by spatial and temporal grid offsets.

    For field-array dimensions ``[a + 1, b + 1]``:
        delta_phi = (L/a) * lateral_diff
        delta_theta = (T/b) * vertical_diff
    """

    loadfilename = "ReferenceData/Make_X0_env/JITVersion/X0data_withPhi0.npz"

    with np.load(loadfilename, allow_pickle=False) as data:
        U = np.asarray(data["U"])
        V = np.asarray(data["V"])
        tspan = np.asarray(data["t"])
        rx = np.asarray(data["rx"])
    #* tspan: 0~T
    #* rx: -L~L

    T = np.ptp(tspan)
    L = np.ptp(rx)

    lateral_diff = lateral_diff % (rx.size - 1)
    vertical_diff = vertical_diff % (tspan.size - 1)

    # Use the reference state for the second oscillator.
    U2, V2 = U[0], V[0]

    # Select the temporal phase of the first oscillator.
    U1 = np.zeros_like(U2)
    V1 = np.zeros_like(V2)
    U1_temp, V1_temp = U[vertical_diff], V[vertical_diff]

    # Apply the spatial phase shift.
    U1[:-1] = np.roll(U1_temp[:-1], lateral_diff)
    V1[:-1] = np.roll(V1_temp[:-1], lateral_diff)
    U1[-1], V1[-1] = U1[0], V1[0]

    #*
    delta_phi = (rx[lateral_diff] - rx[0])
    delta_theta = (2.0*np.pi) * (tspan[vertical_diff] - tspan[0]) / T

    #* mod
    delta_phi = np.mod(delta_phi, L)
    delta_theta = np.mod(delta_theta, 2.0*np.pi)

    print("delta_phi: {:0.4f}".format(delta_phi))
    print("delta_theta: {:0.4f}".format(delta_theta))
    return (U1, V1), (U2, V2), delta_phi, delta_theta
