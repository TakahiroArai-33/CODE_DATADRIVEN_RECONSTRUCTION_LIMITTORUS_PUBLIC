#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import cupy as cp
from typing import Tuple, List, Callable, Literal
from pathlib import Path
import os, gc, time


# Set to False to evaluate selected tensor operations on the GPU.
use_cpu_in_update = False



def pick_L():
    """Calculate L from one intermediate-data archive."""
    import  Module_get_envname
    INTERIM_DIR = Module_get_envname.INTERIM_DIR

    # Directory containing candidate .npz files
    data_dir = Path(INTERIM_DIR)

    # Optional: filter by keyword via env var NPZ_KEYWORD (e.g., "state410")
    keyword = os.environ.get("NPZ_KEYWORD", None)

    # Collect candidates
    candidates = sorted(
        [p for p in data_dir.glob("*.npz") if (keyword is None or keyword in p.name)]
    )

    if not candidates:
        raise FileNotFoundError(f"No .npz files found in {data_dir} matching keyword={keyword!r}")

    # Selection strategy:
    # - If NPZ_LATEST is truthy (default), pick the newest file.
    # - Else, use NPZ_INDEX (default 0) to pick by index in sorted list.
    if os.environ.get("NPZ_LATEST", "1").lower() in ("1", "true", "yes"):
        selected = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        idx = int(os.environ.get("NPZ_INDEX", "0"))
        selected = candidates[idx % len(candidates)]


    print("------------------------------------------------", flush=True)
    print("Calculating L from the following file.", flush=True)
    print(f"Loading NPZ: {selected}", flush=True)
    print("------------------------------------------------", flush=True)
    xxxx = np.load(selected)
    return np.ptp(xxxx["rx"]) / 2.0

#* PICK L
params = {"L": pick_L()}
#* ---------------------------------------





class BayesianEstimator:
    def __init__(self, Ms: int = 30, Mt: int = 5, 
                 params: dict = params):
        r"""
        arguments:
        Ms, Mt:
            g_\boldsymbol{m}(\Phi_i - \Phi_j, \Theta_i - \Theta_j)
            := \exp \left[ \mathrm{i} \left(
                m_s \frac{\Phi_i - \Phi_j}{2L} + m_t \frac{\Theta_i - \Theta_j}{2\pi}
            \right)\right],
            \boldsymbol{m} = (m_s, m_t)
            Ms and Mt specify the maximum basis orders in each direction.
        params: dict
        params["L"]: float, half the spatial domain length.
        params["complex64"]: bool, store G as complex64 when true.
        """
        self.Ms = Ms
        self.Mt = Mt
        self.L = params["L"]  # Half the spatial domain length
        self.rtol = float(params.get("rtol", 1e-5))  # Relative convergence tolerance
        self.complex64 = bool(params.get("complex64", False))
        self.gpu_chunk = int(params.get("gpu_chunk", 20))
        
        # Estimate a diagonal phase-noise covariance matrix when true.
        self.noise_cov_zero = bool(params.get("noise_cov_zero", False)) 

        if self.noise_cov_zero:
            print("Estimator check: using a diagonal phase-noise covariance matrix.")
        else:
            print("Estimator check: using a full phase-noise covariance matrix.")


        if self.complex64:
            print("Storing tensor G as complex64.")

        self._m_profile = []  # Fourier-index pairs (ms, mt)
        for m_s in range(-Ms, Ms + 1):
            for m_t in range(-Mt, Mt + 1):
                self._m_profile.append((m_s, m_t))
        self.R = int(len(self._m_profile))  # Number of basis functions

        #* ------ prior_condition ---------
        self.a_prior = np.zeros((2*self.R, ), dtype=np.complex128)  #* size(2R,)
        
        #* (Sigma^{-1} = Xi_prior)
        # self.Xi_prior = np.zeros((2*self.R, 2*self.R), dtype=np.complex128)
        self.Xi_prior = (1.0e-6) * np.eye(2*self.R, dtype=np.complex128)  #* # prior with L2 norm 

        self.E_prior = np.eye(2, dtype=np.float64)  #* size(2, 2)

        #* ----- init_condition ------------
        self.a = self.a_prior.copy()
        self.Xi = self.Xi_prior.copy()
        self.E = self.E_prior.copy()



   
    def func_g(self, ms: int, mt: int, Delta_Phi: np.ndarray, Delta_Theta: np.ndarray) -> np.ndarray:
        """ 
        Evaluate one Fourier basis function.
        ms, mt: int
        Delta_Phi: ndarray or float
        Delta_Theta: ndarray or float
        """
        L = self.L
        phase = (ms * Delta_Phi / (2.0 * L) * (2.0 * np.pi) + mt * Delta_Theta)
        gm = np.exp(1j * phase)
        return gm


    
    def coef_partial_g(self, ms: int, mt: int, key:Literal["s", "t"]):
        """
        Return the coefficient for a Phi or Theta derivative of g.
        ms, mt: int
        key: Literal["s", "t"]
            s: multiply by 1j*ms*(np.pi/L) for a Phi derivative.
            t: multiply by 1j*mt for a Theta derivative.
        returns:
            coef: complex128
        """
        L = self.L
        if key == "s":
            return 1j*ms*(np.pi/L)
        if key == "t":
            return 1j*mt




    def _cal_G(self, Delta_Phi_ast: np.ndarray,
                     Delta_Theta_ast: np.ndarray,
                    mode: Literal["partial", "normal"] = "normal") -> np.ndarray:
        r"""
        Construct the time series of block matrices G_n.
        G = (g^T, 0 \\
             0, g^T)
        g = {g_\boldsymbol{m}(\Phi_i - \Phi_j, \Theta_i - \Theta_j)}_\boldsymbol{m}
        g_\boldsymbol{m}(\Phi_i - \Phi_j, \Theta_i - \Theta_j)
        := \exp \left[ \mathrm{i} \left(
            m_s \frac{\Phi_i - \Phi_j}{2L} + m_t \frac{\Theta_i - \Theta_j}{2\pi}
        \right)\right],
        is evaluated for every sample.
        arguments:
            Delta_Phi_ast [ndarray]: midpoint differences Phi_i^* - Phi_j^*.
            Delta_Theta_ast [ndarray]: midpoint differences Theta_i^* - Theta_j^*.

            mode: str
                "partial": construct partial G.
                "normal": construct G.
        returns:
            G [ndarray]: (2, 2R, N)
                G[:, :, n] corresponds to G_n; R is self.R.
        """

        L = self.L
        R = self.R

        # Number of samples.
        N = int(Delta_Phi_ast.size)

        # Validate input sizes.
        assert Delta_Phi_ast.size == N, "Delta_Phi_ast size mismatch."
        assert Delta_Theta_ast.size == N, "Delta_Theta_ast size mismatch."

        #* Phi, Theta
        Delta_Phi = np.mod(Delta_Phi_ast, 2.0*L) # size(N)
        Delta_Theta = np.mod(Delta_Theta_ast, 2.0*np.pi) # size(N)

        # Coefficients for partial_s G_n and partial_t G_n.
        coef_s = []
        coef_t = []
        for ms, mt in self._m_profile:
            coef_s.append(self.coef_partial_g(ms, mt, key="s"))
            coef_t.append(self.coef_partial_g(ms, mt, key="t"))
        coef_s = np.array(coef_s, dtype=np.complex128)
        coef_t = np.array(coef_t, dtype=np.complex128)


        # Construct G_tensor.
        if self.complex64:
            G_tensor = np.zeros([N, 2, 2*R], dtype=np.complex64)
        else:
            G_tensor = np.zeros([N, 2, 2*R], dtype=np.complex128)

        # Pre-zero unaffected blocks once (avoid repeated writes inside the loop)
        if mode == "normal":
            # first row's second block (s-row, t-block) is zero
            G_tensor[:, 0, R:] = 0.0
            # second row's first block (t-row, s-block) is zero
            G_tensor[:, 1, :R] = 0.0
        elif mode == "partial":
            # partial_s occupies only the s block; partial_t only the t block.
            G_tensor[:, 0, R:] = 0.0  # s-row, t-block
            G_tensor[:, 1, :R] = 0.0  # t-row, s-block

        # Fill each active Fourier-mode column.
        for m_idx, (ms, mt) in enumerate(self._m_profile):
            gm_array = self.func_g(ms, mt, Delta_Phi, Delta_Theta).ravel()
            cs = coef_s[m_idx]
            ct = coef_t[m_idx]
            if mode == "normal":
                # Fill only the active columns for s-row and t-row
                G_tensor[:, 0, m_idx]   = gm_array   # s row, s block
                G_tensor[:, 1, m_idx+R] = gm_array   # t row, t block
            elif mode == "partial":
                # Apply derivative coefficients to the corresponding blocks.
                G_tensor[:, 0, m_idx]   = cs * gm_array  
                G_tensor[:, 1, m_idx+R] = ct * gm_array    
            gc.collect()

        return np.transpose(G_tensor, (1, 2, 0))  # Shape: (2, 2R, N)


    def create_G_dotchi(self,
                        Phi_i_ast: np.ndarray, 
                        Phi_j_ast: np.ndarray,
                        Theta_i_ast: np.ndarray,
                        Theta_j_ast: np.ndarray,
                        dot_Phi_i: np.ndarray,
                        dot_Phi_j: np.ndarray,
                        dot_Theta_i: np.ndarray,
                        dot_Theta_j: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Construct G, partial_G, and dot_chi.
        """

        assert Phi_i_ast.shape == Phi_j_ast.shape == Theta_i_ast.shape == Theta_j_ast.shape \
            == dot_Phi_i.shape == dot_Phi_j.shape == dot_Theta_i.shape == dot_Theta_j.shape, \
                f"All input arrays must have the same shape."

        #* size: (2,N)
        dot_chi = np.array([dot_Phi_i.ravel(),
                            dot_Theta_i.ravel()]) #* size(2, N)
        #* size: (2, 2R, N)
        Delta_Phi_ast = np.mod(Phi_i_ast - Phi_j_ast, 2.0*self.L)
        Delta_Theta_ast = np.mod(Theta_i_ast - Theta_j_ast, 2.0*np.pi)
        G_normal = self._cal_G(Delta_Phi_ast=Delta_Phi_ast, Delta_Theta_ast=Delta_Theta_ast, 
                               mode="normal")  #* size(2, 2R, N)
        G_partial = self._cal_G(Delta_Phi_ast=Delta_Phi_ast, Delta_Theta_ast=Delta_Theta_ast, 
                                mode="partial")  #* size(2, 2R, N)

        # Report the memory required by G.
        G_datasize = G_normal.nbytes / (1024 ** 3)
        print(f"G data size: {G_datasize:.3f} GB (shape={G_normal.shape}, dtype={G_normal.dtype})",
              flush=True)

        return {
            "G_normal": G_normal, "G_partial": G_partial, "dot_chi": dot_chi
            }



    def update(self, datas: dict) -> None:
        """ 
        Perform one MAP-estimator update.
        
        arguments:
            datas: dict of three ndarrays
                G_normal: ndarray, size(2, 2R, N)
                G_partial: ndarray, size(2, 2R, N)
                dot_chi: ndarray, size(2, N)
        """

        # Delta_t and N are required to evaluate the update equations.
        if not hasattr(self, "Delta_t") or not hasattr(self, "N"):
            raise AttributeError(
                "Delta_t or N not set. Call create_G_dotchi() before update()."
            )
        
        Delta_t = self.Delta_t
        N = self.N

        # Retrieve prepared tensors.
        G_normal = datas["G_normal"] # size(2, 2R, N)
        G_partial = datas["G_partial"] # size(2, 2R, N)
        dot_chi = datas["dot_chi"] # size(2, N)

        #* --- Tool: Hermitian (conjugate) transpose ---
        G_normal_H = np.conj(np.transpose(G_normal, (1, 0, 2)))  # size (2R, 2, N)
        G_partial_H = np.conj(np.transpose(G_partial, (1, 0, 2)))  # size (2R, 2, N)


        # Validate tensor dimensions against N and R.
        if  G_partial.shape[2] != N or G_normal.shape[2] != N or dot_chi.shape[1] != N:
            raise ValueError("Inconsistent time dimension between stored data and N.")
        if G_partial.shape[1] != int(2*self.R) or G_normal.shape[1] != int(2*self.R):
            raise ValueError("Inconsistent spatial dimension in G_partial.")

        #* Keep copies of current estimates (old values) for output
        a_old = self.a.copy()
        # Xi_old = self.Xi.copy() # Not used in update
        # E_old = self.E.copy() # Not used in update

        #* copy prior values
        a_prior = self.a_prior.copy()
        Xi_prior = self.Xi_prior.copy()
        # E_prior = self.E_prior.copy() # Not used in update

        #* --- Residuals for E update ---
        # G:size (2, 2R, N), a_old:size(2R,)
        # Evaluate the tensor product on the CPU when requested.
        if use_cpu_in_update:
            Ga = np.tensordot(G_normal, a_old, axes=([1], [0]))  # shape (2, N)
            residual = (dot_chi - Ga).real

        # Otherwise evaluate the tensor product on the GPU in chunks.
        else:
            array_gpu = cp.zeros((2, N), dtype=cp.complex128)
            a_old_gpu = cp.asarray(a_old)
            for i in range(self.gpu_chunk):
                # Current GPU chunk bounds.
                s = int(i * N / self.gpu_chunk)
                e = int((i + 1) * N / self.gpu_chunk)
                #*
                G_normal_gpu = cp.asarray(G_normal[:, :, s:e])
                array_gpu[:, s:e] = cp.tensordot(G_normal_gpu, a_old_gpu, axes=([1], [0]))
            Ga = cp.asnumpy(array_gpu)
            residual = (dot_chi - Ga).real  # Real-valued residual, shape (2, N)
            del array_gpu, a_old_gpu, G_normal_gpu


        #* --- E update ---
        if self.noise_cov_zero:
            # Compute only diagonal covariance entries.
            E_new_complex = (Delta_t / N) * np.diag(np.sum(residual**2, axis=1))
            E_new = E_new_complex.real.astype(np.float64)  # theoretical E is real-valued

        else:
            # The residual is real, so the adjoint reduces to a transpose.
            E_new_complex = (Delta_t / N) * (residual @ residual.T)
            # E_new = E_new_complex.real  # theoretical E is real-valued
            E_new = ((E_new_complex + E_new_complex.T)/2).real.astype(np.float64) # theoretical E is real-valued

        #* --- Xi update (vectorized over n) ---
        # E: size (2, 2)
        # G: size (2, 2R, N)
        # Apply E_inv to every G_n in one batched tensor product.
        Einv = np.linalg.inv(E_new)
        Einv_G = np.tensordot(Einv, G_normal, axes=([1], [0]))  # size (2, 2R, N)
        # Sum G_n^H (E_inv G_n) over n with einsum.

        # Evaluate einsum on the CPU when requested.
        if use_cpu_in_update: 
            Xi_new = Xi_prior + Delta_t * np.einsum('kin,iqn->kq', G_normal_H, Einv_G, optimize=True)
        # Otherwise evaluate einsum on the GPU in chunks.
        else: 
            array_gpu = cp.zeros_like(Xi_prior, dtype=np.complex128)
            for i in range(self.gpu_chunk):
                # Current GPU chunk bounds.
                s = int(i * N / self.gpu_chunk)
                e = int((i + 1) * N / self.gpu_chunk)
                #* 
                G_normal_H_gpu = cp.asarray(G_normal_H[:, :, s:e])
                Einv_G_gpu = cp.asarray(Einv_G[:, :, s:e])
                array_gpu += cp.einsum('kin,iqn->kq', G_normal_H_gpu, Einv_G_gpu, optimize=True)
            Xi_new = Xi_prior + Delta_t * cp.asnumpy(array_gpu)
            del array_gpu, G_normal_H_gpu, Einv_G_gpu

        # Enforce Hermitian symmetry against numerical roundoff.
        Xi_new = (Xi_new + Xi_new.conj().T) / 2

        #* --- r calculate (vectorized over n) ---
        e_vec = np.ones(2, dtype=np.float64)
        # E: size (2, 2)
        # dot_chi: size (2, N)
        # G: size (2, 2R, N)
        E_inv_dotchi = Einv @ dot_chi  # size (2, N)

        # Evaluate einsum on the CPU when requested.
        if use_cpu_in_update: 
            # Evaluate the sum of G_n^H (E_inv dot_chi_n).
            sum_GH_Einv_dotchi = np.einsum('kin,in->k', G_normal_H, E_inv_dotchi, optimize=True)
            # Evaluate the sum of [partial G_n]^H e.
            sum_partial = np.einsum('kin,i->k', G_partial_H, e_vec, optimize=True)

        # Otherwise evaluate einsum on the GPU in chunks.
        else: 
            e_vec_gpu = cp.asarray(e_vec)
            sum_GH_Einv_dotchi_gpu = cp.zeros_like(a_old, dtype=np.complex128)
            sum_partial_gpu = cp.zeros_like(a_old, dtype=np.complex128)
            for i in range(self.gpu_chunk):
                # Current GPU chunk bounds.
                s = int(i * N / self.gpu_chunk)
                e = int((i + 1) * N / self.gpu_chunk)
                #* 
                G_normal_H_gpu = cp.asarray(G_normal_H[:, :, s:e])
                E_inv_dotchi_gpu = cp.asarray(E_inv_dotchi[:, s:e])
                sum_GH_Einv_dotchi_gpu += cp.einsum('kin,in->k', G_normal_H_gpu, E_inv_dotchi_gpu, optimize=True)
                #* 
                G_partial_H_gpu = cp.asarray(G_partial_H[:, :, s:e])
                sum_partial_gpu += cp.einsum('kin,i->k', G_partial_H_gpu, e_vec_gpu, optimize=True)
            sum_GH_Einv_dotchi = cp.asnumpy(sum_GH_Einv_dotchi_gpu)
            sum_partial = cp.asnumpy(sum_partial_gpu)
            del e_vec_gpu, G_normal_H_gpu, E_inv_dotchi_gpu, G_partial_H_gpu

        # Calculate r using the fixed prior mean.
        r = Xi_prior @ a_prior + Delta_t * sum_GH_Einv_dotchi - (Delta_t / 2.0) * sum_partial

        #* --- a update (Xi @ a = r ↔ a = Xi^{-1} @ r)---
        a_new = np.linalg.solve(Xi_new, r)  # Solve Xi_new @ a_new = r


        # Store posterior estimates
        self.a = a_new.copy()
        self.Xi = Xi_new.copy()
        self.E = E_new.copy()


    def map_method(self, Phi_i_ast: np.ndarray, Phi_j_ast: np.ndarray,
                       Theta_i_ast: np.ndarray, Theta_j_ast: np.ndarray,
                       dot_Phi_i: np.ndarray, dot_Phi_j: np.ndarray,
                       dot_Theta_i: np.ndarray, dot_Theta_j: np.ndarray,
                       Delta_t: float,
                       num_iterations: int = 100) -> None:
        """ 
        Construct G and dot_chi, then iterate the Bayesian update.
        arguments:
            Phi_i_ast: ndarray
            Phi_j_ast: ndarray
            Theta_i_ast: ndarray
            Theta_j_ast: ndarray
                Phase values at time-step midpoints.
            dot_Phi_i: ndarray
            dot_Phi_j: ndarray
            dot_Theta_i: ndarray
            dot_Theta_j: ndarray
                Time derivatives of the phase data.
            Delta_t: float
                Time-step size.
            num_iterations: int
                Maximum number of updates.
        """

        
        datas = self.create_G_dotchi(
                                     Phi_i_ast=Phi_i_ast, Phi_j_ast=Phi_j_ast,
                                     Theta_i_ast=Theta_i_ast, Theta_j_ast=Theta_j_ast,
                                     dot_Phi_i=dot_Phi_i, dot_Phi_j=dot_Phi_j,
                                     dot_Theta_i=dot_Theta_i, dot_Theta_j=dot_Theta_j
                                     )
        
        self.Delta_t = Delta_t
        self.N = Phi_i_ast.shape[0]

        a_list = []
        Xi_list = []
        E_list = []
        error_list = []  # No error value exists for the initial state.

        for i in range(1, num_iterations+1):

            ts_upd = time.time()

            a_old = self.a.copy()
            Xi_old = self.Xi.copy()
            E_old = self.E.copy()

            # Report conditioning before each update.
            lamE  = np.linalg.eigvalsh(E_old)              # 2x2
            lamXi = np.linalg.eigvalsh(Xi_old.real)        # 2R x 2R
            print(f"Before update {i:d}: parameter check")
            print("min λ(E):", lamE.min(), "  min λ(Xi):", lamXi.min(),
                "  cond(Xi):", np.linalg.cond(Xi_old.real), "  ||a||:", np.linalg.norm(a_old))

            if i == 1:
                # Store the initial state.
                a_list.append(a_old)
                Xi_list.append(Xi_old)
                E_list.append(E_old)

            self.update(datas)

            a_new = self.a.copy()
            Xi_new = self.Xi.copy()
            E_new = self.E.copy()

            # Store the updated state.
            a_list.append(a_new)
            Xi_list.append(Xi_new)
            E_list.append(E_new)

            # Use the relative change in a as the convergence criterion.
            denom = np.linalg.norm(a_old)
            rel_error_a = np.linalg.norm(a_new - a_old) / max(denom, 1e-8)
            print(f"Iteration {i}: Relative error in a = {rel_error_a:.6e}", flush=True)
            error_list.append(rel_error_a)

            te_upd = time.time()
            print(f"Update time: {te_upd - ts_upd:.3f} sec", flush=True)

            if rel_error_a < self.rtol:
                print("Convergence achieved.", flush=True)
                break

        else:
            final_error = error_list[-1] if error_list else np.nan
            print(f"Reached {num_iterations} iterations without convergence (final relative error: {final_error:.6e}).", flush=True)

        print("Pass the returned estimator to OutputFunction to evaluate results.", flush=True)
        return self, error_list





class OutputFunction:
    def __init__(self, instance: BayesianEstimator):
        # Store posterior parameters.
        self.a = instance.a
        self.Xi = instance.Xi
        self.E = instance.E
        self.Sigma = np.linalg.inv(self.Xi)

        #* 
        self.a_prior = instance.a_prior
        self.Xi_prior = instance.Xi_prior
        self.E_prior = instance.E_prior

    
        self.N = instance.N
        self.Delta_t = instance.Delta_t
        self.L = instance.L

        self.R = instance.R
        self._m_profile = instance._m_profile

        # Bind the estimator's basis-function implementation.
        self._func_g = type(instance).func_g.__get__(instance, type(instance))
    

    def func_g(self, ms: int, mt: int, Delta_Phi: np.ndarray, Delta_Theta: np.ndarray) -> np.ndarray:
        """ 
        ms, mt: int
        Delta_Phi: ndarray
        Delta_Theta: ndarray
        """
        return self._func_g(ms, mt, Delta_Phi, Delta_Theta)



    def phase_equation(self, XX: np.ndarray, YY: np.ndarray,
                       key: Literal["normal", "const", "nonconst"]="normal") -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Evaluate the inferred phase equation,
            sum_m, a_m^s g_m(XX, YY), 
            and sum_m a_m^t g_m(XX, YY).
        XX: Delta_Phi values as a 1-D or 2-D array.
        YY: Delta_Theta values as a 1-D or 2-D array.
        key: Literal["normal", "const", "nonconst"]
            "normal": full phase equation.
            "const": constant term only.
            "nonconst": nonconstant terms only.
        return:
            ZZ_s: inferred mean of the s component.
            ZZ_t: inferred mean of the t component.
            ZZ_s_var: inferred variance of the s component.
            ZZ_t_var: inferred variance of the t component.
        """

        # Convert 1-D coordinates to a mesh grid.
        if XX.ndim == 1 and YY.ndim == 1:
            _XX, _YY = np.meshgrid(XX, YY)

        # Use matching 2-D coordinates directly.
        elif XX.ndim == 2 and YY.ndim == 2 and XX.shape == YY.shape:
            _XX, _YY = XX, YY

        # Physical means and variances are real-valued.
        ZZ_s_mean = np.zeros_like(_XX, dtype=np.float64)
        ZZ_t_mean = np.zeros_like(_YY, dtype=np.float64)
        ZZ_s_var = np.zeros_like(_XX, dtype=np.float64)
        ZZ_t_var = np.zeros_like(_YY, dtype=np.float64)
        
        R = self.R
        m_profile = self._m_profile
        #* mean
        as_vec = self.a[:R]
        at_vec = self.a[R:]
        #* covariance
        Sigma_ss = self.Sigma[:R, :R]
        Sigma_st = self.Sigma[:R, R:]
        Sigma_ts = self.Sigma[R:, :R]
        Sigma_tt = self.Sigma[R:, R:]

        # Construct Fourier-index vectors.
        ms_vec = np.fromiter((ms for ms, _ in m_profile), dtype=np.int32, count=R)
        mt_vec = np.fromiter((mt for _, mt in m_profile), dtype=np.int32, count=R)

        # Locate the constant Fourier mode.
        m0_idx = next(i for i, (ms, mt) in enumerate(m_profile) if ms == 0 and mt == 0)

        # Vectorize basis evaluation over Fourier modes.
        ms_grid = ms_vec[:, None, None]
        mt_grid = mt_vec[:, None, None] 
        g_tensor = self.func_g(ms_grid, mt_grid, _XX, _YY)              # (R, M, N)


        # Evaluate the requested phase-equation component on the GPU.
        as_gpu = cp.asarray(as_vec)  # Shape: (R,)
        at_gpu = cp.asarray(at_vec) #* size(R,)
        Sigma_ss_gpu = cp.asarray(Sigma_ss) #* size(R, R)
        Sigma_tt_gpu = cp.asarray(Sigma_tt) #* size(R, R)
        g_tensor = cp.asarray(g_tensor)  # Basis values, shape (R, M, N)


        if key=="normal":
            # Keep all Fourier modes.
            pass

        elif key=="const":
            # Zero all coefficients except the constant mode.
            as_gpu.fill(0)
            at_gpu.fill(0)
            as_gpu[m0_idx] = as_vec[m0_idx]
            at_gpu[m0_idx] = at_vec[m0_idx]

            Sigma_ss_gpu.fill(0)
            Sigma_tt_gpu.fill(0)
            Sigma_ss_gpu[m0_idx, m0_idx] = Sigma_ss[m0_idx, m0_idx]
            Sigma_tt_gpu[m0_idx, m0_idx] = Sigma_tt[m0_idx, m0_idx]

        elif key=="nonconst":
            # Remove the constant mode.
            as_gpu[m0_idx] = 0
            at_gpu[m0_idx] = 0
            Sigma_ss_gpu[m0_idx, :] = 0
            Sigma_ss_gpu[:, m0_idx] = 0
            Sigma_tt_gpu[m0_idx, :] = 0
            Sigma_tt_gpu[:, m0_idx] = 0
        

        #* ZZ_s, ZZ_t: size (M, N)
        ZZ_s_mean = cp.asnumpy(cp.real(cp.tensordot(as_gpu, g_tensor, axes=(0, 0)))).astype(np.float64)
        ZZ_t_mean = cp.asnumpy(cp.real(cp.tensordot(at_gpu, g_tensor, axes=(0, 0)))).astype(np.float64)

        #* var_s, var_t: size (M, N)
        M, N = _XX.shape
        var_s = cp.zeros((M, N), dtype=cp.complex128)
        var_t = cp.zeros((M, N), dtype=cp.complex128)
        for row in range(M):
            g_row = g_tensor[:, row, :]  #* (R, N)
            var_s[row,:] = cp.einsum('im,ij,jm->m',
                                  cp.conjugate(g_row), Sigma_ss_gpu, g_row)
            var_t[row,:] = cp.einsum('im,ij,jm->m',
                                  cp.conjugate(g_row), Sigma_tt_gpu, g_row)
        var_s = cp.asnumpy(cp.real(var_s))
        var_t = cp.asnumpy(cp.real(var_t))

        # Equivalent fully vectorized einsum implementation:
        # var_s = cp.asnumpy(cp.real(cp.einsum('imn,ij,jmn->mn',
        #                             cp.conjugate(g_tensor), Sigma_ss_gpu, g_tensor)
        #                             ))
        # var_t = cp.asnumpy(cp.real(cp.einsum('imn,ij,jmn->mn',
        #                             cp.conjugate(g_tensor),Sigma_tt_gpu, g_tensor)
        #                             ))
        #----------------------------------------

        # Clip small negative variances caused by floating-point roundoff.
        ZZ_s_var = np.maximum(0.0, np.real(var_s)).astype(np.float64)
        ZZ_t_var = np.maximum(0.0, np.real(var_t)).astype(np.float64)


        print("Returning the MAP estimate of the phase equation.")
        return ZZ_s_mean, ZZ_t_mean, ZZ_s_var, ZZ_t_var
    



    def get_noise_covariance(self, colesky: bool = False) -> np.ndarray:
        """ 
        Return the inferred noise covariance matrix E.
        """ 
        if colesky:
            return np.linalg.cholesky(self.E)
        else: 
            return self.E







