
import numpy as np
import cupy as cp
from typing import Tuple, List, Callable, Literal
from pathlib import Path
import os, gc
import Module_Bayes_GPU as Module_Bayes  
import time



L = Module_Bayes.params["L"]

"""Prepare chi* and dot(chi) samples for Bayesian inference."""

randseed = 415643

class ArrangeData:
    """Prepare and resample phase-space data for Bayesian inference."""
    def __init__(self):
        pass

    def make_data(self, Phi1_list: List[np.ndarray], 
                  Phi2_list: List[np.ndarray],
                  Theta1_list: List[np.ndarray],
                  Theta2_list: List[np.ndarray],
                  Delta_t: float
                  ):
        """
        Construct midpoint samples chi* and forward differences dot(chi).

        Parameters
        ----------
        Phi1_list : List[np.ndarray]
            Spatial-phase trajectories for system 1.
        Phi2_list : List[np.ndarray]
            Spatial-phase trajectories for system 2.
        Theta1_list : List[np.ndarray]
            Temporal-phase trajectories for system 1.
        Theta2_list : List[np.ndarray]
            Temporal-phase trajectories for system 2.
        Delta_t : float
            Sampling interval.

        Returns
        -------
        dict1 : dict
            Dictionary containing the generated samples with keys
            'Phi1_ast', 'Phi2_ast', 'Theta1_ast', 'Theta2_ast',
                    'dot_Phi1', 'dot_Phi2', 'dot_Theta1', 'dot_Theta2'
        """

        num = len(Phi1_list) 
        assert num == len(Phi2_list) == len(Theta1_list) == len(Theta2_list), "Input data lists must have the same length."

        print(f"Preparing data from {num:d} trajectories.", flush=True)


        Phi_1_ast_list = []
        Phi_2_ast_list = []
        Theta_1_ast_list = []
        Theta_2_ast_list = []
        dot_Phi1_list = []
        dot_Phi2_list = []
        dot_Theta1_list = []
        dot_Theta2_list = []

        for i in range(num):
            # Unwrap each periodic phase coordinate.
            Phi1 = np.unwrap(np.mod(Phi1_list[i], 2.0*L), period=2.0*L)
            Phi2 = np.unwrap(np.mod(Phi2_list[i], 2.0*L), period=2.0*L)
            Theta1 = np.unwrap(np.mod(Theta1_list[i], 2.0*np.pi), period=2.0*np.pi)
            Theta2 = np.unwrap(np.mod(Theta2_list[i], 2.0*np.pi), period=2.0*np.pi)
            # Compute periodic phase values at time-step midpoints.
            Phi1_ast = np.mod(0.5 * (Phi1[:-1] + Phi1[1:]), 2.0*L)
            Phi2_ast = np.mod(0.5 * (Phi2[:-1] + Phi2[1:]), 2.0*L)
            Theta1_ast = np.mod(0.5 * (Theta1[:-1] + Theta1[1:]), 2.0*np.pi)
            Theta2_ast = np.mod(0.5 * (Theta2[:-1] + Theta2[1:]), 2.0*np.pi)
            # Compute forward differences.
            dot_Phi1 = (Phi1[1:] - Phi1[:-1]) / Delta_t
            dot_Phi2 = (Phi2[1:] - Phi2[:-1]) / Delta_t
            dot_Theta1 = (Theta1[1:] - Theta1[:-1]) / Delta_t
            dot_Theta2 = (Theta2[1:] - Theta2[:-1]) / Delta_t

            Phi_1_ast_list.append(Phi1_ast)
            Phi_2_ast_list.append(Phi2_ast)
            Theta_1_ast_list.append(Theta1_ast)
            Theta_2_ast_list.append(Theta2_ast)
            dot_Phi1_list.append(dot_Phi1)
            dot_Phi2_list.append(dot_Phi2)
            dot_Theta1_list.append(dot_Theta1)
            dot_Theta2_list.append(dot_Theta2)


        Phi1_ast = np.concatenate(Phi_1_ast_list)
        Phi2_ast = np.concatenate(Phi_2_ast_list)
        Theta1_ast = np.concatenate(Theta_1_ast_list)
        Theta2_ast = np.concatenate(Theta_2_ast_list)
        dot_Phi1 = np.concatenate(dot_Phi1_list)
        dot_Phi2 = np.concatenate(dot_Phi2_list)
        dot_Theta1 = np.concatenate(dot_Theta1_list)
        dot_Theta2 = np.concatenate(dot_Theta2_list)

        dict1 = {
            'Phi1_ast': Phi1_ast,
            'Phi2_ast': Phi2_ast,
            'Theta1_ast': Theta1_ast,
            'Theta2_ast': Theta2_ast,
            'dot_Phi1': dot_Phi1,
            'dot_Phi2': dot_Phi2,
            'dot_Theta1': dot_Theta1,
            'dot_Theta2': dot_Theta2,
        }
        return dict1



    def thin_data(self, data: dict, 
                  xx: np.ndarray, yy: np.ndarray, num: np.ndarray) -> dict:
        """
        Randomly resample data within each phase-space grid cell.

        Parameters
        ----------
        data: dict
            Data arrays to resample.
        xx, yy: np.ndarray(2d, float)
            Grid coordinates generated with meshgrid(indexing="xy").
        num: np.ndarray(2d, int)  size: [xx.shape[0]-1, xx.shape[1]-1]
            Number of samples to retain in each grid cell.

        Returns
        -------
        dict
            Resampled data arrays.
        """

        thinned_data = {}

        Phi1 = data['Phi1_ast']
        Phi2 = data['Phi2_ast']
        Theta1 = data['Theta1_ast']
        Theta2 = data['Theta2_ast'] 


        def wrap_to_negpos_L(x: np.ndarray) -> np.ndarray:
            """Wrap values from [0, 2L) to [-L, L)."""
            new_x = np.copy(x)
            new_x[new_x >= L] -= 2.0 * L
            return  new_x

        def wrap_to_negpos_pi(x: np.ndarray) -> np.ndarray:
            """Wrap angles from [0, 2pi) to [-pi, pi)."""
            new_x = np.copy(x)
            new_x[new_x >= np.pi] -= 2.0 * np.pi
            return new_x

        Delta_Phi = wrap_to_negpos_L(np.mod(Phi1 - Phi2, 2.0*L))
        Delta_Theta = wrap_to_negpos_pi(np.mod(Theta1 - Theta2, 2.0*np.pi))
        wrap_xx = wrap_to_negpos_L(np.mod(xx, 2.0*L))
        wrap_yy = wrap_to_negpos_pi(np.mod(yy, 2.0*np.pi))

        selected_indices_list = []
        print(f"Starting resampling; max(i), max(j): ({xx.shape[0]-1}, {xx.shape[1]-1})", flush=True)

        # Use deterministic sampling.
        np.random.seed(randseed)
        cp.random.seed(randseed)

        # Transfer phase differences and grid coordinates to the GPU.
        Delta_Phi_GPU = cp.asarray(Delta_Phi)
        Delta_Theta_GPU = cp.asarray(Delta_Theta)
        wrap_xx_GPU = cp.asarray(wrap_xx)
        wrap_yy_GPU = cp.asarray(wrap_yy)
        # xx_GPU = cp.asarray(xx)
        # yy_GPU = cp.asarray(yy)

        for i in range(xx.shape[0]-1):
            for j in range(xx.shape[1]-1):
                
                cp.cuda.Stream.null.synchronize()
                num_samples = num[i, j] 

                # Ignore cells configured with no samples.
                if num_samples > 0:
                    # Locate data in the current grid cell.
                    mask_x = (wrap_xx_GPU[i, j] <= Delta_Phi_GPU) & (Delta_Phi_GPU < wrap_xx_GPU[i, j+1])
                    mask_y = (wrap_yy_GPU[i, j] <= Delta_Theta_GPU) & (Delta_Theta_GPU < wrap_yy_GPU[i+1, j])
                    mask = mask_x & mask_y
                    indices = cp.where(mask)[0]

                    if len(indices) > 0:
                        if len(indices) <= num_samples:
                            # Retain all data when the cell has at most num_samples entries.
                            selected_indices = indices
                        else:
                            # Sample without replacement.
                            selected_indices = cp.random.choice(indices, size=num_samples, replace=False)
                        selected_indices_list.append(selected_indices)

                cp.cuda.Stream.null.synchronize()

        # Combine and deduplicate selected indices.
        selected_indices_all = cp.concatenate(selected_indices_list)
        selected_indices_all = cp.unique(selected_indices_all)

        # Transfer indices back to the CPU.
        selected_indices_all = cp.asnumpy(selected_indices_all)
        gc.collect()

        # Apply the selected indices to every data array.
        for key in data.keys():
            thinned_data[key] = data[key][selected_indices_all]

        print("Resampling complete.", flush=True)
        print(f"Samples before: {len(Phi1)}; samples after: {len(thinned_data['Phi1_ast'])}", flush=True)
        
        return thinned_data

