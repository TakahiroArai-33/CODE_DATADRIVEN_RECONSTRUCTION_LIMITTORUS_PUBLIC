#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Module for phase calculation
"""
import numpy as np
import cupy as cp



class PhaseTransform():
    """
    Kramemann method:
    Kralemann, 2008, PRE
    """
    def __init__(self, M=100):
        self.M = M
    
    
    def kralemann_phase_transform(self,
                                  proto_phase:cp.ndarray, Sm_array:np.ndarray) -> cp.ndarray:
        """
        Transform a one-dimensional protophase array using Fourier
        coefficients returned by :meth:`return_Sm`.
        """
        Sm_array_gpu = cp.asarray(Sm_array)
        phase = cp.copy(proto_phase)
        for m_1 in range(1, self.M + 1):
            Sm = Sm_array_gpu[int(m_1-1)]
            phase += 2.0 * cp.imag((Sm / m_1) * (cp.exp(1j * m_1 * proto_phase) - 1.0))
        return phase


    def return_Sm(self, proto_phase:cp.ndarray) -> np.ndarray:
        """
        Compute the Fourier coefficients of the protophase distribution.
        """
        # Sm_array = np.zeros(self.M, dtype=np.complex128)
        # for m_1 in range(1, self.M + 1):
        #     Sm_array[int(m_1-1)] = cp.mean((cp.exp(-1j * m_1 * proto_phase)))
        # return Sm_array

        Sm_array_gpu = cp.zeros(self.M, dtype=cp.complex128)
        for m in range(1, self.M + 1):
            Sm_array_gpu[m-1] = cp.mean(cp.exp(-1j * m * proto_phase))
        return cp.asnumpy(Sm_array_gpu)  # only if callers need NumPy
