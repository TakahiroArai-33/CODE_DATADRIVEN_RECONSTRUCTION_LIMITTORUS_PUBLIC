from Module_Coupled_GSM import Coupled_GrayScottModel, set_rng_seed
from functions.time_deco import log_execution_time
import numpy as np
import cupy as cp
import gc, os, re
import time
import sys
from scipy import interpolate
from typing import Tuple


"""Simulate a pair of coupled Gray-Scott systems."""


loadfilename = "ReferenceData/Make_X0_env/JITVersion/X0data_withPhi0.npz"
savedir = "CoupledModelSimulationDATA/"


import zlib, math

def make_seed(resume_time: float, a: float, b: float) -> int:
    rt = int(math.floor(float(resume_time) / 1e3))
    aa = round(float(a), 6)
    bb = round(float(b), 6)
    payload = f"{rt:.1f},{aa:.6f},{bb:.6f}".encode()
    return zlib.crc32(payload) & 0xFFFFFFFF


def get_grid_property():
    with np.load(loadfilename, allow_pickle=False) as data:
        t = np.asarray(data["t"])
        rx = np.asarray(data["rx"])
    T = np.ptp(t)
    dtheta = (2.0 * np.pi) * (t[1] - t[0]) / T
    return {
        "shape": (t.size, rx.size),
        "dtheta": float(dtheta),
        "dx": float(rx[1] - rx[0]),
    }


def get_init_condition(lateral_diff:int = 0, vertical_diff:int = 0):
    """ 
    Return two initial states separated by spatial and temporal grid offsets.

    For field-array dimensions ``[a + 1, b + 1]``:
        delta_phi = (L/a) * lateral_diff
        delta_theta = (T/b) * vertical_diff
    """
    with np.load(loadfilename, allow_pickle=False) as data:
        U = np.asarray(data["U"])
        V = np.asarray(data["V"])
        tspan = np.asarray(data["t"])
        rx = np.asarray(data["rx"])

    T = np.ptp(tspan)
    L = np.ptp(rx)

    lateral_diff = lateral_diff % (rx.size - 1)
    vertical_diff = vertical_diff % (tspan.size - 1)

    U2, V2 = U[0], V[0]

    U1 = np.zeros_like(U2)
    V1 = np.zeros_like(V2)
    U1_temp, V1_temp = U[vertical_diff], V[vertical_diff]
    U1[:-1] = np.roll(U1_temp[:-1], lateral_diff)
    V1[:-1] = np.roll(V1_temp[:-1], lateral_diff)
    U1[-1], V1[-1] = U1[0], V1[0]

    delta_phi = (rx[lateral_diff] - rx[0])
    delta_theta = (2.0*np.pi) * (tspan[vertical_diff] - tspan[0]) / T

    #* mod
    delta_phi = np.mod(delta_phi, L)
    delta_theta = np.mod(delta_theta, 2.0*np.pi)

    print("delta_phi: {:0.4f}".format(delta_phi))
    print("delta_theta: {:0.4f}".format(delta_theta))
    return (U1, V1), (U2, V2), delta_phi, delta_theta




if __name__ == '__main__':

    import argparse
    import json

    def load_config(path):
        with open(path, 'r') as f:
            return json.load(f)


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        args = parser.parse_args()

        config = load_config(args.config)

        DT = float(config["DT"])
        GRIDNUM = int(config["GRIDNUM"])
        C_VELOCITY = float(config["C_VELOCITY"])
        T_END = float(config["T_END"])
        epsilon = float(config["EPSILON"])
        sigma = float(config["SIGMA"])
        init_condition = config["INIT_CONDITION"]
        TRANSIENT = float(config["TRANSIENT"])
        REPEAT = int(config["REPEAT"])
        stdout_output = str(config["stdout"])
        label = str(config["label"])
        LOG_DT = float(config["LOG_DT"])
        # Nonpositive LOG_DT disables moment logging.
        if LOG_DT <= 0.0:
            LOG_DT = None




    # Append runtime output to the configured log file.
    os.makedirs(savedir, exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = open(stdout_output, 'a')

    print("===================================================", flush=True)
    print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)


    try:
        ts = time.time()
        print("epsilon: {:0.4e}".format(epsilon), flush=True)
        
        GSM = Coupled_GrayScottModel(paramdict={
                                                "epsilon":epsilon,
                                                "sigma":sigma,
                                                "gridnum":GRIDNUM   
                                                })
        
        

        # Convert the requested phase offsets to reference-data grid indices.
        grid_property = get_grid_property()
        dtheta, h = grid_property["dtheta"], grid_property["dx"]



        # Resume from the latest ``{label}_process_{idx}.npz`` when available.
        pattern = re.compile(rf"{re.escape(label)}_process_(\d+)\.npz")
        candidates = [f for f in os.listdir(savedir) if pattern.match(f)]

        if candidates:
            latest_file = max(candidates, key=lambda f: int(pattern.match(f).group(1)))
            repeat_from = os.path.join(savedir, latest_file)
            print(f"Loading restart file: {repeat_from}", flush=True)
            data = np.load(repeat_from)
            U1, V1 = data["U1"][-1], data["V1"][-1]
            U2, V2 = data["U2"][-1], data["V2"][-1]
            current_time = data["t"][-1]
            resume_time = np.copy(data["t"][-1])
            print("Restart time: ", current_time, flush=True)


        else:
            repeat_from = None
            print("No restart file found; starting from the initial condition.", flush=True)
            _delta_phi, _delta_theta = init_condition
            (U1, V1), (U2, V2), delta_phi, delta_theta = \
                get_init_condition(vertical_diff=round(_delta_theta / dtheta), 
                                   lateral_diff=round(_delta_phi / h))
            print("Snapped delta_phi, delta_theta: ", (delta_phi, delta_theta), flush=True)
            print("Requested delta_phi, delta_theta: ", (_delta_phi, _delta_theta), flush=True)

            # Downsample the reference fields to the requested simulation grid.
            STEP = int( (U1.size - 1) / (GRIDNUM-1) )
            U1 = U1[::STEP]
            V1 = V1[::STEP]
            U2 = U2[::STEP]
            V2 = V2[::STEP]

            # Relax the downsampled fields without coupling or noise.
            pre_GSM = Coupled_GrayScottModel(paramdict={
                                                "epsilon":0.0, 
                                                "sigma":0.0,
                                                "gridnum":GRIDNUM
                                                })
            (U1, V1), (U2, V2), _ = pre_GSM.spde_euler(
                                    init_x1=(U1, V1), init_x2=(U2, V2),
                                    dt=DT, iteration=round(TRANSIENT/DT), c=C_VELOCITY,
                                    log_dt=None, start_time=0.0
                                )
            current_time = 0.0
            resume_time = 0.0
            del _delta_phi, _delta_theta, delta_phi, delta_theta, dtheta, h, grid_property, pre_GSM



        U1save, V1save, U2save, V2save = [],[], [], []
        tsave = []
        moment_time_log: list[float] = []
        U1_bar_log: list[float] = []
        V1_bar_log: list[float] = []
        U2_bar_log: list[float] = []
        V2_bar_log: list[float] = []
        A1_tilde_log: list[complex] = []
        A2_tilde_log: list[complex] = []

        # Store the initial state for this segment.
        U1save.append(U1); V1save.append(V1)
        U2save.append(U2); V2save.append(V2)
        tsave.append(current_time)

        # Use a nominal epsilon when checking progress for uncoupled runs.
        eps_for_time_check = float(epsilon)
        if np.isclose(epsilon, 0.0):
            eps_for_time_check = 1.0e-6

        if (resume_time * eps_for_time_check >= T_END):
            print("The requested duration is already complete; skipping simulation.", flush=True)
            sys.exit(0)


        seed = make_seed(resume_time, init_condition[0], init_condition[1])
        np.random.seed(seed)
        cp.random.seed(seed)
        set_rng_seed(seed)
        print(f"RNG seed: {seed}", flush=True)


        # Advance the remaining slow time for this segment.
        #   slow time = epsilon * fast time 
        target_time = (T_END / REPEAT)
        slow_remaining = target_time - (current_time - resume_time) * eps_for_time_check

        if slow_remaining > 0.0:
            fast_remaining = slow_remaining / eps_for_time_check
            iteration_total = max(int(math.ceil(fast_remaining / DT)), 0)
        else:
            iteration_total = 0

        if np.isclose(epsilon, 0.0) and iteration_total == 0 and LOG_DT is not None:
            iteration_total = max(int(math.ceil(float(LOG_DT) / DT)), 0)

        diagnostics = None
        if iteration_total > 0:
            (U1, V1), (U2, V2), diagnostics = GSM.spde_euler(
                                        init_x1=(U1, V1), init_x2=(U2, V2), 
                                        dt=DT, iteration=iteration_total, c=C_VELOCITY,
                                        log_dt=LOG_DT, start_time=current_time
                                    )

            if diagnostics is not None and diagnostics["time"].size > 0:
                diag_time = diagnostics["time"]
                start_idx = 0
                if moment_time_log and np.isclose(diag_time[0], moment_time_log[-1]):
                    start_idx = 1
                if diag_time.size > start_idx:
                    moment_time_log.extend(diag_time[start_idx:].tolist())
                    U1_bar_log.extend(diagnostics["U1_bar"][start_idx:].tolist())
                    V1_bar_log.extend(diagnostics["V1_bar"][start_idx:].tolist())
                    U2_bar_log.extend(diagnostics["U2_bar"][start_idx:].tolist())
                    V2_bar_log.extend(diagnostics["V2_bar"][start_idx:].tolist())
                    A1_tilde_log.extend(diagnostics["A1_tilde"][start_idx:].tolist())
                    A2_tilde_log.extend(diagnostics["A2_tilde"][start_idx:].tolist())

            current_time += DT * iteration_total

            # Save only the final state of this segment.
            U1save.append(U1); V1save.append(V1)
            U2save.append(U2); V2save.append(V2)
            tsave.append(current_time)

        # Convert the initial/final state lists to arrays.
        U1save = np.array(U1save)
        V1save = np.array(V1save)
        U2save = np.array(U2save)
        V2save = np.array(V2save)
        tsave = np.array(tsave).ravel()

        # Select the next process index.
        if repeat_from is None:
            process_index = 0
        else:
            m = re.search(r"_process_(\d+)\.npz", repeat_from)
            process_index = int(m.group(1)) + 1 if m else 0

        save_path = os.path.join(savedir, f"{label}_process_{process_index}.npz")
        # ``t`` contains segment endpoints; ``moments_t`` is the finer diagnostic grid.
        np.savez(save_path,
                    U1=U1save, V1=V1save, U2=U2save, V2=V2save,
                    t=tsave, rx=GSM.rx,
                    moments_t=np.asarray(moment_time_log, dtype=np.float64),
                    U1_bar=np.asarray(U1_bar_log, dtype=np.float64),
                    V1_bar=np.asarray(V1_bar_log, dtype=np.float64),
                    U2_bar=np.asarray(U2_bar_log, dtype=np.float64),
                    V2_bar=np.asarray(V2_bar_log, dtype=np.float64),
                    A1_tilde=np.asarray(A1_tilde_log, dtype=np.complex128),
                    A2_tilde=np.asarray(A2_tilde_log, dtype=np.complex128))


        te = time.time()
        print("Elapsed time: {:0.4f} min ".format((te - ts)/ 60.0), flush=True)

        del GSM
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    finally:
        print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
        print("")


        # Restore standard output.
        sys.stdout.close()
        sys.stdout = original_stdout
