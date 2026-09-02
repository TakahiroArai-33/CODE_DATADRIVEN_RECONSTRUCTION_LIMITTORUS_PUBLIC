from typing import Literal
import numpy as np
from SpectralDecomposition import calculate_Hk
from time_deco import log_execution_time



def return_to_section(X, rx, t, section, key:Literal["U", "V"], index=0,
                      display=True):
    """
    Return the state and time at an upward crossing of a Poincare section.
    The section is defined using H0, computed from U or V and independent of
    the spatial phase Phi.

    ``index`` selects the crossing counted from the start; zero selects the
    first crossing.
    """

    U, V = X

    if key == "U":
        H = calculate_Hk(U, rx, j=0)
    elif key == "V":
        H = calculate_Hk(V, rx, j=0)

    t_left, t_right = t[:-1], t[1:]
    H_left, H_right = H[:-1], H[1:]

    U_left, U_right = U[:-1], U[1:]
    V_left, V_right = V[:-1], V[1:]

    # True where the trajectory crosses the section upward.
    args = (H_left<=section) & (section<=H_right)

    ## 
    if display:
        print("Key: {:s}; section value: {:0.4f}".format(key, section))
        print("Number of crossings: {:d}".format(args.sum()))
        print("Selecting upward crossing number {:d}.".format(index+1))
    else:
        pass

    ## 
    if args.sum() == 0:
        print("No upward crossing of the section was found.")
        return None

    ## the position of Ture in `args`.
    positions = np.where(args==True)[0]
    ## select which elements in `position` is used.
    arg = positions[index]
    ## 
    t1, t2 = t_left[arg], t_right[arg]
    f1, f2 = H_left[arg], H_right[arg]
    U1, U2 = U_left[arg], U_right[arg]
    V1, V2 = V_left[arg], V_right[arg]

    L = np.abs(f2-f1)
    a = np.abs(section-f1)/L

    tc = (1-a)*t1 + a*t2
    Uc = ((1-a)*U1 + a*U2)
    Vc = ((1-a)*V1 + a*V2)
    
    # State [U, V] and time at upward crossing number index + 1.
    return [Uc, Vc], tc



# @log_execution_time
def measure_period(X, rx, t, section, key="U", index=0):
    """ 
    Measure the time between consecutive crossings of the Poincare section.
    """

    _, t1 = return_to_section(X, rx, t, section, key=key, index=int(index),
                      display=False)
    _, t2 = return_to_section(X, rx, t, section, key=key, index=int(index+1),
                      display=False)
    
    period = np.abs(t2-t1)
    print("Period: {:0.4f} s".format(period))
    return period
