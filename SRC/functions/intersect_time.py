#!/usr/bin/env python3
# coding: utf-8

import numpy as np

def intersect_time(t: np.ndarray, x: np.ndarray, section: float) -> np.ndarray:
    """
    t: 1d-ndarray
    x: 1d-ndarray
    """
    t_left, t_right = t[:-1], t[1:]
    x_left, x_right = x[:-1], x[1:]
    # Detect upward crossings of the section.
    args = (x_left<=section) & (section<=x_right)

    print("Section value: {:0.4f}".format(section))
    print("Number of crossings: {:d}".format(args.sum()))
    if args.sum() == 0:
        print("No upward crossing of the section was found.")
        return []
    
    t1, t2 = t_left[args], t_right[args]
    x1, x2 = x_left[args], x_right[args]
    a = np.abs(section-x1)/np.abs(x2-x1)
    tc = (1-a)*t1 + a*t2
    return tc
