#!/usr/bin/env python

"""
Test the MPS class
"""
import parafermions as pf

import numpy as np

def test_mps_initialisation():
    # should initialise with all zeros
    mps = pf.MPS(4, 4, 2)
    assert(np.sum([np.sum(x) for i, x in mps.M.items()]) == 4.0)

    mps = pf.MPS(4, 4, 2, dtype=np.complex128)
    assert(np.sum([np.sum(x) for i, x in mps.M.items()]) == 4.0)


