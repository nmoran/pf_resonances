#!/usr/bin/env python

"""
Test the MPS class
"""
import unittest
import numpy as np

import parafermions as pf

class Test(unittest.TestCase):

    def test_mps_initialisation(self):
        # should initialise with all zeros
        mps = pf.MPS(4, 4, 2)
        assert(np.sum([np.sum(x) for i, x in mps.M.items()]) == 4.0)

        mps = pf.MPS(4, 4, 2, dtype=np.complex128)
        assert(np.sum([np.sum(x) for i, x in mps.M.items()]) == 4.0)


    def test_foo(self):
        assert(True)
