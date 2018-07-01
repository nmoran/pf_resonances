#!/usr/bin/env python

"""
Test the MPS class
"""
import unittest
import numpy as np

import parafermions as pf

class Test(unittest.TestCase):

    def test_pe_degeneracy(self):
        # should initialise with all zeros
        N, l = 8, 0.2
        pe = PeschelEmerySpinHalf(N, l, dtype=np.dtype('float64'))
        d, v = pe.Diagonalise(k=100)
        assert(np.sum(d[1:11:2]-d[:11:2]) < 1e-10)

        N, l = 8, 1.0
        pe = PeschelEmerySpinHalf(N, l, dtype=np.dtype('float64'))
        d, v = pe.Diagonalise(k=100)
        assert((d[1]-d[0]) < 1e-15)
        assert(np.sum(d[1:11:2]-d[:11:2]) > 1e-2)
