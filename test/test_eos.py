import unittest

import numpy

from src.eos import EOS


class MyTestCase(unittest.TestCase):
    def test_eos_parameters(self):
        accentric_factor = numpy.array([0.2236, 0.7174])
        critical_temperatures = numpy.array([304.21, 723.00])
        ac = numpy.array([3.96211105e+06, 1.18021492e+08])
        temperature = 557.5

        a = EOS.eos_parameters(accentric_factor, critical_temperatures, ac, temperature)
        expected_a = numpy.array([2.23023699e+06, 1.59791896e+08])

        self.assertTrue(numpy.allclose(expected_a, a))

    def test_VdW1fMIX(self):
        a = numpy.array([2.23012210e+06, 1.59785351e+08])
        b = numpy.array([26.65350799, 334.05964905])
        kij = numpy.array([[0., 0.], [0., 0.]])
        lij = numpy.array([[0., 0.], [0., 0.]])
        Composition = numpy.array([[0.9, 0.06261139], [0.1, 0.93754181]])

        amix = numpy.array([0., 0.])
        bmix = numpy.array([0., 0.])

        amix[0], bmix[0] = EOS.VdW1fMIX(2, a, b, kij, lij, Composition[:, 0])
        amix[1], bmix[1] = EOS.VdW1fMIX(2, a, b, kij, lij, Composition[:, 1])

        expected_amix = numpy.array([6.80211104e+06, 1.42630096e+08])
        expected_bmix = numpy.array([57.39412209, 314.81547131])

        self.assertTrue(numpy.allclose(amix, expected_amix))
        self.assertTrue(numpy.allclose(bmix, expected_bmix))

    def test_Eos_Volume(self):
        P = 12.23
        T = 557.5

        amix = numpy.array([6.80211104e+06, 1.42630096e+08])
        bmix = numpy.array([57.39412209, 314.81547131])
        expected_volume = numpy.array([3704.09392444, 424.95313387])
        volume = numpy.array([0., 0.])
        for i in range(2):
            volume[i] = EOS.EoS_Volume(P, T, bmix[i], amix[i], i)

        self.assertTrue(numpy.allclose(expected_volume, volume))
