import unittest

import numpy as np

import circuits


class TestCircuit3Generation(unittest.TestCase):

    def setUp(self):
        self.circuits = circuits.gen_circuits_3()

    def test_zero_sum(self):
        circuit_sum = self.circuits.sum(axis=1)
        zeroes = np.zeros(circuit_sum.shape, circuit_sum.dtype)
        self.assertTrue(np.array_equal(circuit_sum, zeroes))


if __name__ == '__main__':
    unittest.main()
