import unittest

import numpy as np
import pandas

import epistasis


class TestGenCircuitTag(unittest.TestCase):

    def test_0(self):
        self.assertEqual(epistasis.gen_circuit_tag(0, ''), 'a_')

    def test_42(self):
        self.assertEqual(epistasis.gen_circuit_tag(2, '012'), 'c_012')


if __name__ == '__main__':
    unittest.main()
