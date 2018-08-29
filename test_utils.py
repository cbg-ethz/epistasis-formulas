import unittest

import numpy as np
import pandas

import utils


class TestSplitPosNeg(unittest.TestCase):

    def setUp(self):
        m = np.array([
            [ 1, -3,  0],
            [-2, -4,  9],
            [ 1,  2,  3],
            [ 0,  0,  0]
            [ 4, 17, -9]
        ])

        mp = np.array([
            [ 1,  0,  0],
            [ 0,  0,  9],
            [ 1,  2,  3],
            [ 0,  0,  0]
            [ 4, 17,  0]
        ])

        mn = np.array([
            [ 0, -3,  0],
            [-2, -4,  0],
            [ 0,  0,  0],
            [ 0,  0,  0]
            [ 0,  0, -9]
        ])

        p, n = utils.split_pos_neg(m)
        self.assertTrue(np.array_equal(p, mp))
        self.assertTrue(np.array_equal(n, mn))


class TestFormatContext(unittest.TestCase):

    def test_empty_without_context(self):
        self.assertEqual(utils.format_context([], 0), '')

    def test_no_context_2(self):
        self.assertEqual(utils.format_context([], 2), 'AB')

    def test_context_5(self):
        self.assertEqual(utils.format_context([(2, 0), (4, 1)], 5), 'AB0C1')


class TestConvertToPystasis(unittest.TestCase):

    def setUp(self):
        self.exp_w = np.arange(2**5)
        self.pystasis_w = utils.convert_vector_to_pystasis_order(self.exp_w)
        self.target_w = np.array([
             0,  5,  4, 15,  3, 14, 13, 25,
             2, 12, 11, 24, 10, 23, 22, 30,
             1,  9,  8, 21,  7, 20, 19, 29,
             6, 18, 17, 28, 16, 27, 26, 31
        ])

    def test_permute_np_vector(self):
        self.assertTrue(np.array_equal(self.pystasis_w, self.target_w))

    def test_permute_pandas_dataframe(self):
        exp_df = pandas.DataFrame(self.exp_w)
        pystasis_df = utils.convert_dataframe_to_pystasis_order(exp_df)
        target_pdf = pandas.DataFrame(self.target_w)
        self.assertTrue(pystasis_df.equals(target_pdf))


if __name__ == '__main__':
    unittest.main()
