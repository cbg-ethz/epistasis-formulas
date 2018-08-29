import unittest

import numpy as np

import slicing


class TestFourierMatrixGeneration(unittest.TestCase):

    def test_init(self):
        base, rank = 2, 5
        tp = slicing.TensorProjector(base=base, rank=rank)
        self.assertEqual(tp.base, 2)
        self.assertEqual(tp.rank, 5)
        self.assertEqual(tp.all, slice(0, base, 1))

    def test_tensorize(self):
        base, rank = 3, 4
        v = np.arange(0, base**rank)
        tp = slicing.TensorProjector(base=base, rank=rank)
        vt = tp.tensorize(v)
        new_shape = [base] * rank
        vr = v.reshape(new_shape)
        self.assertTrue(np.array_equal(vt, vr))

    def test_project_tensor(self):
        pass

    def test_project_vector(self):
        pass
