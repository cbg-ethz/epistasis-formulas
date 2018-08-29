import unittest

import numpy as np

import fourier


class TestFourierMatrixGeneration(unittest.TestCase):

    def test_inner_product(self):
        self.assertEqual(fourier._inner_product(0, 0), 0)
        self.assertEqual(fourier._inner_product(1, 1), 1)
        self.assertEqual(fourier._inner_product(4, 4), 1)
        self.assertEqual(fourier._inner_product(1, 4), 0)
        self.assertEqual(fourier._inner_product(15, 6), 2)

    def test_generate_full_fourier_matrix_iter_0(self):
        f0 = np.array([[1]])
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_iter(0),
            f0
        ))

    def test_generate_full_fourier_matrix_iter_1(self):
        f1 = np.array([[1, 1], [1, -1]])
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_iter(1),
            f1
        ))

    def test_generate_full_fourier_matrix_iter_2(self):
        f2 = np.array([
             [1, 1, 1, 1],
             [1, -1, 1, -1],
             [1, 1, -1, -1],
             [1, -1, -1, 1],
        ])
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_iter(2),
            f2
        ))

    def test_generate_full_fourier_matrix_rec_0(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_rec(0),
            fourier.generate_full_fourier_matrix_iter(0),
        ))

    def test_generate_full_fourier_matrix_rec_1(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_rec(1),
            fourier.generate_full_fourier_matrix_iter(1),
        ))

    def test_generate_full_fourier_matrix_rec_2(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix_rec(2),
            fourier.generate_full_fourier_matrix_iter(2),
        ))

    def test_generate_full_fourier_matrix_0(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix(0),
            fourier.generate_full_fourier_matrix_rec(0),
        ))

    def test_generate_full_fourier_matrix_1(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix(1),
            fourier.generate_full_fourier_matrix_rec(1),
        ))

    def test_generate_full_fourier_matrix_2(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix(2),
            fourier.generate_full_fourier_matrix_rec(2),
        ))

    def test_generate_full_fourier_matrix_5(self):
        self.assertTrue(np.array_equal(
            fourier.generate_full_fourier_matrix(5),
            fourier.generate_full_fourier_matrix_rec(5),
        ))

    def test_generate_singleton_indices_0(self):
        self.assertCountEqual(
            fourier.generate_singleton_indices(0),
            (0,)
        )

    def test_generate_singleton_indices_1(self):
        self.assertCountEqual(
            fourier.generate_singleton_indices(1),
            (0,1)
        )

    def test_generate_singleton_indices_5(self):
        self.assertCountEqual(
            fourier.generate_singleton_indices(5),
            (0, 1, 2, 4, 8, 16)
        )

    def test_generate_fourier_matrix_2(self):
        self.assertTrue(np.array_equal(
            fourier.generate_fourier_matrix(2),
            np.array([[1, -1, -1, 1]])
        ))

    def test_generate_fourier_matrix_3(self):
        self.assertTrue(np.array_equal(
            fourier.generate_fourier_matrix(3),
            np.array([
                [ 1, -1, -1,  1,  1, -1, -1,  1],
                [ 1, -1,  1, -1, -1,  1, -1,  1],
                [ 1,  1, -1, -1, -1, -1,  1,  1],
                [ 1, -1, -1,  1, -1,  1,  1, -1]
            ])
        ))


if __name__ == '__main__':
    unittest.main()
