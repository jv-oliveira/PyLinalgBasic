import unittest

from matrix import *
from qr_factorization import *


class InitialTests(unittest.TestCase):
    epsilon = 1e-2
    max_iterations = 1000

    @staticmethod
    def check_equal(a, b, epsilon, dtype: np.dtype = float):
        if isinstance(a, Matrix):
            a = a.as_ndarray()
        elif isinstance(b, Matrix):
            b = b.as_ndarray()
        else:
            a = np.array(a, dtype=dtype)
            b = np.array(b, dtype=dtype)
        return np.allclose(a, b, atol=epsilon)

    def assert_eigenvectors(self, eigenvectors, calculated, epsilon=None, dtype: np.dtype = float):
        for i in range(calculated.num_lines()):
            self.assertTrue(InitialTests.check_equal(calculated[i],
                                                     eigenvectors[i],
                                                     epsilon if epsilon is not None else self.epsilon,
                                                     dtype))

    def test_1(self):
        a = Matrix(3, 3, np.array([[3, 4, 0],
                                   [4, 3, 0],
                                   [0, 0, 2]]))
        values, vectors = eigenvalues_and_eigenvectors(a, 1e-2, 1000)
        print(values)
        self.assertTrue(InitialTests.check_equal([7.0, -1.0, 2.0],
                                                 values,
                                                 self.epsilon))

        print(vectors)
        self.assert_eigenvectors([
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], vectors, 3e-1)

    @unittest.skip
    def test_2(self):
        a = Matrix(3, 3, np.array([[3, 4, 0],
                                   [1, 3, 0],
                                   [0, 0, 2]]))
        values = eigenvalues(a, self.epsilon, self.max_iterations)
        self.assertTrue(InitialTests.check_equal([1.0, 5.0, 2.0],
                                                 values,
                                                 self.epsilon))

    @unittest.expectedFailure
    def test_3(self):
        """complex eigenvalues"""
        a = Matrix(2, 2, np.array([[1, 1],
                                   [-3, 1]]))
        values = eigenvalues(a, self.epsilon, self.max_iterations)
        self.assertTrue(InitialTests.check_equal([complex(1, sqrt(3)), complex(1, -sqrt(3))],
                                                 values,
                                                 self.epsilon, dtype=np.complex))

    @unittest.skip
    def test_4(self):
        a = Matrix(2, 2, np.array([[3, 3],
                                   [0.33333, 5]]))
        values = eigenvalues(a, self.epsilon, self.max_iterations)
        self.assertTrue(InitialTests.check_equal([5.41421, 2.585789973],
                                                 values,
                                                 self.epsilon))

    # def test_0(self):
    #     a = Matrix(3, 3, np.array([[3, 4, 0],
    #                                [4, 3, 0],
    #                                [0, 0, 2]]))
    #     values, vectors = eigenvalues_and_eigenvectors(a, eps, 1000)

if __name__ == '__main__':
    unittest.main()
