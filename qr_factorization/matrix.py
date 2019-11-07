from __future__ import annotations

from math import sqrt, isclose

import numpy as np


class Matrix:

    def __init__(self, n: int, m: int, array: np.ndarray = None):
        self._eps = 1e-5
        self._n = n
        self._m = m
        if array is None:
            self._array = np.zeros((n, m))
        else:
            self._array = array.copy().reshape((n, m))

    @classmethod
    def from_file(cls, input_file: str, data_type: np.dtype, delimiter: chr):
        a = np.genfromtxt(fname=input_file, dtype=data_type, delimiter=delimiter)
        print("Matrix:", a)
        n = len(a)
        m = len(a[0])
        return cls(n, m, a)

    def __getitem__(self, index: int):
        return TransposedVectorView(self._m, self._array[index])

    def __setitem__(self, index: int, value):
        self._array[index] = value

    def __add__(self, other: Matrix) -> Matrix:
        return self.copy().add(other)

    def __sub__(self, other: Matrix) -> Matrix:
        return self.copy().subtract(other)

    def __mul__(self, other) -> Matrix:
        if isinstance(other, Matrix):
            return self.dot_product(other)
        else:
            return self.copy().scalar_product(other)

    def __rmul__(self, other) -> Matrix:
        if isinstance(other, Matrix):
            return other.dot_product(self)
        else:
            return self.copy().scalar_product(other)

    def __contains__(self, item):
        if isinstance(item, float) and self._array.dtype == np.float64:
            for value in np.nditer(self._array):
                if isclose(item, value):
                    return True
            return False
        else:
            return item in self._array

    def __truediv__(self, scalar):
        assert scalar
        return (1 / scalar) * self

    def __str__(self):
        return str(self._array)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        else:
            other = other.as_matrix()

        if self.size() != other.size():
            return False
        else:
            for i in range(self.num_lines()):
                for j in range(self.num_columns()):
                    if not isclose(self[i][j], other[i][j]):
                        return False
            return True

    def as_matrix(self):
        return self

    def scalar_product(self, other):
        for i in range(self._n):
            for j in range(self._m):
                self[i][j] *= other
        return self

    @staticmethod
    def __get_ret_value(m: int, i: int, j: int, a: Matrix, b: Matrix):
        sum_value = 0
        for k in range(m):
            sum_value += a[i][k] * b[k][j]
        return sum_value

    def dot_product(self, other: Matrix):
        assert self._m == other._n
        ret = Matrix(self._n, other._m)
        for i in range(self._n):
            for j in range(other._m):
                ret[i][j] = Matrix.__get_ret_value(self._m, i, j, self.as_matrix(), other.as_matrix())
        return ret

    def size(self):
        return self._n, self._m

    def num_columns(self):
        return self._m

    def num_lines(self):
        return self._n

    def column(self, m: int):
        return VectorView(self._n, self._array[:, m].view())

    def line(self, n: int):
        return self[n]

    def diagonal(self):
        assert self._n == self._m
        return TransposedVectorView(self._n, np.diagonal(self._array.view()))

    def copy(self):
        return Matrix(self._n, self._m, self._array)

    def add(self, other: Matrix) -> Matrix:
        assert self._m == other._m
        assert self._n == other._n
        for i in range(self._n):
            for j in range(self._m):
                self.as_matrix()[i][j] += other.as_matrix()[i][j]
        return self

    def subtract(self, other: Matrix) -> Matrix:
        assert self._m == other._m
        assert self._n == other._n
        for i in range(self._n):
            for j in range(self._m):
                self[i][j] -= other[i][j]
        return self

    def divide(self, scalar):
        return self.scalar_product(1 / scalar)

    def transpose(self) -> Matrix:
        ret = Matrix(self._m, self._n)
        for i in range(ret._n):
            for j in range(ret._m):
                ret[i][j] = self[j][i]
        return ret

    def as_ndarray(self):
        return self._array

    def get_eps(self):
        return self._eps

    def set_eps(self, value):
        self._eps = value

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        ret = Matrix(matrix._m, matrix._n)
        for i in range(ret._n):
            for j in range(ret._m):
                ret[i][j] = matrix[j][i]
        return ret

    @staticmethod
    def identity(n: int) -> Matrix:
        return Matrix(n, n, np.identity(n))


class __UnidirectionalMatrixBase(Matrix):
    def __init__(self, n: int, m: int, array: np.ndarray = None):
        assert m == 1 or n == 1
        super().__init__(n, m, array)

    def __getitem__(self, i: int):
        assert self._n * self._m > i
        return self._array[i % self._n][i % self._m]

    def __setitem__(self, i: int, value):
        assert self._n * self._m > i
        self._array[i % self._n][i % self._m] = value

    def __mul__(self, other):
        return self.multiply(self, other)

    def __add__(self, other):
        ret = super().__add__(other)
        ret.__class__ = self.__class__
        return ret

    def __truediv__(self, other):
        ret = super().__truediv__(other)
        ret.__class__ = self.__class__
        return ret

    def as_matrix(self) -> Matrix:
        return Matrix(self._n, self._m, self._array)

    def norm(self):
        a = self * self
        return sqrt(a)

    def size(self):
        return self._n * self._m

    def multiply(self, other):
        pass


class __VectorBase(__UnidirectionalMatrixBase):
    def __init__(self, n: int, m: int, array: np.ndarray = None):
        super().__init__(n, m, array)

    @classmethod
    def multiply(cls, a, b):
        if isinstance(a, cls) and isinstance(b, cls):
            return (a.transpose().as_matrix() * b.as_matrix())[0][0]
        else:
            ret = super(a).__mul__(b)
            ret.__class__ = a.__class__
            return ret


class __TransposedVectorBase(__UnidirectionalMatrixBase):
    def __init__(self, n: int, m: int, array: np.ndarray = None):
        super().__init__(n, m, array)

    @classmethod
    def multiply(cls, a, b):
        if isinstance(a, cls) and isinstance(b, cls):
            return (a.as_matrix() * b.transpose().as_matrix())[0][0]
        else:
            ret = super(a).__mul__(b)
            ret.__class__ = a.__class__
            return ret


class Vector(__VectorBase):

    def __init__(self, n: int, array: np.ndarray = None):
        super().__init__(n, 1, array)

    def transpose(self) -> TransposedVector:
        return TransposedVector(self.size(), self._array)


class TransposedVector(__TransposedVectorBase):

    def __init__(self, n: int, array: np.ndarray = None):
        super().__init__(1, n, array)

    def as_matrix(self) -> Matrix:
        return Matrix(1, self.size(), self._array)

    def transpose(self) -> Vector:
        return Vector(self.size(), self._array)


class __VectorViewerBase(__UnidirectionalMatrixBase):

    def __init__(self, n: int, m: int, array: np.ndarray = None):
        assert m == 1 or n == 1
        self._n = n
        self._m = m
        if array is None:
            self._array = np.zeros((n, m))
        else:
            self._array = array.view()
        self._array.shape = (n, m)


class VectorView(__VectorViewerBase, __VectorBase):

    def __init__(self, n: int, array: np.ndarray = None):
        super().__init__(n, 1, array.view())

    def transpose(self) -> TransposedVectorView:
        return TransposedVectorView(self.size(), self._array.view())


class TransposedVectorView(__VectorViewerBase, __TransposedVectorBase):

    def __init__(self, n: int, array: np.ndarray = None):
        super().__init__(1, n, array.view())

    def transpose(self) -> VectorView:
        return VectorView(self.size(), self._array.view())
