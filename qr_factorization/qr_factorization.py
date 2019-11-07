from matrix import Matrix, Vector, TransposedVectorView

eps = 1.0e-2


def householder(h_v: Vector) -> Matrix:
    """Get Householder transformation Matrix"""
    return Matrix.identity(h_v.size()).subtract(2 * h_v * h_v.transpose() / (h_v * h_v))


def canonical_base(n: int, i: int):
    base = Vector(n)
    base[i] = 1.0
    return base


def factorization(a: Matrix) -> (Matrix, Matrix):
    """Get Q and R from QR factorization"""
    size, size = a.size()
    q = Matrix.identity(size)
    r = a.copy()
    for i in range(size - 1):
        a_i = Vector(size)
        col = r.column(i)
        for j in range(i, r.num_columns()):
            a_i[j] = col[j]
        delta = a_i[i] / abs(a_i[i])
        v_i = a_i + delta * a_i.norm() * canonical_base(size, i)
        h_v = householder(v_i)
        q = q * h_v
        r = h_v * r
    return q, r


def almost_zero(value, epsilon=eps) -> bool:
    return abs(value) < epsilon


def almost_upper_triangular(a: Matrix, epsilon=eps):
    n, m = a.size()
    for i in range(n - 1):
        for j in range(i + 1, m):
            if not almost_zero(a[i][j], epsilon):
                return False
    return True


def eigenvalues(a: Matrix, epsilon=eps, max_iterations=1000) -> TransposedVectorView:
    """Eigenvalues using QR factorization"""
    a_k = a.copy()
    for i in range(max_iterations):
        if almost_upper_triangular(a_k, epsilon):
            break
        q_i, r_i = factorization(a_k)
        a_k = r_i * q_i
    return a_k.diagonal()


def is_symmetric(a: Matrix) -> bool:
    n, m = a.size()
    for i in range(n):
        for j in range(m):
            if i == j:
                break
            elif a[i][j] != a[j][i]:
                return False
    return True


def eigenvalues_and_eigenvectors(a: Matrix, epsilon=eps, max_iterations=1000) \
        -> (TransposedVectorView, Matrix):
    """Eigenvalues|vectors with QR factorization"""
    assert is_symmetric(a)
    a_k = a.copy()
    v_k = Matrix.identity(a.size()[0])
    for i in range(max_iterations):
        if almost_upper_triangular(a_k, epsilon):
            break
        q_i, r_i = factorization(a_k)
        a_k = r_i * q_i
        v_k = v_k * q_i
    return a_k.diagonal(), v_k
