# PyLinalgBasic
Retirado originalmente de um EP (Exercicio Programa) para a disciplina MAP3122 - Métodos Numéricos e Aplicações (2019)

Feita com o princípio de ser fácil de usar. Não muito focada em performance.

Fiz uma [adaptação dessas classes em C++17](https://github.com/jv-oliveira/linalg_basic) que pode ser utilizada para performance.

## Classes

### Matrix

classe generica de uma matriz. Possui os operadores aritméticos clássicos implementados.

### Vector
Visto como uma matriz (n, 1).

### TransposedVector
Visto como uma matriz (1, n).

### VectorView e TransposedVectorView

São classes que são utilizadas para visualizar dados, geralmente de uma matriz.

## Exemplo

Há um exemplo que foi a atividade da disciplina. A *fatorização QR*.

Para ver a simplicidade no uso, eis a seguinte função que mostrará a lógica da fatoração QR:

```python
def factorization(a: Matrix) -> (Matrix, Matrix):
    """Get Q and R from QR factorization"""
    assert is_symmetric(a)
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
```
