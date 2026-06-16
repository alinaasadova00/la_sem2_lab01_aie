# algorithms/canonical_form.py

"""Приведение TT-тензора в лево- и право-канонические формы."""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt, backend):
    """Возвращает TT-тензор в лево-канонической форме."""
    if tt.order <= 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape

        M = core.reshape((r_prev * n_k, r_next))
        Q, R = backend.qr(M)
        cores[k] = Q.reshape((r_prev, n_k, r_next))

        next_core = cores[k + 1]
        n_next = next_core.shape[1]
        r_next_next = next_core.shape[2]

        for i in range(n_next):
            slice_mat = next_core[:, i, :].reshape((r_next, r_next_next))
            new_slice = backend.matmul(R, slice_mat)
            for a in range(r_next):
                for b in range(r_next_next):
                    next_core[a, i, b] = new_slice[a, b]

    return TTTensor(cores)


def right_canonicalize(tt, backend):
    """Возвращает TT-тензор в право-канонической форме."""
    if tt.order <= 1:
        return tt.copy()

    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape

        M = core.reshape((r_prev, n_k * r_next))
        M_t = backend.transpose(M)
        Q_t, R_t = backend.qr(M_t)

        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)
        cores[k] = Q.reshape((r_prev, n_k, r_next))

        prev_core = cores[k - 1]
        n_prev = prev_core.shape[1]
        r_prev_prev = prev_core.shape[0]

        for i in range(n_prev):
            slice_mat = prev_core[:, i, :].reshape((r_prev_prev, r_prev))
            new_slice = backend.matmul(slice_mat, R)
            for a in range(r_prev_prev):
                for b in range(r_prev):
                    prev_core[a, i, b] = new_slice[a, b]

    return TTTensor(cores)


# ────────────────────────────────────────────────
# Вспомогательные функции (остались из заглушек)
# ────────────────────────────────────────────────

def _numerical_rank(S, rel_tol=1e-8, abs_tol=1e-12):
    r"""Числовой ранг по вектору сингулярных значений."""
    if S.size == 0:
        return 0

    s_max = max(abs(s) for s in S.data)
    threshold = max(abs_tol, rel_tol * s_max)

    rank = 0
    for s in S.data:
        if abs(s) > threshold:
            rank += 1
        else:
            break

    return max(1, rank)


def _truncate_columns(matrix, rank, backend):
    """Первые rank столбцов матрицы."""
    m, n = matrix.shape
    actual_rank = min(rank, n)
    data = [0.0] * (m * actual_rank)
    for i in range(m):
        for j in range(actual_rank):
            data[i * actual_rank + j] = matrix[i, j]
    return DenseTensor((m, actual_rank), data=data)


def _truncate_rows(matrix, rank, backend):
    """Первые rank строк матрицы."""
    k, n = matrix.shape
    actual_rank = min(rank, k)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        for j in range(n):
            data[i * n + j] = matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)


def _truncate_vector(vector, rank, backend):
    """Первые rank элементов вектора."""
    actual_rank = min(rank, vector.size)
    data = [vector.data[i] for i in range(actual_rank)]
    return DenseTensor((actual_rank,), data=data)


def _multiply_diag_matrix(diag_vec, matrix, rank, backend):
    """diag(diag_vec) @ matrix."""
    m, n = matrix.shape
    actual_rank = min(rank, m, diag_vec.size)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        s = diag_vec.data[i]
        for j in range(n):
            data[i * n + j] = s * matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)


def _multiply_columns_by_diag(matrix, diag_vec, backend):
    """matrix @ diag(diag_vec)."""
    m, n = matrix.shape
    actual_len = min(n, diag_vec.size)
    data = [0.0] * (m * n)
    for i in range(m):
        for j in range(actual_len):
            data[i * n + j] = matrix[i, j] * diag_vec.data[j]
    return DenseTensor((m, n), data=data)
