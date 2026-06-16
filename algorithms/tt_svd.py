# algorithms/tt_svd.py

"""TT-SVD алгоритм: разложение плотного тензора в TT-формат."""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(tensor, backend, max_rank=None, eps=1e-10):
    """Возвращает TTTensor — тензор в TT-формате."""
    shape = tensor.shape
    order = len(shape)

    if order == 1:
        return TTTensor([tensor.reshape((1, shape[0], 1))])

    norm_a = backend.norm(tensor)
    delta = 0.0
    if norm_a > 1e-30:
        delta = eps * norm_a / math.sqrt(order - 1)

    # добавляем ведущую размерность 1, чтобы на каждом шаге сворачивать первые две
    C = tensor.reshape((1,) + shape)
    r_prev = 1
    cores = []

    for k in range(order - 1):
        n_k = C.shape[1]
        rows = r_prev * n_k
        cols = C.size // rows

        C_mat = C.reshape((rows, cols))
        U, S, Vt = backend.svd(C_mat)

        rank = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank)
        G_k = U_trunc.reshape((r_prev, n_k, rank))
        cores.append(G_k)

        S_trunc = _truncate_vector(S, rank)
        Vt_trunc = _truncate_rows(Vt, rank)
        C_mat_new = _multiply_diag_matrix(S_trunc, Vt_trunc, rank)

        C = C_mat_new.reshape((rank,) + C.shape[2:])
        r_prev = rank

    G_last = C.reshape((r_prev, shape[-1], 1))
    cores.append(G_last)

    return TTTensor(cores)


# ────────────────────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────────────────────

def _compute_truncated_rank(S, delta, max_rank):
    """Возвращает ранг усечения по сингулярным значениям."""
    k = S.size
    if k == 0:
        return 1

    s_max = max(abs(s) for s in S.data)
    threshold = max(1e-12, 1e-8 * s_max)

    r_tilde = 0
    for s in S.data:
        if abs(s) > threshold:
            r_tilde += 1
        else:
            break

    if r_tilde == 0:
        r_tilde = 1

    rank = r_tilde

    if delta > 0.0 and rank > 1:
        tail = 0.0
        while rank > 1:
            next_tail = tail + S.data[rank - 1] * S.data[rank - 1]
            if next_tail <= delta * delta:
                tail = next_tail
                rank -= 1
            else:
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    return max(1, rank)


def _truncate_columns(matrix, rank):
    """Возвращает первые rank столбцов матрицы."""
    m, n = matrix.shape
    actual_rank = min(rank, n)
    data = [0.0] * (m * actual_rank)
    for i in range(m):
        for j in range(actual_rank):
            data[i * actual_rank + j] = matrix[i, j]
    return DenseTensor((m, actual_rank), data=data)


def _truncate_rows(matrix, rank):
    """Возвращает первые rank строк матрицы."""
    k, n = matrix.shape
    actual_rank = min(rank, k)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        for j in range(n):
            data[i * n + j] = matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)


def _truncate_vector(vector, rank):
    """Возвращает первые rank элементов вектора."""
    actual_rank = min(rank, vector.size)
    data = [vector.data[i] for i in range(actual_rank)]
    return DenseTensor((actual_rank,), data=data)


def _multiply_diag_matrix(diag_vec, matrix, rank):
    """Возвращает diag(diag_vec) @ matrix."""
    m, n = matrix.shape
    actual_rank = min(rank, m, diag_vec.size)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        s = diag_vec.data[i]
        for j in range(n):
            data[i * n + j] = s * matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)
