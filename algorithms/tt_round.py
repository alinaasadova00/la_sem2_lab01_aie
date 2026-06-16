# algorithms/tt_round.py

"""TT-округление."""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(tt, backend, max_rank=None, eps=1e-10):
    """Возвращает TT-тензор с уменьшенными рангами."""
    if tt.order <= 1:
        return tt.copy()

    tt_right = right_canonicalize(tt, backend)
    cores = [core.copy() for core in tt_right.cores]

    order = len(cores)
    norm_first = backend.norm(cores[0])
    delta = 0.0
    if norm_first > 1e-30:
        delta = eps * norm_first / math.sqrt(order - 1)

    for k in range(order - 1):
        core = cores[k]
        r_prev, n_k, r_next = core.shape

        M = core.reshape((r_prev * n_k, r_next))
        U, S, Vt = backend.svd(M)

        rank = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank)
        cores[k] = U_trunc.reshape((r_prev, n_k, rank))

        S_trunc = _truncate_vector(S, rank)
        Vt_trunc = _truncate_rows(Vt, rank)
        R = _multiply_diag_matrix(S_trunc, Vt_trunc, rank)

        next_core = cores[k + 1]
        n_next = next_core.shape[1]
        r_next_next = next_core.shape[2]

        new_next_core = DenseTensor.zeros((rank, n_next, r_next_next))
        for i in range(n_next):
            slice_mat = next_core[:, i, :].reshape((r_next, r_next_next))
            new_slice = backend.matmul(R, slice_mat)
            for a in range(rank):
                for b in range(r_next_next):
                    new_next_core[a, i, b] = new_slice[a, b]

        cores[k + 1] = new_next_core

    return TTTensor(cores)


# ────────────────────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────────────────────

def _compute_rank(S, delta, max_rank):
    """Ранг усечения по вектору сингулярных значений."""
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
    """Первые rank столбцов."""
    m, n = matrix.shape
    actual_rank = min(rank, n)
    data = [0.0] * (m * actual_rank)
    for i in range(m):
        for j in range(actual_rank):
            data[i * actual_rank + j] = matrix[i, j]
    return DenseTensor((m, actual_rank), data=data)


def _truncate_rows(matrix, rank):
    """Первые rank строк."""
    k, n = matrix.shape
    actual_rank = min(rank, k)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        for j in range(n):
            data[i * n + j] = matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)


def _truncate_vector(vector, rank):
    """Первые rank элементов."""
    actual_rank = min(rank, vector.size)
    data = [vector.data[i] for i in range(actual_rank)]
    return DenseTensor((actual_rank,), data=data)


def _multiply_diag_matrix(diag_vec, matrix, rank):
    """diag(diag_vec) @ matrix."""
    m, n = matrix.shape
    actual_rank = min(rank, m, diag_vec.size)
    data = [0.0] * (actual_rank * n)
    for i in range(actual_rank):
        s = diag_vec.data[i]
        for j in range(n):
            data[i * n + j] = s * matrix[i, j]
    return DenseTensor((actual_rank, n), data=data)
