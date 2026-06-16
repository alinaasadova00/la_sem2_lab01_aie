# algorithms/tensor_operations.py

"""Базовые операции с TT-тензорами без восстановления полного тензора."""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(tt1, tt2, backend):
    """Поэлементное сложение двух TT-тензоров."""
    if tt1.shape != tt2.shape:
        raise ValueError("формы тензоров не совпадают")

    order = tt1.order
    new_cores = []

    for k in range(order):
        core_a = tt1.cores[k]
        core_b = tt2.cores[k]
        r_a_prev, n_k, r_a_next = core_a.shape
        r_b_prev, _, r_b_next = core_b.shape

        if k == 0:
            new_core = DenseTensor.zeros((1, n_k, r_a_next + r_b_next))
            for i in range(n_k):
                for b in range(r_a_next):
                    new_core[0, i, b] = core_a[0, i, b]
                for b in range(r_b_next):
                    new_core[0, i, r_a_next + b] = core_b[0, i, b]
        elif k == order - 1:
            new_core = DenseTensor.zeros((r_a_prev + r_b_prev, n_k, 1))
            for i in range(n_k):
                for a in range(r_a_prev):
                    new_core[a, i, 0] = core_a[a, i, 0]
                for a in range(r_b_prev):
                    new_core[r_a_prev + a, i, 0] = core_b[a, i, 0]
        else:
            new_core = DenseTensor.zeros(
                (r_a_prev + r_b_prev, n_k, r_a_next + r_b_next)
            )
            for i in range(n_k):
                for a in range(r_a_prev):
                    for b in range(r_a_next):
                        new_core[a, i, b] = core_a[a, i, b]
                for a in range(r_b_prev):
                    for b in range(r_b_next):
                        new_core[r_a_prev + a, i, r_a_next + b] = core_b[a, i, b]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(tt, alpha, backend):
    """Умножение TT-тензора на скаляр (меняем только первое ядро)."""
    alpha = float(alpha)
    new_cores = [core.copy() for core in tt.cores]
    first = new_cores[0]
    r_prev, n, r_next = first.shape

    for a in range(r_prev):
        for i in range(n):
            for b in range(r_next):
                first[a, i, b] *= alpha

    return TTTensor(new_cores)


def tt_hadamard(tt1, tt2, backend):
    """Поэлементное произведение (Адамар)."""
    if tt1.shape != tt2.shape:
        raise ValueError("формы тензоров не совпадают")

    order = tt1.order
    new_cores = []

    for k in range(order):
        core_a = tt1.cores[k]
        core_b = tt2.cores[k]
        r_a_prev, n_k, r_a_next = core_a.shape
        r_b_prev, _, r_b_next = core_b.shape

        new_r_prev = r_a_prev * r_b_prev
        new_r_next = r_a_next * r_b_next
        new_core = DenseTensor.zeros((new_r_prev, n_k, new_r_next))

        for i in range(n_k):
            for a in range(r_a_prev):
                for b in range(r_a_next):
                    val_a = core_a[a, i, b]
                    for ib in range(r_b_prev):
                        for jb in range(r_b_next):
                            row = a * r_b_prev + ib
                            col = b * r_b_next + jb
                            new_core[row, i, col] = val_a * core_b[ib, i, jb]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_dot(tt1, tt2, backend):
    """Скалярное произведение <tt1, tt2>."""
    if tt1.shape != tt2.shape:
        raise ValueError("формы тензоров не совпадают")

    order = tt1.order
    Z = None

    for k in range(order):
        core_a = tt1.cores[k]
        core_b = tt2.cores[k]
        n_k = core_a.shape[1]

        if k == 0:
            r_a_next = core_a.shape[2]
            r_b_next = core_b.shape[2]
            Z_data = [0.0] * (r_a_next * r_b_next)
            for i in range(n_k):
                for ba in range(r_a_next):
                    for bb in range(r_b_next):
                        Z_data[ba * r_b_next + bb] += core_a[0, i, ba] * core_b[0, i, bb]
            Z = DenseTensor((r_a_next, r_b_next), data=Z_data)
        else:
            r_a_prev = core_a.shape[0]
            r_a_next = core_a.shape[2]
            r_b_prev = core_b.shape[0]
            r_b_next = core_b.shape[2]
            Z_new = [0.0] * (r_a_next * r_b_next)

            for i in range(n_k):
                for aa in range(r_a_prev):
                    for ba in range(r_a_next):
                        for ab in range(r_b_prev):
                            for bb in range(r_b_next):
                                Z_new[ba * r_b_next + bb] += (
                                    core_a[aa, i, ba] * Z[aa, ab] * core_b[ab, i, bb]
                                )

            Z = DenseTensor((r_a_next, r_b_next), data=Z_new)

    return Z[0, 0]


def tt_norm(tt, backend):
    """Фробениусова норма TT-тензора."""
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(tt1, tt2, backend):
    """Норма разности ||tt1 - tt2||_F без восстановления полных тензоров."""
    n1 = tt_norm(tt1, backend)
    n2 = tt_norm(tt2, backend)
    d = tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0.0, n1 * n1 + n2 * n2 - 2.0 * d))
