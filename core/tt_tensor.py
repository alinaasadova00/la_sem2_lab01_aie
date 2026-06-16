# core/tt_tensor.py

"""Тензор в TT-формате (Tensor Train)."""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.linalg import matmul
from core.utils import validate_shape, compute_size


class TTTensor:
    """Тензор в TT-формате."""

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    def __init__(self, cores):
        if not isinstance(cores, list) or not cores:
            raise ValueError("cores должен быть непустым списком DenseTensor")

        validated = []
        for core in cores:
            if not isinstance(core, DenseTensor) or core.ndim != 3:
                raise ValueError("каждое ядро должно быть 3D DenseTensor")
            validated.append(core.copy())

        order = len(validated)
        shape = []
        ranks = []

        for k, core in enumerate(validated):
            r_prev, n_k, r_next = core.shape
            shape.append(n_k)
            ranks.append(r_prev)
            if k > 0:
                prev_r_next = validated[k - 1].shape[2]
                if r_prev != prev_r_next:
                    raise ValueError(
                        f"несогласованные ранги между ядрами {k - 1} и {k}"
                    )

        ranks.append(validated[-1].shape[2])

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        self.cores = validated
        self.order = order
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)

    @staticmethod
    def random(shape, ranks, seed=None):
        if seed is not None:
            random.seed(seed)

        shape = validate_shape(shape)
        order = len(shape)

        if len(ranks) == order + 1:
            ranks = tuple(ranks)
        elif len(ranks) == order - 1:
            ranks = (1,) + tuple(ranks) + (1,)
        else:
            raise ValueError("некорректная длина ranks")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        cores = []
        for k in range(order):
            core_shape = (ranks[k], shape[k], ranks[k + 1])
            cores.append(DenseTensor.random(core_shape, integer=False))

        return TTTensor(cores)

    def get_element(self, indices):
        if len(indices) != self.order:
            raise ValueError("длина индекса не совпадает с порядком тензора")

        vector = [1.0]

        for k, idx in enumerate(indices):
            core = self.cores[k]
            r_prev = core.shape[0]
            r_next = core.shape[2]
            new_vector = [0.0] * r_next

            for a in range(r_prev):
                for b in range(r_next):
                    new_vector[b] += vector[a] * core[a, idx, b]

            vector = new_vector

        return vector[0]

    def full(self):
        if self.order == 0:
            return DenseTensor(())

        current = self.cores[0].reshape((self.shape[0], self.ranks[1]))

        for k in range(1, self.order):
            core = self.cores[k]
            r_k = core.shape[0]
            n_k = core.shape[1]
            r_next = core.shape[2]
            core_matrix = core.reshape((r_k, n_k * r_next))
            current = matmul(current, core_matrix)
            current = current.reshape((current.shape[0] * n_k, r_next))

        return current.reshape(self.shape)

    def core_sizes(self):
        return [core.shape for core in self.cores]

    def total_storage(self):
        return sum(core.size for core in self.cores)

    def compression_ratio(self):
        full_size = compute_size(self.shape)
        storage = self.total_storage()
        if storage == 0:
            return float('inf')
        return full_size / storage

    def copy(self):
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self):
        return (
            f"TTTensor(order={self.order}, shape={self.shape}, "
            f"ranks={self.ranks}, storage={self.total_storage()})"
        )

    def __str__(self):
        return self.__repr__()
