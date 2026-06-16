# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math
from typing import Any

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """Плотный тензор произвольного порядка."""

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    def __init__(self, shape, data=None, fill=0.0):
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [float(fill)] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(
                    f"размер data ({len(data)}) не совпадает с shape {self.shape}"
                )
            self.data = [float(x) for x in data]

    @staticmethod
    def zeros(shape):
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape):
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(shape, low=-5, high=5, integer=True, seed=None):
        if seed is not None:
            random.seed(seed)

        shape = validate_shape(shape)
        size = compute_size(shape)

        if integer:
            data = [float(random.randint(low, high)) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]

        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested):
        """Создаёт тензор из вложенного списка."""
        shape = DenseTensor._infer_shape(nested)
        data = DenseTensor._flatten_nested(nested)
        return DenseTensor(shape, data=data)

    @staticmethod
    def _infer_shape(obj):
        if not isinstance(obj, list):
            return ()
        if not obj:
            return (0,)

        sub_shape = DenseTensor._infer_shape(obj[0])
        for item in obj[1:]:
            if DenseTensor._infer_shape(item) != sub_shape:
                raise ValueError("вложенный список неоднороден")

        return (len(obj),) + sub_shape

    @staticmethod
    def _flatten_nested(obj):
        if not isinstance(obj, list):
            return [float(obj)]

        result = []
        for item in obj:
            result.extend(DenseTensor._flatten_nested(item))
        return result

    def _normalize_index(self, multi_index):
        """Нормализует индекс: int/tuple/list/slice."""
        if isinstance(multi_index, (int, slice)):
            if self.ndim == 1:
                return (multi_index,)
            if self.ndim == 0:
                return ()
            raise IndexError(
                "скалярный индекс/срез допустим только для 1D тензора"
            )

        if isinstance(multi_index, list):
            multi_index = tuple(multi_index)

        if not isinstance(multi_index, tuple):
            raise IndexError("индекс должен быть int, tuple, list или slice")

        if len(multi_index) != self.ndim:
            raise IndexError(
                "размерность индекса не совпадает с порядком тензора"
            )

        return multi_index

    def _validate_scalar_index(self, multi_index):
        normalized = self._normalize_index(multi_index)
        for idx in normalized:
            if isinstance(idx, slice):
                raise IndexError("установка значения не поддерживает срезы")
        for idx, dim in zip(normalized, self.shape):
            if idx < 0 or idx >= dim:
                raise IndexError(f"индекс {idx} выходит за границы {dim}")
        return normalized

    def __getitem__(self, multi_index):
        normalized = self._normalize_index(multi_index)

        for idx, dim in zip(normalized, self.shape):
            if isinstance(idx, int) and (idx < 0 or idx >= dim):
                raise IndexError(f"индекс {idx} выходит за границы {dim}")

        if all(isinstance(idx, int) for idx in normalized):
            flat = multi_index_to_flat(normalized, self.strides)
            return self.data[flat]

        # возвращаем подтензор
        out_shape = []
        ranges = []
        for idx, dim in zip(normalized, self.shape):
            if isinstance(idx, int):
                ranges.append(None)
            else:
                start, stop, step = idx.indices(dim)
                if step <= 0:
                    raise ValueError("шаг среза должен быть положительным")
                rng = list(range(start, stop, step))
                out_shape.append(len(rng))
                ranges.append(rng)

        out_shape = tuple(out_shape)
        out_size = compute_size(out_shape)
        out_strides = compute_strides(out_shape)
        out_data = [0.0] * out_size

        for flat_out in range(out_size):
            multi_out = flat_to_multi_index(flat_out, out_shape)
            multi_in = []
            out_pos = 0
            for d in range(self.ndim):
                if ranges[d] is None:
                    multi_in.append(normalized[d])
                else:
                    multi_in.append(ranges[d][multi_out[out_pos]])
                    out_pos += 1
            flat_in = multi_index_to_flat(tuple(multi_in), self.strides)
            out_data[flat_out] = self.data[flat_in]

        return DenseTensor(out_shape, data=out_data)

    def __setitem__(self, multi_index, value):
        index = self._validate_scalar_index(multi_index)
        flat = multi_index_to_flat(index, self.strides)
        self.data[flat] = float(value)

    def reshape(self, new_shape):
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError(
                f"несовместимые формы для reshape: {self.shape} -> {new_shape}"
            )
        return DenseTensor(new_shape, data=self.data[:])

    def unfolding(self, mode):
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode должен быть в [0, {self.ndim}), получен {mode}")

        n_rows = self.shape[mode]
        n_cols = self.size // n_rows
        result = DenseTensor.zeros((n_rows, n_cols))

        shape_without = self.shape[:mode] + self.shape[mode + 1:]
        strides_without = compute_strides(shape_without)

        for flat in range(self.size):
            multi = flat_to_multi_index(flat, self.shape)
            row = multi[mode]
            col_multi = multi[:mode] + multi[mode + 1:]
            col = multi_index_to_flat(col_multi, strides_without)
            result[row, col] = self.data[flat]

        return result

    def left_unfolding(self, k):
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(
                f"k должен быть в [0, {self.ndim - 1}), получен {k}"
            )

        n_rows = 1
        for dim in self.shape[:k + 1]:
            n_rows *= dim
        n_cols = self.size // n_rows
        return self.reshape((n_rows, n_cols))

    def copy(self):
        return DenseTensor(self.shape, data=self.data[:])

    def norm(self):
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other):
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(
            self.shape,
            data=[a + b for a, b in zip(self.data, other.data)]
        )

    def __sub__(self, other):
        check_shapes_match(self.shape, other.shape)
        return DenseTensor(
            self.shape,
            data=[a - b for a, b in zip(self.data, other.data)]
        )

    def __mul__(self, scalar):
        return DenseTensor(
            self.shape,
            data=[x * float(scalar) for x in self.data]
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return self.__mul__(-1.0)

    def allclose(self, other, atol=1e-8, rtol=1e-5):
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self):
        return self._to_nested_list(self.shape, self.data)

    def _to_nested_list(self, shape, data):
        if not shape:
            return data[0]
        if len(shape) == 1:
            return list(data)

        n = shape[0]
        sub_size = 1
        for dim in shape[1:]:
            sub_size *= dim
        result = []
        for i in range(n):
            result.append(
                self._to_nested_list(shape[1:], data[i * sub_size:(i + 1) * sub_size])
            )
        return result

    def __repr__(self):
        if self.size <= 20:
            data_repr = str(self.data)
        else:
            data_repr = f"[... {self.size} элементов ...]"
        return f"DenseTensor(shape={self.shape}, data={data_repr})"

    def __str__(self):
        return self.__repr__()
