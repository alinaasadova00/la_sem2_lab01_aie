# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(shape):
    """Проверяет корректность формы и возвращает tuple."""
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape должен быть tuple или list")

    validated = tuple(shape)

    for dim in validated:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("все размеры shape должны быть положительными целыми")

    return validated


def compute_size(shape):
    """Возвращает общее число элементов."""
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape):
    """Возвращает strides для row-major (C-order) хранения."""
    if not shape:
        return ()

    d = len(shape)
    strides = [0] * d
    stride = 1
    strides[d - 1] = 1
    for k in range(d - 2, -1, -1):
        stride *= shape[k + 1]
        strides[k] = stride

    return tuple(strides)


def multi_index_to_flat(multi_index, strides):
    """Переводит мультииндекс в плоский индекс."""
    if len(multi_index) != len(strides):
        raise ValueError("размерность индекса и strides не совпадает")

    flat = 0
    for idx, stride in zip(multi_index, strides):
        flat += idx * stride

    return flat


def flat_to_multi_index(flat_index, shape):
    """Возвращает мультииндекс по плоскому индексу."""
    strides = compute_strides(shape)
    multi_index = []
    remaining = flat_index

    for stride in strides:
        multi_index.append(remaining // stride)
        remaining %= stride

    return tuple(multi_index)


def check_shapes_match(shape1, shape2):
    """Проверяет совпадение форм."""
    if shape1 != shape2:
        raise ValueError(f"Формы тензоров не совпадают: {shape1} и {shape2}")
