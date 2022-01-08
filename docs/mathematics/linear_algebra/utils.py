import numpy as np
from typing import List, Union


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the dot product of two vectors.

    We assume both vectors are flattened, i.e. they are 1D arrays.

    Args:
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        dot_product_v1_v2 (float): The dot product of two vectors.

    Examples:
        >>> v1 = np.asarray([1, 2, 3, 4, 5])
        >>> v2 = np.asarray([2, 4, 6, 8, 10])
        >>> dot_product(v1, v2)
    """

    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()

    dot_product_v1_v2 = 0
    for element_1, element_2 in zip(v1, v2):
        dot_product_v1_v2 += element_1 * element_2

    # same as np.dot but does not take into the orientation of vectors
    assert dot_product_v1_v2 == np.dot(v1.T, v2)

    return dot_product_v1_v2


def average_set(vec: Union[np.ndarray, set]) -> float:
    """Average a set of numbers using dot product.

    Given a set of numbers {v1, v2, ..., vn}, the average is defined as:
    avg = (v1 + v2 + ... + vn) / n

    To use the dot product, we can convert the set to a col/row vector (array) `vec` and
    perform the dot product with the vector of ones to get `sum(set)`. Lastly, we divide by the number of elements in the set.

    Args:
        vec (Union[np.ndarray, set]): A set of numbers.

    Returns:
        average (float): The average of the set.

    Examples:
        >>> v = np.asarray([1, 2, 3, 4, 5])
        >>> v_set = {1,2,3,4,5} # same as v but as a set.
        >>> average = average_set(v_set)
        >>> average = 3.0
    """

    if isinstance(vec, set):
        vec = np.asarray(list(vec)).flatten()

    ones = np.ones(shape=vec.shape)
    total_sum = dot_product(vec, ones)
    average = total_sum / vec.shape[0]

    assert average == np.mean(vec)
    return average
