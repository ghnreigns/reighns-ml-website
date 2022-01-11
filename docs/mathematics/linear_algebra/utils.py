import numpy as np
import torch
from typing import List, Union, Tuple


def linear_combination_vectors(
    weights: List[float], *args: np.ndarray
) -> np.ndarray:
    """Computes the linear combination of vectors.

    Args:
        weights (List[float]): The set of weights corresponding to each vector.

    Returns:
        linear_weighted_sum (np.ndarray): The linear combination of vectors.

    Examples:
        >>> v1 = np.asarray([1, 2, 3, 4, 5]).reshape(-1, 1)
        >>> v2 = np.asarray([2, 4, 6, 8, 10]).reshape(-1, 1)
        >>> v3 = np.asarray([3, 6, 9, 12, 15]).reshape(-1, 1)
        >>> weights = [10, 20, 30]
        >>> linear_combination_vectors([10, 20, 30], v1, v2, v3)
    """

    linear_weighted_sum = np.zeros(shape=args[0].shape)
    for weight, vec in zip(weights, args):
        linear_weighted_sum += weight * vec
    return linear_weighted_sum


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


def check_matmul_shape(A: np.ndarray, B: np.ndarray) -> Tuple[int, int, int]:
    """Check if the shape of the matrices A and B are compatible for matrix multiplication.

    If A and B are of size (m, n) and (n, p), respectively, then the shape of the resulting matrix is (m, p).

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Raises:
        ValueError: Raises a ValueError if the shape of the matrices A and B are not compatible for matrix multiplication.

    Returns:
        (Tuple[int, int, int]): (m, n, p) where (m, n) is the shape of A and (n, p) is the shape of B.
    """

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"The number of columns of A must be equal to the number of rows of B, but got {A.shape[1]} and {B.shape[0]} respectively."
        )

    return (A.shape[0], A.shape[1], B.shape[1])


def np_matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """

    num_rows_A, common_index, num_cols_B = check_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # 1st loop: loops through first matrix A
    for i in range(num_rows_A):
        summation = 0
        # 2nd loop: loops through second matrix B
        for j in range(num_cols_B):
            # 3rd loop: computes dot prod
            for k in range(common_index):
                summation += A[i, k] * B[k, j]
                matmul[i, j] = summation

    return matmul


def np_matmul_element_wise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices using element wise method.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """

    num_rows_A, _, num_cols_B = check_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # 1st loop: loops through first matrix A
    for row_i in range(num_rows_A):
        # 2nd loop: loops through second matrix B
        for col_j in range(num_cols_B):
            # computes dot product of row i with column j of B and
            # assign the result to the element of the matrix matmul at row i and column j.
            matmul[row_i, col_j] = dot_product(A[row_i, :], B[:, col_j])

    return matmul


def np_matmul_column_wise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices using column wise method.

    Recall the section on Matrix Multiplication using Right Multiplication.

    Column i of C is represented by: Ab_i

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """
    num_rows_A, _, num_cols_B = check_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # we just need to populate the columns of C

    for col_i in range(matmul.shape[1]):
        # b_i
        col_i_B = B[:, col_i]
        # Ab_i
        linear_comb_A_on_col_i_B = linear_combination_vectors(col_i_B, *A.T)
        # C_i = Ab_i
        matmul[:, col_i] = linear_comb_A_on_col_i_B

    return matmul
