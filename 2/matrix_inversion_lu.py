###############################################
# Matrix inversion by LU factorization method #
#               Radin Shayanfar               #
#                   9731032                   #
###############################################

from typing import Any, Tuple

import numpy as np


def lu_factor(aMat: np.ndarray) -> Tuple[Any, Any]:
    n = aMat.shape[0]
    uMat = aMat.copy()
    lMat = np.eye(n, dtype=np.double)
    for i in range(n):
        sliced = uMat[i:, i:]

        lMat[i + 1:, i] = sliced[1:, 0] / sliced[0][0]
        for r_ind in range(1, sliced.shape[0]):
            coef = -1 * sliced[r_ind][0] / sliced[0][0]
            sliced[r_ind] = sliced[r_ind] + coef * sliced[0]

        uMat[i:, i:] = sliced

    return lMat, uMat


def forward_sub(augmented: np.ndarray) -> np.ndarray:
    for i in range(augmented.shape[0]):
        augmented[i + 1:, -1] -= augmented[i + 1:, i] * augmented[i, -1]

    return augmented[:, -1]


def backward_sub(augmented: np.ndarray) -> np.ndarray:
    for i in reversed(range(augmented.shape[0])):
        augmented[i, -1] /= augmented[i, i]
        augmented[:i, -1] -= augmented[:i, i] * augmented[i, -1]

    return augmented[:, -1]


def solve_with_lu(l: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    augmented = np.concatenate((l, b[..., np.newaxis]), axis=1)
    y = forward_sub(augmented)
    augmented = np.concatenate((u, y[..., np.newaxis]), axis=1)
    return backward_sub(augmented)


def inverse_matrix(l: np.ndarray, u: np.ndarray) -> np.ndarray:
    n = l.shape[0]
    I = np.eye(n)
    inv = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        inv[:, i] = solve_with_lu(l, u, I[:, i])

    return inv


if __name__ == '__main__':
    n = int(input("Enter dimension:\n"))
    print("Enter A matrix:")
    a = []
    for i in range(n):
        a.append(list(map(float, input().split())))
    a = np.array(a, dtype=np.double)

    l, u = lu_factor(a)
    print(f"L matrix:\n{l}")
    print("--------------")
    print(f"U matrix:\n{u}")
    print("--------------")

    print(f"A inverse:\n{inverse_matrix(l, u)}")
