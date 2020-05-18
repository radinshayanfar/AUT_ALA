###################
#   Matrix Bases  #
# Radin Shayanfar #
#     9731032     #
###################

import numpy as np


def count_zeros(arr):
    zeros = []
    arr = np.round(arr, 5)
    for row in arr:
        for ind, element in enumerate(row):
            if element != 0:
                zeros.append(ind)
                break
            elif ind == arr.shape[1] - 1:
                zeros.append(ind + 1)
                break

    return zeros


def sort_by_zeros(arr):
    zeros = count_zeros(arr)
    arr_sorted = arr[np.argsort(zeros)]

    return np.sort(zeros), arr_sorted


def rr_down(sliced):
    m, n = sliced.shape
    row_zero = (sliced[0, :] / sliced[0, 0])[..., np.newaxis].T
    coefs_mat = np.repeat(a=row_zero, repeats=m - 1, axis=0) * (-1 * sliced[1:, 0][:, np.newaxis])
    sliced[1:, :] += coefs_mat


def rr_up(sliced):
    m, n = sliced.shape
    sliced[-1, :] /= sliced[-1, 0]
    row_last = sliced[-1, :][..., np.newaxis].T
    coefs_mat = np.repeat(a=row_last, repeats=m - 1, axis=0) * (-1 * sliced[:-1, 0][:, np.newaxis])
    sliced[:-1, :] += coefs_mat


def echelon_form(mat):
    mat = mat.copy()
    for i in range(mat.shape[0] - 1):
        sliced = mat[i:, :]
        zeros, sliced = sort_by_zeros(sliced)

        pivot_position = zeros[0]
        rr_down(sliced[:, pivot_position:])
        mat[i:, :] = sliced
    return mat


def reduced_echelon_form(mat):
    ef = mat.copy()
    zeros = count_zeros(ef)

    m, n = ef.shape
    for i in reversed(range(m)):
        sliced = ef[:i + 1, :]

        pivot_position = zeros[i]
        if pivot_position == n:
            continue

        rr_up(sliced[:, pivot_position:])
        ef[:i + 1, :] = sliced
    return ef


def get_null_space(ref, pivots):
    m, n = ref.shape
    null_space = np.zeros((n, n - pivots.shape[0]))
    for ref_row_index in range(pivots.shape[0]):
        pivot = pivots[ref_row_index]
        ref_row = ref[ref_row_index, pivot + 1:]
        for i, e in enumerate(ref_row[ref_row != 0]):
            null_space[pivot, pivot - ref_row_index + i] = -1 * e

    col = 0
    for row in [i for i in range(n) if i not in pivots.tolist()]:
        null_space[row, col] = 1
        col += 1
    return null_space


def get_row_space(ref, pivots):
    return ref[:pivots.shape[0], :].T


def get_col_space(mat, pivots):
    return mat[:, pivots]


def solve_equation(augmented):
    ef = echelon_form(augmented)
    return reduced_echelon_form(ef)[:augmented.shape[1] - 1, -1].reshape(-1, 1)


def ordinal_suffix(i):
    if i == 1:
        return "st"
    elif i == 2:
        return "nd"
    elif i == 3:
        return "rd"
    else:
        return "th"


def print_mat(matrix):
    print(matrix)
    print('==============')


if __name__ == '__main__':
    m, n = map(int, input("Enter dimension:\n").split())
    print("Enter A matrix:")
    mat = []
    for i in range(m):
        mat.append(list(map(float, input().split())))
    mat = np.array(mat, dtype=np.double)

    ef = echelon_form(mat)
    zeros = count_zeros(ef)
    ref = reduced_echelon_form(ef)
    print("A Reduced Echelon Form:")
    print_mat(np.concatenate((np.round(ref, 3), np.zeros((ref.shape[0], 1))), axis=1))

    pivots = np.array(zeros)
    pivots = pivots[pivots != n]

    null_space = get_null_space(ref, pivots)
    row_space = get_row_space(ref, pivots)
    col_space = get_col_space(mat, pivots)
    print("A Null Space Basis (Each column is a vector of basis)")
    print_mat(null_space)
    print("A Row Space Basis (Each column is a vector of basis)")
    print_mat(row_space)
    print("A Column Space Basis (Each column is a vector of basis)")
    print_mat(col_space)

    for col in [i for i in range(n) if i not in pivots.tolist()]:
        print(f"Left multiplying Col A Matrix (which printed above) by following vector results in {col + 1}{ordinal_suffix(col + 1)} column of A:")
        augmented = np.concatenate((col_space, mat[:, col][..., np.newaxis]), axis=1)
        print(solve_equation(augmented))
