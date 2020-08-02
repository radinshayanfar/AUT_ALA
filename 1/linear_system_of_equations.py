############################################
# n by n System of Linear Equations Solver #
#             Radin Shayanfar              #
#                 9731032                  #
############################################

from sys import exit
import numpy as np


def count_zeros(arr):
    zeros = []
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


def print_mat(matrix):
    print(matrix)
    print('==============')


def rr_down(sliced):
    sliced[0] = sliced[0] / sliced[0][0]
    for r_ind in range(1, sliced.shape[0]):
        coef = -1 * sliced[r_ind][0]
        sliced[r_ind] = sliced[r_ind] + coef * sliced[0]


def rr_up(sliced):
    for r_ind in range(0, sliced.shape[0] - 1):
        coef = -1 * sliced[r_ind][0]
        sliced[r_ind] = sliced[r_ind] + coef * sliced[-1]


def forward_phase(augmented):
    for i in range(augmented.shape[0]):
        sliced = augmented[i:, :]
        zeros, sliced = sort_by_zeros(sliced)
        if zeros[-1] == n + 1:
            print("Unlimited answers")
            exit(0)
        elif zeros[-1] == n:
            print("No answer")
            exit(0)
        pivot_position = zeros[0]
        rr_down(sliced[:, pivot_position:])
        augmented[i:, :] = sliced
        print_mat(augmented)


def backward_phase(augmented):
    for i in reversed(range(1, augmented.shape[0])):
        sliced = augmented[:i + 1, :]
        rr_up(sliced[:, i:])
        augmented[:i + 1, :] = sliced
        print_mat(augmented)


n = int(input("Enter dimension:\n"))
print("Enter coefficients matrix:")
coefs = []
for i in range(n):
    coefs.append(list(map(float, input().split())))
b = list(map(float, input("Enter vector b:\n").split()))

coefs = np.array(coefs, dtype=np.double)
b = np.array([b], dtype=np.double).T

augmented = np.concatenate((coefs, b), axis=1)
print("Augmented Matrix:")
print(augmented)

print("Forward Phase:")
forward_phase(augmented)
print("Backward Phase:")
backward_phase(augmented)

for i in range(n):
    print(f"x{i + 1}:\t{augmented[i, -1]}")
