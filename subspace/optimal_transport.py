import numpy as np
from scipy.optimize import linprog


def wasserstein_distance(p, q, D):
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


def word_rotator_distance(x, y):
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)


def word_mover_distance(x, y):
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    D = np.sqrt(np.square(x[:, None] - y[None, :]).mean(axis=2))
    return wasserstein_distance(p, q, D)


def word_rotator_similarity(x, y):
    return 1 - word_rotator_distance(x, y)


def word_mover_similarity(x, y):
    return 1 - word_mover_distance(x, y)


