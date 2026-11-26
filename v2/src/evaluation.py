"""
Soccer model v2 prediction evaluation.
Builds score probabilities and evaluates 90-min match outcomes.
Date: November 2026
"""

import math
import numpy as np


def poisson_pmf(lmbda, k):
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def dixon_coles_tau(i, j, lh, la, rho):
    if (i, j) == (0, 0):
        return 1.0 + lh * la * rho
    if (i, j) == (1, 0):
        return 1.0 - la * rho
    if (i, j) == (0, 1):
        return 1.0 - lh * rho
    if (i, j) == (1, 1):
        return 1.0 + rho
    return 1.0


def exact_match_probability(h_goals, a_goals, lh, la, rho):
    base_prob = poisson_pmf(lh, h_goals) * poisson_pmf(la, a_goals)
    tau = dixon_coles_tau(h_goals, a_goals, lh, la, rho)
    return max(base_prob * tau, 1e-12)


def score_matrix(lh, la, rho, max_goals=6):
    mat = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            base = poisson_pmf(lh, i) * poisson_pmf(la, j)
            mat[i, j] = base * dixon_coles_tau(i, j, lh, la, rho)

    # normalize after the finite score cutoff
    return mat / mat.sum()


def outcome_probs(mat):
    home = away = draw = 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i > j:
                home += mat[i, j]
            elif i < j:
                away += mat[i, j]
            else:
                draw += mat[i, j]

    return home, away, draw


def actual_outcome(h_goals, a_goals):
    if h_goals > a_goals:
        return 0
    if h_goals < a_goals:
        return 1
    return 2


def brier_score(probs, h_goals, a_goals):
    actual = [0.0, 0.0, 0.0]
    actual[actual_outcome(h_goals, a_goals)] = 1.0
    return sum((p - o) ** 2 for p, o in zip(probs, actual))


def log_loss(probs, h_goals, a_goals):
    outcome_idx = actual_outcome(h_goals, a_goals)
    return -math.log(max(probs[outcome_idx], 1e-12))