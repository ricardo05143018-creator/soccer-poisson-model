"""
Bivariate Poisson match predictor with Dixon-Coles correction (V1.1).
Date: July 2026
"""

import math
import os
import numpy as np

# Hardcoded tournament averages from completed group stage
TOURNAMENT_AVG_HOME_GOALS = 1.42
TOURNAMENT_AVG_AWAY_GOALS = 1.18

# rough estimates from group-stage averages (manual tuning)
teams_data = {
    "Spain": {"attack": 1.55, "defense": 0.75},
    "Austria": {"attack": 1.10, "defense": 1.02}
}


def calculate_expected_goals(home_team, away_team):
    """compute expected goals (lambdas)"""
    home_lambda = teams_data[home_team]["attack"] * teams_data[away_team]["defense"] * TOURNAMENT_AVG_HOME_GOALS
    away_lambda = teams_data[away_team]["attack"] * teams_data[home_team]["defense"] * TOURNAMENT_AVG_AWAY_GOALS
    return home_lambda, away_lambda


def poisson_probability(lmbda, k):
    """poisson pmf"""
    if lmbda <= 0:
        return 0.0 if k > 0 else 1.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def dixon_coles_tau(i, j, lmbda_h, lmbda_a, rho=0.08):
    # simplified DC-style correction for low-score cells
    if (i, j) == (0, 0):
        return 1.0 - (lmbda_h * lmbda_a * rho)
    elif (i, j) == (1, 0):
        return 1.0 + (lmbda_a * rho)
    elif (i, j) == (0, 1):
        return 1.0 + (lmbda_h * rho)
    elif (i, j) == (1, 1):
        return 1.0 - rho
    return 1.0


def generate_score_matrix(lmbda_h, lmbda_a, max_goals=6):
    """build joint probability matrix with DC patch"""
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            base_prob = poisson_probability(lmbda_h, i) * poisson_probability(lmbda_a, j)
            correction = dixon_coles_tau(i, j, lmbda_h, lmbda_a)
            prob_matrix[i, j] = base_prob * correction

    prob_matrix = prob_matrix / np.sum(prob_matrix)
    return prob_matrix


# =============================================================
if __name__ == "__main__":

    home_team = "Spain"
    away_team = "Austria"

    print(f"Running Poisson Predictor v1.1 | {home_team} vs {away_team}")

    lmbda_h, lmbda_a = calculate_expected_goals(home_team, away_team)
    print(f"  lambda_Home: {lmbda_h:.4f}")
    print(f"  lambda_Away: {lmbda_a:.4f}\n")

    prob_matrix = generate_score_matrix(lmbda_h, lmbda_a, max_goals=6)

    home_win_p = 0.0
    away_win_p = 0.0
    draw_p = 0.0

    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            if i > j:
                home_win_p += prob_matrix[i, j]
            elif i < j:
                away_win_p += prob_matrix[i, j]
            else:
                draw_p += prob_matrix[i, j]

    print(f"=== {home_team} vs {away_team} Inference Summary ===")
    print(f"  Win Prob ({home_team}): {home_win_p * 100:.2f}%")
    print(f"  Win Prob ({away_team}): {away_win_p * 100:.2f}%")
    print(f"  Draw Prob:           {draw_p * 100:.2f}%")
    print("-" * 40)

    max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    print(f"  Most Likely Score: {max_idx[0]}-{max_idx[1]}")
    print(f"  Confidence:        {prob_matrix[max_idx] * 100:.2f}%")