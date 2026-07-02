"""
Poisson predictor with form decay overlay (V1.2).
Also hotfixes the DC sign bug from v1.1.
Date: July 2026
"""

import math
import os
import numpy as np

# Hardcoded tournament averages from completed group stage
TOURNAMENT_AVG_HOME_GOALS = 1.42
TOURNAMENT_AVG_AWAY_GOALS = 1.18

RHO = 0.08
DECAY_RATE = 0.15  # parameter for form decay weight

# Rough estimates locked onto Spain vs Austria, overlaid with recent form shifters
teams_data = {
    "Spain": {"attack": 1.55, "defense": 0.75, "form": 1.10},       # group stage: 3W 0D 0L
    "Austria": {"attack": 1.10, "defense": 1.02, "form": 0.95}      # dropped points late in group stage
}


def apply_form_decay(team):
    """apply form decay multiplier"""
    form = teams_data[team]["form"]
    return 1.0 + DECAY_RATE * (form - 1.0)


def calculate_expected_goals(home_team, away_team):
    """compute expected goals (lambdas) factoring in form decay"""
    home_form = apply_form_decay(home_team)
    away_form = apply_form_decay(away_team)

    home_lambda = teams_data[home_team]["attack"] * teams_data[away_team]["defense"] * home_form * TOURNAMENT_AVG_HOME_GOALS
    away_lambda = teams_data[away_team]["attack"] * teams_data[home_team]["defense"] * away_form * TOURNAMENT_AVG_AWAY_GOALS
    return home_lambda, away_lambda


def poisson_probability(lmbda, k):
    """poisson pmf"""
    if lmbda <= 0:
        return 0.0 if k > 0 else 1.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def dixon_coles_tau(i, j, lmbda_h, lmbda_a):
    # HOTFIX: Reverted the signs from v1.1.
    # Draws (0,0) and (1,1) must be inflated (+), non-draws (1,0) and (0,1) deflated (-).
    if (i, j) == (0, 0):
        return 1.0 + (lmbda_h * lmbda_a * RHO)
    elif (i, j) == (1, 0):
        return 1.0 - (lmbda_a * RHO)
    elif (i, j) == (0, 1):
        return 1.0 - (lmbda_h * RHO)
    elif (i, j) == (1, 1):
        return 1.0 + RHO
    return 1.0


def generate_score_matrix(lmbda_h, lmbda_a, max_goals=6):
    """build joint probability matrix"""
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            base_prob = poisson_probability(lmbda_h, i) * poisson_probability(lmbda_a, j)
            prob_matrix[i, j] = base_prob * dixon_coles_tau(i, j, lmbda_h, lmbda_a)

    # renormalize
    prob_matrix = prob_matrix / np.sum(prob_matrix)
    return prob_matrix


# =============================================================
if __name__ == "__main__":

    home_team = "Spain"
    away_team = "Austria"

    print(f"Running Poisson Predictor v1.2 | {home_team} vs {away_team}")

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