"""
Poisson predictor with a quick backtest loop (V1.3).
Tracks Brier scores across completed matches.
Date: July 2026
"""

import math
import numpy as np

TOURNAMENT_AVG_HOME_GOALS = 1.42
TOURNAMENT_AVG_AWAY_GOALS = 1.18
DECAY_RATE = 0.05
RHO = 0.08

teams_data = {
    "South Africa":  {"attack": 0.72, "defense": 1.18, "form": 0.88},
    "Canada":        {"attack": 0.95, "defense": 1.05, "form": 0.97},
    "Brazil":        {"attack": 1.45, "defense": 0.80, "form": 1.08},
    "Japan":         {"attack": 1.05, "defense": 0.98, "form": 0.96},
    "Germany":       {"attack": 1.35, "defense": 0.88, "form": 1.02},
    "Paraguay":      {"attack": 0.95, "defense": 1.05, "form": 0.94},
    "Netherlands":   {"attack": 1.30, "defense": 0.90, "form": 1.00},
    "Morocco":       {"attack": 1.00, "defense": 0.95, "form": 0.98},
    "Ivory Coast":   {"attack": 1.10, "defense": 1.00, "form": 0.95},
    "Norway":        {"attack": 1.35, "defense": 0.88, "form": 1.05},
    "France":        {"attack": 1.55, "defense": 0.78, "form": 1.12},
    "Sweden":        {"attack": 1.05, "defense": 1.02, "form": 0.93},
    "Mexico":        {"attack": 1.10, "defense": 0.98, "form": 1.00},
    "Ecuador":       {"attack": 1.00, "defense": 1.05, "form": 0.95},
    "England":       {"attack": 1.40, "defense": 0.85, "form": 1.05},
    "Congo":         {"attack": 0.85, "defense": 1.10, "form": 0.90},
    "Belgium":       {"attack": 1.45, "defense": 0.82, "form": 1.08},
    "Senegal":       {"attack": 1.15, "defense": 0.92, "form": 0.96},
    "USA":           {"attack": 1.20, "defense": 0.95, "form": 1.03},
    "Bosnia":        {"attack": 1.00, "defense": 1.08, "form": 0.92},
    "Spain":         {"attack": 1.55, "defense": 0.75, "form": 1.10},
    "Austria":       {"attack": 1.10, "defense": 1.02, "form": 0.95},
    "Portugal":      {"attack": 1.50, "defense": 0.80, "form": 1.08},
    "Croatia":       {"attack": 1.20, "defense": 0.92, "form": 1.00},
}

historical_matches = [
    {"home": "South Africa", "away": "Canada",      "home_goals": 0, "away_goals": 1},
    {"home": "Brazil",       "away": "Japan",        "home_goals": 2, "away_goals": 1},
    {"home": "Germany",      "away": "Paraguay",     "home_goals": 1, "away_goals": 1},
    {"home": "Netherlands",  "away": "Morocco",      "home_goals": 1, "away_goals": 1},
    {"home": "Ivory Coast",  "away": "Norway",       "home_goals": 1, "away_goals": 2},
    {"home": "France",       "away": "Sweden",       "home_goals": 3, "away_goals": 0},
    {"home": "Mexico",       "away": "Ecuador",      "home_goals": 2, "away_goals": 0},
    {"home": "England",      "away": "Congo",        "home_goals": 2, "away_goals": 1},
    {"home": "Belgium",      "away": "Senegal",      "home_goals": 3, "away_goals": 2},
    {"home": "USA",          "away": "Bosnia",       "home_goals": 2, "away_goals": 0},
    {"home": "Spain",        "away": "Austria",      "home_goals": 3, "away_goals": 0},
    {"home": "Portugal",     "away": "Croatia",      "home_goals": 2, "away_goals": 1},
]


def apply_form_decay(team):
    form = teams_data[team]["form"]
    return 1.0 + DECAY_RATE * (form - 1.0)


def calculate_expected_goals(home_team, away_team):
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


def dixon_coles_tau(i, j, lmbda_h, lmbda_a, rho=RHO):
    if (i, j) == (0, 0):
        return 1.0 + (lmbda_h * lmbda_a * rho)
    elif (i, j) == (1, 0):
        return 1.0 - (lmbda_a * rho)
    elif (i, j) == (0, 1):
        return 1.0 - (lmbda_h * rho)
    elif (i, j) == (1, 1):
        return 1.0 + rho
    return 1.0


def generate_score_matrix(lmbda_h, lmbda_a, max_goals=6):
    """build joint probability matrix"""
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            base_prob = poisson_probability(lmbda_h, i) * poisson_probability(lmbda_a, j)
            correction = dixon_coles_tau(i, j, lmbda_h, lmbda_a)
            prob_matrix[i, j] = base_prob * correction
    prob_matrix = prob_matrix / np.sum(prob_matrix)
    return prob_matrix


def get_outcome_probs(prob_matrix):
    h, a, d = 0.0, 0.0, 0.0
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            if i > j:
                h += prob_matrix[i, j]
            elif i < j:
                a += prob_matrix[i, j]
            else:
                d += prob_matrix[i, j]
    return h, a, d


def brier_score(prob_matrix, actual_home, actual_away):
    # multi-class brier: one-hot encode actual outcome
    if actual_home > actual_away:
        actual = [1, 0, 0]
    elif actual_home < actual_away:
        actual = [0, 1, 0]
    else:
        actual = [0, 0, 1]
    h, a, d = get_outcome_probs(prob_matrix)
    predicted = [h, a, d]
    return sum((p - o) ** 2 for p, o in zip(predicted, actual))


if __name__ == "__main__":

    print("=== Backtest ===")
    brier_scores = []

    for match in historical_matches:
        lh, la = calculate_expected_goals(match["home"], match["away"])
        matrix = generate_score_matrix(lh, la)
        bs = brier_score(matrix, match["home_goals"], match["away_goals"])
        brier_scores.append(bs)
        h, a, d = get_outcome_probs(matrix)
        print(f"  {match['home']} vs {match['away']} | "
              f"H:{h:.2f} A:{a:.2f} D:{d:.2f} | Brier: {bs:.4f}")

    print(f"\n  Avg Brier Score: {sum(brier_scores)/len(brier_scores):.4f}")
    print("-" * 40)

    home_team = "Paraguay"
    away_team = "France"

    print(f"\nRunning Poisson Predictor v1.3 | {home_team} vs {away_team}")
    lh, la = calculate_expected_goals(home_team, away_team)
    print(f"  lambda_Home: {lh:.4f}")
    print(f"  lambda_Away: {la:.4f}\n")

    prob_matrix = generate_score_matrix(lh, la)
    h, a, d = get_outcome_probs(prob_matrix)

    print(f"=== {home_team} vs {away_team} Inference Summary ===")
    print(f"  Win Prob ({home_team}): {h * 100:.2f}%")
    print(f"  Win Prob ({away_team}): {a * 100:.2f}%")
    print(f"  Draw Prob:           {d * 100:.2f}%")
    print("-" * 40)

    max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    print(f"  Most Likely Score: {max_idx[0]}-{max_idx[1]}")
    print(f"  Confidence:        {prob_matrix[max_idx] * 100:.2f}%")