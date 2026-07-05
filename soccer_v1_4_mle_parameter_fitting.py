"""
Poisson predictor with MLE parameter calibration (V1.4).
Optimizes rho and decay_rate via SLSQP.
Date: July 2026
"""

import math
import numpy as np
from scipy.optimize import minimize

TOURNAMENT_AVG_HOME_GOALS = 1.42
TOURNAMENT_AVG_AWAY_GOALS = 1.18
LOCAL_DEV = True

# rough estimates from group-stage averages (manual tuning)
teams_data = {
    "Spain":        {"attack": 1.55, "defense": 0.75, "form": 1.10},
    "Austria":      {"attack": 1.10, "defense": 1.02, "form": 0.95},
    "Portugal":     {"attack": 1.50, "defense": 0.80, "form": 1.08},
    "Croatia":      {"attack": 1.20, "defense": 0.92, "form": 1.00},
    "Switzerland":  {"attack": 1.15, "defense": 0.95, "form": 1.02},
    "Algeria":      {"attack": 0.90, "defense": 1.08, "form": 0.91},
    "Australia":    {"attack": 0.95, "defense": 1.05, "form": 0.96},
    "Egypt":        {"attack": 0.90, "defense": 1.05, "form": 0.93},
    "Argentina":    {"attack": 1.50, "defense": 0.82, "form": 1.10},
    "Cape Verde":   {"attack": 0.80, "defense": 1.12, "form": 0.88},
    "Colombia":     {"attack": 1.25, "defense": 0.95, "form": 1.00},
    "Ghana":        {"attack": 0.90, "defense": 1.08, "form": 0.92},
    "Canada":       {"attack": 0.95, "defense": 1.05, "form": 0.97},
    "Morocco":      {"attack": 1.00, "defense": 0.95, "form": 0.98},
    "Paraguay":     {"attack": 0.95, "defense": 1.05, "form": 0.94},
    "France":       {"attack": 1.55, "defense": 0.78, "form": 1.12},
    "Brazil":       {"attack": 1.45, "defense": 0.80, "form": 1.08},
    "Norway":       {"attack": 1.35, "defense": 0.88, "form": 1.05},
}


# Regular time only. Ignore penalty shootouts.
historical_matches = [
    {"home": "Spain",       "away": "Austria",    "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Portugal",    "away": "Croatia",    "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Switzerland", "away": "Algeria",    "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Australia",   "away": "Egypt",      "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Argentina",   "away": "Cape Verde", "h_goals": 3, "a_goals": 2, "result": "H"},
    {"home": "Colombia",    "away": "Ghana",      "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Canada",      "away": "Morocco",    "h_goals": 0, "a_goals": 3, "result": "A"},
    {"home": "Paraguay",    "away": "France",     "h_goals": 0, "a_goals": 1, "result": "A"},
]


def apply_form_decay(team, decay_rate):
    form = teams_data[team]["form"]
    return 1.0 + decay_rate * (form - 1.0)


def calculate_expected_goals(home_team, away_team, decay_rate):
    home_form = apply_form_decay(home_team, decay_rate)
    away_form = apply_form_decay(away_team, decay_rate)
    home_lambda = teams_data[home_team]["attack"] * teams_data[away_team]["defense"] * home_form * TOURNAMENT_AVG_HOME_GOALS
    away_lambda = teams_data[away_team]["attack"] * teams_data[home_team]["defense"] * away_form * TOURNAMENT_AVG_AWAY_GOALS
    return home_lambda, away_lambda


def poisson_pmf(lmbda, k):
    """poisson pmf"""
    if lmbda <= 0:
        return 0.0 if k > 0 else 1.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def dixon_coles_tau(i, j, lmbda_h, lmbda_a, rho):
    # DC correction for low-score cells
    if (i, j) == (0, 0):
        return 1.0 + (lmbda_h * lmbda_a * rho)
    elif (i, j) == (1, 0):
        return 1.0 - (lmbda_a * rho)
    elif (i, j) == (0, 1):
        return 1.0 - (lmbda_h * rho)
    elif (i, j) == (1, 1):
        return 1.0 + rho
    return 1.0


def build_score_matrix(lmbda_h, lmbda_a, rho, max_goals=6):
    """build joint probability matrix"""
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            base_prob = poisson_pmf(lmbda_h, i) * poisson_pmf(lmbda_a, j)
            prob_matrix[i, j] = base_prob * dixon_coles_tau(i, j, lmbda_h, lmbda_a, rho)
    return prob_matrix / np.sum(prob_matrix)


def outcome_probs(prob_matrix):
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


def evaluate_system_brier(rho, decay_rate):
    total_brier = 0.0
    for match in historical_matches:
        home, away, actual = match["home"], match["away"], match["result"]
        lmbda_h, lmbda_a = calculate_expected_goals(home, away, decay_rate)
        prob_matrix = build_score_matrix(lmbda_h, lmbda_a, rho)
        p_home, p_away, p_draw = outcome_probs(prob_matrix)

        y_home = 1.0 if actual == "H" else 0.0
        y_draw = 1.0 if actual == "D" else 0.0
        y_away = 1.0 if actual == "A" else 0.0

        total_brier += (p_home - y_home) ** 2 + (p_draw - y_draw) ** 2 + (p_away - y_away) ** 2
    return total_brier / len(historical_matches)


def negative_log_likelihood(params):
    rho, decay_rate = params
    if not (0.0 <= rho <= 0.20) or not (0.0 <= decay_rate <= 0.40):
        return 1e6

    total_neg_ll = 0.0
    for match in historical_matches:
        home, away = match["home"], match["away"]
        h_g, a_g = match["h_goals"], match["a_goals"]
        lmbda_h, lmbda_a = calculate_expected_goals(home, away, decay_rate)
        prob_matrix = build_score_matrix(lmbda_h, lmbda_a, rho)
        match_prob = prob_matrix[min(h_g, 5), min(a_g, 5)]
        if match_prob <= 0:
            total_neg_ll += 100.0
        else:
            total_neg_ll -= math.log(match_prob)
    return total_neg_ll


# =============================================================
if __name__ == "__main__":
    if LOCAL_DEV:
        initial_rho = 0.08
        initial_decay = 0.15
        initial_brier = evaluate_system_brier(initial_rho, initial_decay)

        print(f"  Baseline: rho={initial_rho:.2f} | decay={initial_decay:.2f} | Brier={initial_brier:.4f}\n")

        mle_result = minimize(
            negative_log_likelihood,
            [initial_rho, initial_decay],
            method='SLSQP',
            bounds=((0.0, 0.20), (0.0, 0.40))
        )

        if mle_result.success:
            opt_rho, opt_decay = mle_result.x
            optimized_brier = evaluate_system_brier(opt_rho, opt_decay)

            print(f"=== MLE Calibration Summary (v1.4) ===")
            print(f"  rho:        {opt_rho:.4f}")
            print(f"  decay:      {opt_decay:.4f}")
            print(f"  Brier (pre):  {initial_brier:.4f}")
            print(f"  Brier (post): {optimized_brier:.4f}")
            print(f"  Delta:        {(optimized_brier - initial_brier):.4f}")
            print("-" * 40)

            home_team = "Brazil"
            away_team = "Norway"

            print(f"\nRunning Poisson Predictor v1.4 | {home_team} vs {away_team}")
            lmbda_h, lmbda_a = calculate_expected_goals(home_team, away_team, opt_decay)
            print(f"  lambda_Home: {lmbda_h:.4f}")
            print(f"  lambda_Away: {lmbda_a:.4f}\n")

            prob_matrix = build_score_matrix(lmbda_h, lmbda_a, opt_rho)
            h, a, d = outcome_probs(prob_matrix)

            print(f"=== {home_team} vs {away_team} Inference Summary ===")
            print(f"  Win Prob ({home_team}): {h * 100:.2f}%")
            print(f"  Win Prob ({away_team}): {a * 100:.2f}%")
            print(f"  Draw Prob:           {d * 100:.2f}%")
            print("-" * 40)

            max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
            print(f"  Most Likely Score: {max_idx[0]}-{max_idx[1]}")
            print(f"  Confidence:        {prob_matrix[max_idx] * 100:.2f}%")

        else:
            print("  Optimizer failed to converge.")
