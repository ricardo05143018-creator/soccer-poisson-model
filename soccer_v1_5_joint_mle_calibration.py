"""
Poisson soccer model with Dixon-Coles correction + MLE calibration (v1.5 clean).
Adds L2 regularization layer to anchor parameter drift.
Date: July 2026
"""

import math
import numpy as np
from scipy.optimize import minimize

TOURNAMENT_AVG_HOME_GOALS = 1.42
TOURNAMENT_AVG_AWAY_GOALS = 1.18

L2_LAMBDA = 15.0
RHO_ANCHOR = 0.06
DECAY_ANCHOR = 0.15

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
    "Mexico":       {"attack": 1.10, "defense": 0.98, "form": 1.00},
    "England":      {"attack": 1.40, "defense": 0.85, "form": 1.05}
}


# Regular time only. Ignore penalty shootouts.
historical_matches = [
    {"home": "Spain", "away": "Austria", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Portugal", "away": "Croatia", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Switzerland", "away": "Algeria", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Australia", "away": "Egypt", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Argentina", "away": "Cape Verde", "h_goals": 3, "a_goals": 2, "result": "H"},
    {"home": "Colombia", "away": "Ghana", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Canada", "away": "Morocco", "h_goals": 0, "a_goals": 3, "result": "A"},
    {"home": "Paraguay", "away": "France", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Brazil", "away": "Norway", "h_goals": 1, "a_goals": 2, "result": "A"},
    {"home": "Mexico", "away": "England", "h_goals": 2, "a_goals": 3, "result": "A"},
]


def apply_form_decay(team, decay_rate):
    form = teams_data[team]["form"]
    return 1.0 + decay_rate * (form - 1.0)


def expected_goals(home, away, decay_rate):
    home_form = apply_form_decay(home, decay_rate)
    away_form = apply_form_decay(away, decay_rate)

    home_xg = (
        teams_data[home]["attack"] *
        teams_data[away]["defense"] *
        home_form *
        TOURNAMENT_AVG_HOME_GOALS
    )

    away_xg = (
        teams_data[away]["attack"] *
        teams_data[home]["defense"] *
        away_form *
        TOURNAMENT_AVG_AWAY_GOALS
    )

    return home_xg, away_xg


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


def score_matrix(lh, la, rho, max_goals=6):
    mat = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            base = poisson_pmf(lh, i) * poisson_pmf(la, j)
            mat[i, j] = base * dixon_coles_tau(i, j, lh, la, rho)

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


def brier_score(mat, actual_home, actual_away):
    if actual_home > actual_away:
        y = [1, 0, 0]
    elif actual_home < actual_away:
        y = [0, 1, 0]
    else:
        y = [0, 0, 1]

    p = outcome_probs(mat)
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y))


def nll(params):
    rho, decay = params

    if not (0.0 <= rho <= 0.2 and 0.0 <= decay <= 0.4):
        return 1e6

    loss = 0.0

    for m in historical_matches:
        lh, la = expected_goals(m["home"], m["away"], decay)
        mat = score_matrix(lh, la, rho)

        i = min(m["h_goals"], 5)
        j = min(m["a_goals"], 5)

        prob = max(mat[i, j], 1e-12)
        loss -= math.log(prob)

    penalty = L2_LAMBDA * (
        (rho - RHO_ANCHOR) ** 2 +
        (decay - DECAY_ANCHOR) ** 2
    )

    return loss + penalty


# =============================================================
if __name__ == "__main__":

    if LOCAL_DEV:
        base_rho = 0.08
        base_decay = 0.15

        base_brier = 0.0

        for m in historical_matches:
            lh, la = expected_goals(m["home"], m["away"], base_decay)
            mat = score_matrix(lh, la, base_rho)
            base_brier += brier_score(mat, m["h_goals"], m["a_goals"])

        base_brier /= len(historical_matches)

        print("Baseline:")
        print(f"  rho={base_rho:.2f}, decay={base_decay:.2f}, brier={base_brier:.4f}\n")

        res = minimize(
            nll,
            [base_rho, base_decay],
            method="SLSQP",
            bounds=((0.0, 0.2), (0.0, 0.4))
        )

        if res.success:
            rho, decay = res.x

            opt_brier = 0.0
            for m in historical_matches:
                lh, la = expected_goals(m["home"], m["away"], decay)
                mat = score_matrix(lh, la, rho)
                opt_brier += brier_score(mat, m["h_goals"], m["a_goals"])

            opt_brier /= len(historical_matches)

            print("Calibration Result:")
            print(f"  rho:   {rho:.4f}")
            print(f"  decay: {decay:.4f}")
            print(f"  brier: {base_brier:.4f} -> {opt_brier:.4f}")