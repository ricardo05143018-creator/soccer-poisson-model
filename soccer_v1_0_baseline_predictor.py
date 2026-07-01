"""
Naive independent Poisson match predictor.
Just a statistical baseline, assumes zero correlation between home and away goals (obviously unrealistic for knockouts).
Date: July 2026
"""

import math
import numpy as np

# tournament averages (group stage estimates)
AVG_HOME = 1.42
AVG_AWAY = 1.18

# team strength inputs (fixed for replication)
teams = {
    "Spain": {"att": 1.55, "def": 0.75},
    "Austria": {"att": 1.10, "def": 1.02}
}


def expected_goals(home, away):
    h = teams[home]["att"] * teams[away]["def"] * AVG_HOME
    a = teams[away]["att"] * teams[home]["def"] * AVG_AWAY
    return h, a


def poisson(lmbda, k):
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


if __name__ == "__main__":

    home, away = "Spain", "Austria"

    lh, la = expected_goals(home, away)

    max_g = 6
    P = np.zeros((max_g, max_g))

    # joint independence assumption
    for i in range(max_g):
        for j in range(max_g):
            P[i, j] = poisson(lh, i) * poisson(la, j)

    home_win = 0.0
    away_win = 0.0
    draw = 0.0

    for i in range(max_g):
        for j in range(max_g):
            if i > j:
                home_win += P[i, j]
            elif i < j:
                away_win += P[i, j]
            else:
                draw += P[i, j]

    print(round(home_win, 4), round(away_win, 4), round(draw, 4))

    idx = np.unravel_index(np.argmax(P), P.shape)
    raw_idx = (int(idx[0]), int(idx[1]))
    print(raw_idx, round(P[idx], 4))