"""
Poisson soccer model v1.6.1
Strict 90-min data. Added relaxation to iterative scaling to fix limit cycles.
Date: July 2026
"""

import math
import numpy as np
from scipy.optimize import minimize

SMOOTHING_GAMES = 2.0  # smooth out low-sample noise

historical_matches = [
    {"home": "South Korea", "away": "Czech Republic", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Canada", "away": "Bosnia", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "USA", "away": "Paraguay", "h_goals": 3, "a_goals": 1, "result": "H"},
    {"home": "Qatar", "away": "Switzerland", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Brazil", "away": "Morocco", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Haiti", "away": "Scotland", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Australia", "away": "Turkey", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Germany", "away": "Curacao", "h_goals": 7, "a_goals": 1, "result": "H"},
    {"home": "Netherlands", "away": "Japan", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "Ivory Coast", "away": "Ecuador", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Sweden", "away": "Tunisia", "h_goals": 4, "a_goals": 1, "result": "H"},
    {"home": "Spain", "away": "Cape Verde", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Belgium", "away": "Egypt", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Saudi Arabia", "away": "Uruguay", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Iran", "away": "New Zealand", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "France", "away": "Senegal", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Iraq", "away": "Norway", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Argentina", "away": "Algeria", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Austria", "away": "Jordan", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Portugal", "away": "Congo", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "England", "away": "Croatia", "h_goals": 4, "a_goals": 2, "result": "H"},
    {"home": "Ghana", "away": "Panama", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Uzbekistan", "away": "Colombia", "h_goals": 1, "a_goals": 2, "result": "A"},
    {"home": "Czech Republic", "away": "South Africa", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Switzerland", "away": "Bosnia", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Canada", "away": "Qatar", "h_goals": 5, "a_goals": 0, "result": "H"},
    {"home": "Mexico", "away": "South Korea", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "USA", "away": "Australia", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Scotland", "away": "Morocco", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Brazil", "away": "Haiti", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Turkey", "away": "Paraguay", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Netherlands", "away": "Sweden", "h_goals": 5, "a_goals": 1, "result": "H"},
    {"home": "Germany", "away": "Ivory Coast", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Ecuador", "away": "Curacao", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Tunisia", "away": "Japan", "h_goals": 0, "a_goals": 4, "result": "A"},
    {"home": "Spain", "away": "Saudi Arabia", "h_goals": 4, "a_goals": 0, "result": "H"},
    {"home": "Belgium", "away": "Iran", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Uruguay", "away": "Cape Verde", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "New Zealand", "away": "Egypt", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Argentina", "away": "Austria", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "France", "away": "Iraq", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Norway", "away": "Senegal", "h_goals": 3, "a_goals": 1, "result": "H"},
    {"home": "Jordan", "away": "Algeria", "h_goals": 1, "a_goals": 2, "result": "A"},
    {"home": "Portugal", "away": "Uzbekistan", "h_goals": 5, "a_goals": 0, "result": "H"},
    {"home": "England", "away": "Ghana", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Panama", "away": "Croatia", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Colombia", "away": "Congo", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Switzerland", "away": "Canada", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Bosnia", "away": "Qatar", "h_goals": 3, "a_goals": 1, "result": "H"},
    {"home": "Scotland", "away": "Brazil", "h_goals": 0, "a_goals": 3, "result": "A"},
    {"home": "Morocco", "away": "Haiti", "h_goals": 4, "a_goals": 2, "result": "H"},
    {"home": "Czech Republic", "away": "Mexico", "h_goals": 0, "a_goals": 2, "result": "A"},
    {"home": "South Africa", "away": "South Korea", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Ecuador", "away": "Germany", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Curacao", "away": "Ivory Coast", "h_goals": 0, "a_goals": 2, "result": "A"},
    {"home": "Tunisia", "away": "Netherlands", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Japan", "away": "Sweden", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Turkey", "away": "USA", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "Paraguay", "away": "Australia", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Norway", "away": "France", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Senegal", "away": "Iraq", "h_goals": 5, "a_goals": 0, "result": "H"},
    {"home": "Uruguay", "away": "Spain", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Cape Verde", "away": "Saudi Arabia", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "New Zealand", "away": "Belgium", "h_goals": 1, "a_goals": 4, "result": "A"},
    {"home": "Egypt", "away": "Iran", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Panama", "away": "England", "h_goals": 0, "a_goals": 2, "result": "A"},
    {"home": "Croatia", "away": "Ghana", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Colombia", "away": "Portugal", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Congo", "away": "Uzbekistan", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Jordan", "away": "Argentina", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Algeria", "away": "Austria", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "South Africa", "away": "Canada", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "Brazil", "away": "Japan", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Germany", "away": "Paraguay", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Netherlands", "away": "Morocco", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Ivory Coast", "away": "Norway", "h_goals": 1, "a_goals": 2, "result": "A"},
    {"home": "France", "away": "Sweden", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Mexico", "away": "Ecuador", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "England", "away": "Congo", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Belgium", "away": "Senegal", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "USA", "away": "Bosnia", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Spain", "away": "Austria", "h_goals": 3, "a_goals": 0, "result": "H"},
    {"home": "Portugal", "away": "Croatia", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Switzerland", "away": "Algeria", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Australia", "away": "Egypt", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Argentina", "away": "Cape Verde", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Colombia", "away": "Ghana", "h_goals": 1, "a_goals": 0, "result": "H"},
    {"home": "Canada", "away": "Morocco", "h_goals": 0, "a_goals": 2, "result": "A"},
    {"home": "Paraguay", "away": "France", "h_goals": 0, "a_goals": 1, "result": "A"},
    {"home": "Brazil", "away": "Norway", "h_goals": 0, "a_goals": 2, "result": "A"},
    {"home": "Mexico", "away": "England", "h_goals": 2, "a_goals": 3, "result": "A"},
    {"home": "Portugal", "away": "Spain", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "USA", "away": "Belgium", "h_goals": 1, "a_goals": 3, "result": "A"},
    {"home": "Argentina", "away": "Egypt", "h_goals": 2, "a_goals": 2, "result": "D"},
    {"home": "Switzerland", "away": "Colombia", "h_goals": 0, "a_goals": 0, "result": "D"},
    {"home": "France", "away": "Morocco", "h_goals": 2, "a_goals": 0, "result": "H"},
    {"home": "Spain", "away": "Belgium", "h_goals": 2, "a_goals": 1, "result": "H"},
    {"home": "Norway", "away": "England", "h_goals": 1, "a_goals": 1, "result": "D"},
    {"home": "Argentina", "away": "Switzerland", "h_goals": 1, "a_goals": 1, "result": "D"}
]

# global baseline expected goals
total_goals = sum(m["h_goals"] + m["a_goals"] for m in historical_matches)
GLOBAL_BASELINE_LAMBDA = total_goals / (2.0 * len(historical_matches))

team_names = sorted(list(set([m["home"] for m in historical_matches] + [m["away"] for m in historical_matches])))
team_to_idx = {name: i for i, name in enumerate(team_names)}
N_TEAMS = len(team_names)

# iterative scaling with relaxation to fix limit cycle oscillation
alphas = np.ones(N_TEAMS)
betas = np.ones(N_TEAMS)

RELAXATION = 0.5
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-6

converged_round = 0
last_delta = float("inf")

for round_idx in range(MAX_ITERATIONS):
    new_alphas = np.zeros(N_TEAMS)
    new_betas = np.zeros(N_TEAMS)
    for i, team in enumerate(team_names):
        gf, ga, exp_gf_denom, exp_ga_denom = 0, 0, 0, 0
        for m in historical_matches:
            if m["home"] == team:
                opp_idx = team_to_idx[m["away"]]
                gf += m["h_goals"]
                ga += m["a_goals"]
                exp_gf_denom += betas[opp_idx] * GLOBAL_BASELINE_LAMBDA
                exp_ga_denom += alphas[opp_idx] * GLOBAL_BASELINE_LAMBDA
            elif m["away"] == team:
                opp_idx = team_to_idx[m["home"]]
                gf += m["a_goals"]
                ga += m["h_goals"]
                exp_gf_denom += betas[opp_idx] * GLOBAL_BASELINE_LAMBDA
                exp_ga_denom += alphas[opp_idx] * GLOBAL_BASELINE_LAMBDA

        new_alphas[i] = (gf + SMOOTHING_GAMES * GLOBAL_BASELINE_LAMBDA) / (
                exp_gf_denom + SMOOTHING_GAMES * GLOBAL_BASELINE_LAMBDA)
        new_betas[i] = (ga + SMOOTHING_GAMES * GLOBAL_BASELINE_LAMBDA) / (
                exp_ga_denom + SMOOTHING_GAMES * GLOBAL_BASELINE_LAMBDA)

    new_alphas = new_alphas / np.mean(new_alphas)
    new_betas = new_betas / np.mean(new_betas)

    # apply damping to prevent endless oscillation
    updated_alphas = (RELAXATION * new_alphas + (1.0 - RELAXATION) * alphas)
    updated_betas = (RELAXATION * new_betas + (1.0 - RELAXATION) * betas)

    max_change = max(
        np.max(np.abs(updated_alphas - alphas)),
        np.max(np.abs(updated_betas - betas))
    )

    alphas = updated_alphas
    betas = updated_betas
    last_delta = max_change

    if max_change < CONVERGENCE_THRESHOLD:
        converged_round = round_idx + 1
        break

teams_data = {name: {"attack": alphas[i], "defense": betas[i]} for i, name in enumerate(team_names)}

# dynamic cap for rho so probabilities don't go negative
max_lambda_found = 0.0
for m in historical_matches:
    lh = GLOBAL_BASELINE_LAMBDA * teams_data[m["home"]]["attack"] * teams_data[m["away"]]["defense"]
    la = GLOBAL_BASELINE_LAMBDA * teams_data[m["away"]]["attack"] * teams_data[m["home"]]["defense"]
    max_lambda_found = max(max_lambda_found, lh, la)

SAFE_RHO_UPPER = min(0.20, 0.999 / max_lambda_found)

def poisson_pmf(lmbda, k):
    if lmbda <= 0: return 1.0 if k == 0 else 0.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)

def dixon_coles_tau(i, j, lh, la, rho):
    if (i, j) == (0, 0): return 1.0 + lh * la * rho
    if (i, j) == (1, 0): return 1.0 - la * rho
    if (i, j) == (0, 1): return 1.0 - lh * rho
    if (i, j) == (1, 1): return 1.0 + rho
    return 1.0

def exact_match_probability(h_goals, a_goals, lh, la, rho):
    base_prob = poisson_pmf(lh, h_goals) * poisson_pmf(la, a_goals)
    tau = dixon_coles_tau(h_goals, a_goals, lh, la, rho)
    return max(base_prob * tau, 1e-12)

def nll_objective(params):
    rho = params[0]
    loss = 0.0
    for m in historical_matches:
        lh = GLOBAL_BASELINE_LAMBDA * teams_data[m["home"]]["attack"] * teams_data[m["away"]]["defense"]
        la = GLOBAL_BASELINE_LAMBDA * teams_data[m["away"]]["attack"] * teams_data[m["home"]]["defense"]
        prob = exact_match_probability(m["h_goals"], m["a_goals"], lh, la, rho)
        loss -= math.log(prob)
    return loss

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

if __name__ == "__main__":
    print(f"[BOOT] Loaded v1.6.1 | Pure 90-min dataset + Damped Iteration")
    if converged_round > 0:
        print(f"[BOOT] Strength iteration converged in {converged_round} rounds.")
    else:
        print(f"[BOOT] Strength iteration hit max {MAX_ITERATIONS} rounds (delta={last_delta:.2e}).")

    # empirical vs independent poisson expected counts
    obs_counts = {"0-0": 0, "1-0": 0, "0-1": 0, "1-1": 0}
    exp_counts = {"0-0": 0.0, "1-0": 0.0, "0-1": 0.0, "1-1": 0.0}
    for m in historical_matches:
        s_str = f"{m['h_goals']}-{m['a_goals']}"
        if s_str in obs_counts: obs_counts[s_str] += 1
        lh = GLOBAL_BASELINE_LAMBDA * teams_data[m["home"]]["attack"] * teams_data[m["away"]]["defense"]
        la = GLOBAL_BASELINE_LAMBDA * teams_data[m["away"]]["attack"] * teams_data[m["home"]]["defense"]
        exp_counts["0-0"] += poisson_pmf(lh, 0) * poisson_pmf(la, 0)
        exp_counts["1-0"] += poisson_pmf(lh, 1) * poisson_pmf(la, 0)
        exp_counts["0-1"] += poisson_pmf(lh, 0) * poisson_pmf(la, 1)
        exp_counts["1-1"] += poisson_pmf(lh, 1) * poisson_pmf(la, 1)

    print("\n--- LOW-SCORE DIAGNOSTICS ---")
    print(f"  Scoreline | Actual Count | Poisson Expected")
    for tag in ["0-0", "1-0", "0-1", "1-1"]:
        print(f"    {tag}     |      {obs_counts[tag]:2d}      |       {exp_counts[tag]:.2f}")

    # run MLE solver
    res = minimize(nll_objective, [0.05], method="SLSQP", bounds=[(0.0, SAFE_RHO_UPPER)])

    if res.success:
        opt_rho = res.x[0]
        math_cap = 0.999 / max_lambda_found

        print("\n[Calibration Results]")
        print(f"  math cap: {math_cap:.4f}")
        print(f"  applied ceiling: 0.2000")
        print(f"  optimal rho: {opt_rho:.4f} " + (
            "(UPPER_BOUND)" if abs(opt_rho - SAFE_RHO_UPPER) < 1e-4 else "(INTERIOR)"))

        # run inference for FRA vs ESP
        lh = GLOBAL_BASELINE_LAMBDA * teams_data["France"]["attack"] * teams_data["Spain"]["defense"]
        la = GLOBAL_BASELINE_LAMBDA * teams_data["Spain"]["attack"] * teams_data["France"]["defense"]

        mat = score_matrix(lh, la, opt_rho)
        p_home, p_away, p_draw = outcome_probs(mat)
        max_idx = np.unravel_index(np.argmax(mat), mat.shape)

        print(f"\n--- INFERENCE: France vs Spain ---")
        print(f"  probabilities: France {p_home * 100:.2f}% | Spain {p_away * 100:.2f}% | Draw {p_draw * 100:.2f}%")
        print(f"  top 90-min score: {max_idx[0]}-{max_idx[1]} ({mat[max_idx] * 100:.2f}%)")