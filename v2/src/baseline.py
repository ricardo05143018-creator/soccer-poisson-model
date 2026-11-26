"""
Soccer model v2 frozen baseline.
Rebuilds the v1.6.5 time-decayed Poisson and Dixon-Coles fit.
Date: November 2026
"""

import math
import numpy as np
from scipy.optimize import minimize

from v2.src.evaluation import exact_match_probability, outcome_probs, poisson_pmf, score_matrix


SMOOTHING_GAMES = 2.0
HALF_LIFE_MATCHES = 40.0
RELAXATION = 0.5
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-6
RHO_CEILING = 0.20


def add_time_weights(matches):
    weighted_matches = [dict(m) for m in matches]
    xi = math.log(2) / HALF_LIFE_MATCHES
    n_matches = len(weighted_matches)

    for i, m in enumerate(weighted_matches):
        time_diff = n_matches - 1 - i
        m["weight"] = math.exp(-xi * time_diff)

    return weighted_matches


def fit_strengths(matches):
    total_weighted_goals = sum((m["h_goals"] + m["a_goals"]) * m["weight"] for m in matches)
    total_weight = sum(m["weight"] for m in matches)
    global_lambda = total_weighted_goals / (2.0 * total_weight)

    team_names = sorted(list(set([m["home"] for m in matches] + [m["away"] for m in matches])))
    team_to_idx = {name: i for i, name in enumerate(team_names)}
    n_teams = len(team_names)

    alphas = np.ones(n_teams)
    betas = np.ones(n_teams)
    converged_round = 0
    last_delta = float("inf")

    # keep the v1 damping path unchanged
    for round_idx in range(MAX_ITERATIONS):
        new_alphas = np.zeros(n_teams)
        new_betas = np.zeros(n_teams)

        for i, team in enumerate(team_names):
            gf, ga, exp_gf_denom, exp_ga_denom = 0, 0, 0, 0

            for m in matches:
                w = m["weight"]

                if m["home"] == team:
                    opp_idx = team_to_idx[m["away"]]
                    gf += m["h_goals"] * w
                    ga += m["a_goals"] * w
                    exp_gf_denom += betas[opp_idx] * global_lambda * w
                    exp_ga_denom += alphas[opp_idx] * global_lambda * w
                elif m["away"] == team:
                    opp_idx = team_to_idx[m["home"]]
                    gf += m["a_goals"] * w
                    ga += m["h_goals"] * w
                    exp_gf_denom += betas[opp_idx] * global_lambda * w
                    exp_ga_denom += alphas[opp_idx] * global_lambda * w

            new_alphas[i] = (gf + SMOOTHING_GAMES * global_lambda) / (
                    exp_gf_denom + SMOOTHING_GAMES * global_lambda)
            new_betas[i] = (ga + SMOOTHING_GAMES * global_lambda) / (
                    exp_ga_denom + SMOOTHING_GAMES * global_lambda)

        new_alphas = new_alphas / np.mean(new_alphas)
        new_betas = new_betas / np.mean(new_betas)

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

    return {
        "global_lambda": global_lambda,
        "teams_data": teams_data,
        "converged_round": converged_round,
        "last_delta": last_delta
    }


def expected_goals(home_team, away_team, model):
    teams_data = model["teams_data"]
    global_lambda = model["global_lambda"]

    lh = global_lambda * teams_data[home_team]["attack"] * teams_data[away_team]["defense"]
    la = global_lambda * teams_data[away_team]["attack"] * teams_data[home_team]["defense"]

    return lh, la


def fit_rho(matches, model):
    max_lambda_found = 0.0

    for m in matches:
        lh, la = expected_goals(m["home"], m["away"], model)
        max_lambda_found = max(max_lambda_found, lh, la)

    # keep every low-score correction positive
    safe_rho_upper = min(RHO_CEILING, 0.999 / max_lambda_found)

    def nll_objective(params):
        rho = params[0]
        loss = 0.0

        for m in matches:
            lh, la = expected_goals(m["home"], m["away"], model)
            prob = exact_match_probability(m["h_goals"], m["a_goals"], lh, la, rho)
            loss -= m["weight"] * math.log(prob)

        return loss

    res = minimize(nll_objective, [0.05], method="SLSQP", bounds=[(0.0, safe_rho_upper)])

    return {
        "rho": res.x[0],
        "nll": res.fun,
        "optimizer_success": res.success,
        "max_lambda_found": max_lambda_found,
        "safe_rho_upper": safe_rho_upper
    }


def fit_baseline(matches):
    weighted_matches = add_time_weights(matches)
    model = fit_strengths(weighted_matches)
    model.update(fit_rho(weighted_matches, model))
    return model


def low_score_diagnostics(matches, model):
    obs_counts = {"0-0": 0, "1-0": 0, "0-1": 0, "1-1": 0}
    exp_counts = {"0-0": 0.0, "1-0": 0.0, "0-1": 0.0, "1-1": 0.0}

    for m in matches:
        score = f"{m['h_goals']}-{m['a_goals']}"
        if score in obs_counts:
            obs_counts[score] += 1

        lh, la = expected_goals(m["home"], m["away"], model)
        exp_counts["0-0"] += poisson_pmf(lh, 0) * poisson_pmf(la, 0)
        exp_counts["1-0"] += poisson_pmf(lh, 1) * poisson_pmf(la, 0)
        exp_counts["0-1"] += poisson_pmf(lh, 0) * poisson_pmf(la, 1)
        exp_counts["1-1"] += poisson_pmf(lh, 1) * poisson_pmf(la, 1)

    return obs_counts, exp_counts


def predict_match(home_team, away_team, model):
    lh, la = expected_goals(home_team, away_team, model)
    mat = score_matrix(lh, la, model["rho"])
    p_home, p_away, p_draw = outcome_probs(mat)
    max_idx = np.unravel_index(np.argmax(mat), mat.shape)

    return {
        "lh": lh,
        "la": la,
        "mat": mat,
        "probs": (p_home, p_away, p_draw),
        "top_score": (max_idx[0], max_idx[1]),
        "top_score_prob": mat[max_idx]
    }