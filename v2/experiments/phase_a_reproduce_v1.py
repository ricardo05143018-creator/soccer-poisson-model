"""
Soccer model v2 Phase A frozen reproduction.
Reproduces the v1.6.5 final forecast from the V2 CSV data layer.
Date: November 2026
"""

from v2.src.baseline import fit_baseline, low_score_diagnostics, predict_match
from v2.src.load_data import frozen_v1_matches, load_matches


TOLERANCE = 1e-8

EXPECTED_GLOBAL_LAMBDA = 1.2810594876929695
EXPECTED_CONVERGED_ROUND = 35
EXPECTED_MAX_LAMBDA = 3.0042741489390568
EXPECTED_RHO = 0.2
EXPECTED_NLL = 116.13543444530382

EXPECTED_OBS_COUNTS = {"0-0": 11, "1-0": 6, "0-1": 7, "1-1": 18}
EXPECTED_EXP_COUNTS = {
    "0-0": 9.404247254289624,
    "1-0": 10.872244679472788,
    "0-1": 9.822670828419897,
    "1-1": 11.303634579192913
}

EXPECTED_LH = 1.3288522483152894
EXPECTED_LA = 0.5650968946908446
EXPECTED_PROBS = (0.5315855621255974, 0.1358376703167592, 0.33257676755764337)
EXPECTED_TOP_SCORE = (1, 0)
EXPECTED_TOP_SCORE_PROB = 0.17780806428415727


def check_reproduction(matches, model, obs_counts, exp_counts, pred):
    # use the exact 102 rows from v1.6.5 (matches 2 through 103)
    assert [m["match_no"] for m in matches] == list(range(2, 104))
    assert model["converged_round"] == EXPECTED_CONVERGED_ROUND
    assert model["optimizer_success"]

    assert abs(model["global_lambda"] - EXPECTED_GLOBAL_LAMBDA) <= TOLERANCE
    assert abs(model["max_lambda_found"] - EXPECTED_MAX_LAMBDA) <= TOLERANCE
    assert abs(model["rho"] - EXPECTED_RHO) <= TOLERANCE
    assert abs(model["nll"] - EXPECTED_NLL) <= TOLERANCE
    assert obs_counts == EXPECTED_OBS_COUNTS

    for score in EXPECTED_EXP_COUNTS:
        assert abs(exp_counts[score] - EXPECTED_EXP_COUNTS[score]) <= TOLERANCE

    assert abs(pred["lh"] - EXPECTED_LH) <= TOLERANCE
    assert abs(pred["la"] - EXPECTED_LA) <= TOLERANCE
    for actual, expected in zip(pred["probs"], EXPECTED_PROBS):
        assert abs(actual - expected) <= TOLERANCE

    assert tuple(pred["top_score"]) == EXPECTED_TOP_SCORE
    assert abs(pred["top_score_prob"] - EXPECTED_TOP_SCORE_PROB) <= TOLERANCE


if __name__ == "__main__":
    print("=== Phase A: Reproduce v1.6.5 ===")

    all_matches = load_matches()
    matches = frozen_v1_matches(all_matches)
    model = fit_baseline(matches)
    obs_counts, exp_counts = low_score_diagnostics(matches, model)
    pred = predict_match("Spain", "Argentina", model)

    check_reproduction(matches, model, obs_counts, exp_counts, pred)
    print(f"  converged in {model['converged_round']} rounds | global lambda: {model['global_lambda']:.6f}")

    print("\n--- Low-Score Diagnostics ---")
    print("  Scoreline | Actual | Poisson Expected")
    for score in ["0-0", "1-0", "0-1", "1-1"]:
        print(f"    {score}     |   {obs_counts[score]:2d}   |      {exp_counts[score]:.2f}")

    rho_status = "UPPER_BOUND" if abs(model["rho"] - model["safe_rho_upper"]) < 1e-4 else "INTERIOR"
    p_home, p_away, p_draw = pred["probs"]
    top_home, top_away = pred["top_score"]

    print(f"\n  optimal rho: {model['rho']:.4f} ({rho_status})")
    print(f"  Spain vs Argentina: Spain {p_home * 100:.2f}% | Argentina {p_away * 100:.2f}% | Draw {p_draw * 100:.2f}%")
    print(f"  top 90-min score: {top_home}-{top_away} ({pred['top_score_prob'] * 100:.2f}%)")
    print(f"  matched v1.6.5 within {TOLERANCE:.0e}.")
