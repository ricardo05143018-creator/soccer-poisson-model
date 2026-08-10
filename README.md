# Building a Poisson Soccer Model During the 2026 World Cup

This repository records a personal project I built and updated during the 2026 World Cup.

I started with a small independent-Poisson predictor for one match. As the tournament continued, I kept the earlier versions instead of replacing them: first to see what each assumption was doing, and later because the failed versions made the final one easier to understand.

The final model was frozen before Spain vs Argentina. Its pre-match script contains 102 completed-match records, with every score entered as a 90-minute result.

I did not build this as a betting model. I wanted to see what changed as I removed hand-set assumptions, fitted more of the model from data, and kept the versions that failed.

## Final Forecast: Spain vs Argentina

Before the match, v1.6.5 produced:

- Spain win in 90 minutes: **53.16%**
- Draw after 90 minutes: **33.26%**
- Argentina win in 90 minutes: **13.58%**
- Most likely 90-minute scoreline: **Spain 1-0** (**17.78%**)

The match was 0-0 after 90 minutes. Spain then scored in the 106th minute and won 1-0 after extra time.

So the regulation-time prediction was wrong. The model rated Spain above Argentina, and the final score eventually became 1-0, but extra time was outside the model's scope. I am keeping this as a miss rather than changing the evaluation rule after seeing the result.

## How the Model Changed

| Version | Change | Why it changed |
| --- | --- | --- |
| v1.0 | Independent Poisson baseline with manually entered team strengths | A simple starting point for Spain vs Austria |
| v1.1 | Added a Dixon-Coles-style adjustment for the four lowest score cells | The independent model could not represent low-score dependence |
| v1.2 | Changed the sign convention for the low-score adjustment and added a hand-tuned form multiplier | With `rho` constrained to be positive, v1.1 moved probability away from the low-score draws I intended to model |
| v1.3 | Added an outcome-level Brier-score loop | I wanted a numerical check instead of judging predictions one match at a time |
| v1.4.0 | Used constrained maximum likelihood to tune `rho` and the form-decay parameter | The fitted parameters immediately exposed a problem: both preferred a boundary |
| v1.5.0 | Added L2 regularization, bound checks, and a small decay sensitivity sweep | The regularized fit produced interior values instead of treating boundary solutions as automatically meaningful |
| v1.6.0 | Removed the manual team ratings and rebuilt the model around iteratively estimated attack and defense strengths | This was the main shift from a hand-tuned predictor to one fitted from tournament results |
| v1.6.1 | Added damping and a convergence condition to the iterative updates | The original updates could oscillate instead of settling |
| v1.6.3 | Added exponential time weighting with a 40-match half-life | Recent matches should affect estimated team strength more than early group-stage matches |
| v1.6.5 | Added the semifinal and third-place results, then froze the final forecast | Spain vs Argentina was predicted from the 102-match list in the frozen script |

The small versions between these milestones mostly add completed match results and change the next fixture to infer. They are retained to preserve the chronology of the project.

## Final Model

The final version estimates each team's attack and defense multipliers from completed regulation-time results.

It begins with a tournament-wide scoring baseline, then repeatedly updates each team's attack and defense from goals scored and conceded against the opponents it faced. A smoothing term keeps very small samples from dominating the estimate. The updates are damped and stopped when their largest change is below a convergence threshold.

Older matches receive less weight:

```text
weight = exp(-log(2) × matches_ago / 40)
```

The expected goals for a fixture are then built from the weighted baseline and the two fitted team strengths. A Dixon-Coles-style `rho` adjustment reallocates probability among `0-0`, `1-0`, `0-1`, and `1-1`.

For the final, the fitted `rho` reached the imposed ceiling of `0.20`. The mathematical safety cap was higher (`0.3325`), so this was not a case where the probabilities themselves forced the limit. It means the chosen modeling bound was still constraining the fit.

## A Useful Miss: France vs England

v1.6.4 forecast the third-place match as:

- France win: **44.40%**
- Draw: **32.44%**
- England win: **23.16%**
- Most likely scoreline: **1-1**

England won 5-3.

The result was far outside the model's low-score region. This version had no feature for match context: a third-place game was treated like any other fixture. I left the result in the dataset rather than treating it as an exception.

## Limitations

- The early Brier-score checks are in-sample checks, not a clean backtest.
- The final forecast is genuinely out of sample, but one match is not a validation framework.
- The model does not include extra time, penalties, lineups, injuries, or match-specific tactical context.
- Results are entered in the code as a chronological match list; there is no automated data-ingestion pipeline.
- Repeatedly hitting the `rho` ceiling suggests that the low-score correction needs a separate investigation.
- A post-tournament fixture check found that Mexico 2-0 South Africa was missing from the frozen 102-match list. Adding it changes Spain's win probability from 53.16% to 53.12%; v1.6.5 remains unchanged because it is the version that actually produced the forecast.

The first post-tournament test I want to run is a rolling evaluation: refit before each match using only earlier results, save every forecast, and score the full sequence at the end.

## Repository Guide

Start here:

- `soccer_v1_0_baseline_predictor.py` — original independent-Poisson baseline
- `soccer_v1_1_dixon_coles_correction.py` — first low-score adjustment
- `soccer_v1_2_feature_expansion.py` — revised `rho` convention and form overlay
- `soccer_v1_3_historical_backtest.py` — initial Brier-score evaluation
- `soccer_v1_4_0_mle_parameter_fitting.py` — first constrained MLE fit
- `soccer_v1_5_0_regularized_rho_decay_fit.py` — regularized calibration
- `soccer_v1_6_0_mle_optimization_diagnostics.py` — data-estimated attack and defense strengths
- `soccer_v1_6_1_mle_optimization_diagnostics.py` — damped convergence
- `soccer_v1_6_3_mle_optimization_diagnostics.py` — exponential time weighting
- `soccer_v1_6_5_mle_optimization_diagnostics.py` — frozen final forecast

## Running the Final Version

```bash
pip install numpy scipy
python soccer_v1_6_5_mle_optimization_diagnostics.py
```
