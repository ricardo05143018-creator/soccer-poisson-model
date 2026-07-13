# World Cup Poisson Predictor (v1.6.0)

The old v1.5.x versions were basically broken. I didn't realize extra-time and penalty shootouts were contaminating the 90-minute Poisson baseline. The data definition was wrong, so I ended up rebuilding the whole thing.

This version uses a clean dataset with only 90-minute regular-time results (99 matches total).

## How I fixed it

* Manually converted every knockout match back to its pure 90-minute score (took forever).
* Dropped the hardcoded team prior ratings. Too much guesswork.
* Wrote a simple 25-iteration loop to estimate attack/defense ratings based on opponent strength (basic SOS adjustment).
* Hard-capped Dixon-Coles `rho` so the model stops spitting out negative probabilities.
* Added a quick low-score diagnostic table to see why the optimization keeps getting stuck.

## Calibration Log

* `rho` safety ceiling: `0.2000`
* Constrained `rho`: `0.2000` (optimizer hit the ceiling again...)

**Low-Score Diagnostics:**

* 0-0: `11 actual / 10.54 expected`
* 1-0: `6 actual / 11.27 expected`
* 0-1: `7 actual / 9.64 expected`
* 1-1: `17 actual / 9.90 expected` (no wonder `rho` keeps hitting the ceiling)

## Match Inference (France vs Spain)

* France win: `32.33%`
* Spain win: `22.29%`
* Draw: `45.39%` (90 mins regular time)
* Top score: `0-0 (31.22%)`

## What's still broken

* The 25-iteration solver is pretty crude and still hardcoded. I'll replace it with a proper convergence check later.
* No match recency weighting or time-decay constants yet.
* Still no out-of-sample backtesting pipeline.
