# World Cup Poisson Predictor (v1.6.1)

The crude 25-iteration solver from v1.6.0 was too basic. The parameters were actually stuck in a limit cycle oscillation (endlessly bouncing around) because of the low-sample teams in the dataset. 

Added a relaxation factor (damping) and a proper convergence threshold check. The iterative scaling solver now dynamically converges.

## How I fixed it
- Replaced the hardcoded 25-round loop with a proper convergence check (residual `delta < 1e-6`).
- Added a damping factor (`RELAXATION = 0.5`) to smooth out updates and stop the parameter oscillation. It now converges in 46 rounds.
- Wrote a dynamic check for the mathematical ceiling of Dixon-Coles rho (max limit is `0.2702` for this dataset).
- Kept the conservative boundary constraint at `0.2000` just to be safe.

## Calibration Log
- Iteration: `Converged in 46 rounds`
- Mathematical rho limit: `0.2702`
- Applied ceiling: `0.2000`
- Constrained rho: `0.2000` (optimizer hit the ceiling again, but the 1-1 empirical variance explains it)

Low-Score Diagnostics:
- 0-0: `11 actual / 10.54 expected`
- 1-0: `6 actual / 11.27 expected`
- 0-1: `7 actual / 9.64 expected`
- 1-1: `17 actual / 9.90 expected` (no wonder rho keeps hitting the ceiling)

## Match Inference (France vs Spain)
- France win: `32.33%`
- Spain win:  `22.29%`
- Draw:       `45.39%` (90 mins regular time)
- Top score:  `0-0 (31.22%)`

## What's still broken
- No match recency weighting or time-decay constants yet.
- Still no out-of-sample backtesting pipeline.
