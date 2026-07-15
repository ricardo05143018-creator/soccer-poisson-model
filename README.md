# World Cup Poisson Predictor (v1.6.3)

So v1.6.2 completely missed the France vs Spain match because it treated years-old defensive stats the exact same as recent form. I added the 0-2 result in a rolling update this morning, but the underlying engine still needed an actual fix.

This version finally adds exponential time-decay weighting. Recent matches matter a lot, old matches barely matter.

## Updates
- Added a time-decay weight to every match based on its index (half-life = 40 matches).
- Applied these weights to the global baseline, the iterative solver, and the final MLE loss function.
- Cleaned up some duplicate variables I left in the script.

## Calibration Log
- Iteration: `Converged in 32 rounds` (way faster now that old noise is damped out)
- Mathematical rho limit: `0.3724` (jumped up because recent matches are tighter)
- Applied ceiling: `0.2000`
- Constrained rho: `0.2000` (optimizer still hits the ceiling, 1-1 draws are just too common)

Low-Score Diagnostics (Unweighted Baseline):
- 0-0: `11 actual / 10.28 expected`
- 1-0: `6 actual / 11.46 expected`
- 0-1: `7 actual / 10.29 expected`
- 1-1: `17 actual / 11.28 expected`

## Match 2 Prediction (Argentina vs England)

With time-decay active, England's recent form gives them a much clearer edge than in v1.6.2.

- Argentina win: `29.45%`
- England win:   `41.33%`
- Draw:          `29.22%`
- Top score:     `1-1 (13.79%)` (nailed it!)

## What's still broken
- Still no proper out-of-sample backtesting pipeline.
