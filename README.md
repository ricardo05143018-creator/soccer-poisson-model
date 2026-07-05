# World Cup Poisson Predictor (v1.4)

Dumped manual tuning after V1.3's 1-1 France vs Paraguay prediction got cleanly slapped by Mbappe's 0-1 penalty reality check. Swapped in automated MLE calibration over recent knockout data.

## What changed
- Replaced human parameter guessing with automated optimization using scipy SLSQP minimize.
- Training set: 8 completed historical knockout matches (Jul 3-5).

## MLE Calibration Log
- rho: `0.0000` (DC correction collapsed — 8 samples not enough to stabilize)
- decay: `0.4000` (hit the hard upper bound, heavy overfitting)
- Brier (pre): `0.3599`
- Brier (post): `0.3463`
- Delta: `-0.0136`

## Blind Test (Brazil vs Norway)
- `50.29% 26.80% 22.91%` (Home Win, Away Win, Draw)
- Most likely score: `1-1` at `10.36%`

## Still broken
- 8 matches is nowhere near enough data for stable joint likelihood estimation.
- Team baseline attack/defense parameters are still arbitrary fixed constants from group tables.
- Need regularization (L2 or Dirichlet priors) on rho to stop it from flattening to absolute zero.
