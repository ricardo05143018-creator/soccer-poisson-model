# World Cup Poisson Predictor (v1.5)

Added an L2 regularization layer after V1.4 overfitted and slammed straight into the boundaries on this morning's upsets (Brazil 1-2, Mexico 2-3).

## What changed
- Added L2 penalty layer (lambda = 15.0) to anchor rho and decay.
- Expanded historical training data to 10 knockout matches to include today's results.
- Kept optimization strictly in 2D to prevent overfitting under small sample size.

## MLE Calibration Log
- rho: `0.0230` (pulled back from 0.0000 collapse via L2 anchor)
- decay: `0.1589` (escaped the 0.4000 boundary wall trapping)
- Brier (pre): `0.4207` (V1.4 baseline heavily punished by July 6 cold results)
- Brier (post): `0.4160`

## Still broken
- Just blindly trusting the SLSQP success flag without tracking bounds.
- No clue if the optimal valley is actually stable. Need a quick decay sweep.
