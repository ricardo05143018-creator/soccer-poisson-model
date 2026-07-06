# World Cup Poisson Predictor (v1.5.1)

V1.4 completely blew up on the Brazil and Mexico upsets. Slapped an L2 penalty on the loss function to drag parameters away from the walls, and added bounds/sweep diagnostics so the engine actually predicts upcoming matches instead of just idling.

## What changed
- Added L2 penalty (lambda = 15.0) to anchor rho and decay.
- Tossed today's 2 new knockout matches into the training pool (10 total).
- Kept optimization strictly in 2D.
- Added a quick bounds check for SLSQP limits.
- Added a decay sensitivity sweep.
- Brought back the inference loop for live upcoming predictions.

## MLE Calibration Log
- rho: `0.0230`
- decay: `0.1589`
- brier: `0.4207 -> 0.4160`

## Blind Test (Portugal vs Spain)
- Win Prob (Portugal): `40.50%` (90 mins)
- Win Prob (Spain): `34.97%` (90 mins)
- Push to Extra Time: `24.53%` (90 mins draw)
- Most Likely 90-Min Score: `1-1 (11.15%)`

## Still broken
- Teams data dictionary is completely static. Hardcoded group stage stats don't update automatically.
