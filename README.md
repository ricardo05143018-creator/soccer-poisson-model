# World Cup Poisson Predictor (v1.6.4)

The time-decay model from v1.6.3 successfully hit the 1-1 draw (90-min regular time) for Argentina vs England. 

For v1.6.4, I just added this latest 1-1 result into the historical dataset as a rolling update to run the inference for the third-place match (France vs England). No changes were made to the underlying math or decay logic.

## Calibration Log
- Iteration: `Converged in 32 rounds`
- Mathematical rho limit: `0.3751`
- Applied ceiling: `0.2000`
- Constrained rho: `0.2000` (hit the ceiling again)

Low-Score Diagnostics (Unweighted Baseline):
- 0-0: `11 actual / 10.42 expected`
- 1-0: `6 actual / 11.63 expected`
- 0-1: `7 actual / 10.45 expected`
- 1-1: `18 actual / 11.45 expected`

## Third-Place Match Prediction (France vs England)
- France win: `44.40%`
- England win: `23.16%`
- Draw: `32.44%`
- Top score: `1-1 (15.32%)`

## Known Gaps
- Still no out-of-sample backtesting pipeline.
