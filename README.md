# World Cup Poisson Predictor (v1.6.5)

The model missed the France–England third-place match badly (3–5). I have kept that result in the data rather than treating it as an exception, but it is a useful reminder that this version is not designed to handle unusually open, high-variance matches well.

The inference has now been updated for the final, Spain vs Argentina. I did not change the time-decay settings for this update.

## Calibration Log

- Iteration: `Converged in 35 rounds`
- Mathematical rho limit: `0.3325`
- Applied ceiling: `0.2000`
- Constrained rho: `0.2000` (hit the ceiling again)

### Low-Score Diagnostics (Unweighted Baseline)

- 0-0: `11 actual / 9.40 expected`
- 1-0: `6 actual / 10.87 expected`
- 0-1: `7 actual / 9.82 expected`
- 1-1: `18 actual / 11.30 expected`

## Final Match Prediction: Spain vs Argentina

- Spain win: `53.16%`
- Argentina win: `13.58%`
- Draw: `33.26%`
- Top score: `1-0 (17.78%)`

Argentina's estimated win probability is lower than I expected. The current weighting scheme places substantial emphasis on recent scoring form, which favors Spain after Argentina's recent low-scoring draws. Whether that weighting is too aggressive is something I should test rather than assume away.

## Known Gaps

- No out-of-sample backtesting pipeline yet.
- Performance is likely weak in unusually open, high-variance matches.
- The rho ceiling is binding again and needs separate investigation.
