# World Cup Poisson Predictor (v1.6.2)

Since the math and convergence logic are finally stable in v1.6.1, this is just a quick refactor to run multiple matches at once instead of hardcoding a single fixture.

Added the Argentina vs England semi-final alongside France vs Spain.

## Updates
- Refactored the single-match inference block into a simple loop.
- Calibration logs and mathematical boundaries are identical to v1.6.1.

## Calibration Log
- Iteration: `Converged in 46 rounds`
- Mathematical rho limit: `0.2702`
- Applied ceiling: `0.2000`
- Constrained rho: `0.2000` (optimizer hit the ceiling again, but the 1-1 empirical variance explains it)

Low-Score Diagnostics:
- 0-0: `11 actual / 10.54 expected`
- 1-0: `6 actual / 11.27 expected`
- 0-1: `7 actual / 9.64 expected`
- 1-1: `17 actual / 9.90 expected`

## Semi-Final Predictions (90-min regular time)

**Match 1: France vs Spain**
- France win: `32.33%`
- Spain win:  `22.29%`
- Draw:       `45.39%`
- Top score:  `0-0 (31.22%)`

**Match 2: Argentina vs England**
- Argentina win: `31.01%`
- England win:   `39.64%`
- Draw:          `29.35%`
- Top score:     `1-1 (13.84%)`

## What's still broken
- No match recency weighting or time-decay constants yet.
- Still no out-of-sample backtesting pipeline.

## Rolling Update (July 15)

The v1.6.2 model completely missed the France vs Spain match (actual score 0-2). 

Like I mentioned in the known gaps yesterday, not having time-decay weighting is a big problem. The model treated France's old defensive stats exactly the same as their current form, so Spain was just undervalued.

I added this 0-2 result back into the dataset (now 100 matches total) and re-ran the solver to see how it affects the Argentina vs England match.

### What changed after adding the match
- The win probabilities for Argentina and England actually shifted. Since the solver estimates attack/defense based on opponent strength, Spain winning 2-0 rippled through the rest of the teams.
- The iteration loop converged in 45 rounds instead of 46.
- Math cap for rho shifted slightly to `0.2712`.

### Updated Inference (Argentina vs England)
- Argentina win: `31.70%` (+0.69%)
- England win:   `38.89%` (-0.75%)
- Draw:          `29.41%`
- Top score:     `1-1 (13.86%)`

I definitely need to figure out time-decay weighting for v1.6.3.
