# World Cup Poisson Predictor (v1.1)

DC correction patch on top of v1.0. The independence assumption was clearly garbage for low-scoring games.

## What changed
- Dixon-Coles tau correction for (0,0), (1,0), (0,1), (1,1) cells.
- Matrix renormalized after correction.

## Output (Spain vs Austria)
- `65.91% 16.14% 17.95%` (Home Win, Away Win, Draw)
- Most likely score: `2-0` at `10.37%`

## Still broken
- Parameters still purely heuristic.
- No backtesting framework yet.
- Rho hardcoded at 0.08, not fitted from data.
