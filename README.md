# World Cup Poisson Predictor (v1.0)

Built this before the Round of 16. Just a baseline, nothing fancy.

## What it does
- Ratings handcoded from group stage tables.
- Standard multiplicative lambda computation.
- Joint independence assumption: P(home, away) = P(home) * P(away).

## Output (Spain vs Austria)
- `0.6339 0.1499 0.1886` (Home Win, Away Win, Draw)
- Most likely score: `(2, 0)` at `0.1008`

## Known issues
- Independence assumption is garbage for knockout football. Completely underestimates low-scoring deadlocks.
- All parameters are manually assigned, purely heuristic.
- No backtesting at all.

Let's see how hard reality slaps this.