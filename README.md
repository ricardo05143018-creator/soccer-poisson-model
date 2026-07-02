# World Cup Poisson Predictor (v1.2)

Form decay overlay on top of v1.1. Also hotfixes the DC sign bug.

## What changed
- Added exponential form decay multiplier per team.
- Fixed DC tau signs: draws (0,0)(1,1) inflated, non-draws (1,0)(0,1) deflated.

## Output (Spain vs Austria)
- `65.24% 14.26% 20.50%` (Home Win, Away Win, Draw)
- Most likely score: `2-0` at `10.42%`

## Still broken
- DECAY_RATE and RHO both hardcoded, not fitted.
- Still no backtesting framework.
