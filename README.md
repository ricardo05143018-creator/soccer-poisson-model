# World Cup Poisson Predictor (v1.3)

Added a basic evaluation loop over completed knockout matches to check Brier score miscalibration.

## What changed
- Batch evaluation loop over 12 completed historical matches.
- Multi-class Brier score tracking per match + average.

## Output Matrix Log
- Avg Brier Score: `0.4193`
- Worst fit: Germany vs Paraguay `0.9809` (Model expected a home win, heavily punished by the 1-1 deadlock)

## Blind Test (Paraguay vs France)
- `19.55% 56.54% 23.91%` (Home Win, Away Win, Draw)
- Most likely score: `1-1` at `11.28%`

## Still broken
- DECAY_RATE and RHO are still static empirical constants, not fitted.
- Team baseline ratings are still manually assigned from group tables.
