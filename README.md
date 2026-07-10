# World Cup Poisson Predictor (v1.5.3)

Switzerland 0-0 draw dragged rho back to 0.0329. Added Argentina and Switzerland results to the training pool (14 total).

## What changed
- Tossed 2 more knockout matches into the training pool (14 total).
- Switched inference block to France vs Morocco.

## MLE Calibration Log
- rho: `0.0329`
- decay: `0.1568`
- brier: `0.4413 -> 0.4388`

## Blind Test (France vs Morocco)
- Win Prob (France): `64.14%` (90 mins)
- Win Prob (Morocco): `15.12%` (90 mins)
- Push to Extra Time: `20.74%` (90 mins draw)
- Most Likely 90-Min Score: `2-0 (11.01%)` (Nailed it！)

## Still broken
- Teams data dictionary is completely static. Hardcoded group stage stats don't update automatically.
