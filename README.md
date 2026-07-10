# World Cup Poisson Predictor (v1.5.4)

Added France 2-0 (nailed the exact score). Total pool at 15 matches. rho stayed flat at 0.0329 due to 2-0 DC math boundary.

## What changed
- Added France vs Morocco to historical_matches.
- Switched next match to Spain vs Belgium.

## MLE Calibration Log
- rho: `0.0329`
- decay: `0.1572`
- brier: `0.4253 -> 0.4225`

## Blind Test (Spain vs Belgium)
- Win Prob (Spain): `50.70%` (90 mins)
- Win Prob (Belgium): `25.92%` (90 mins)
- Push to Extra Time: `23.38%` (90 mins draw)
- Most Likely 90-Min Score: `1-1 (10.57%)`

## Still broken
- Teams data dictionary is completely static. Hardcoded group stage stats don't update automatically.
