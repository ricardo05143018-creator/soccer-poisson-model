# World Cup Poisson Predictor (v1.5.5)

Added Spain 2-1 (hit direction). Total pool at 16 matches. rho stayed flat at 0.0329 due to 2-1 DC math boundary.

## What changed
- Added Spain vs Belgium to historical_matches.
- Switched next match to Norway vs England and Argentina vs Switzerland.

## MLE Calibration Log
- rho: `0.0329`
- decay: `0.1572`
- brier: `0.4220 -> 0.4189`

## Blind Test (Norway vs England)
- Win Prob (Norway): `41.38%` (90 mins)
- Win Prob (England): `33.93%` (90 mins)
- Push to Extra Time: `24.69%` (90 mins draw)
- Most Likely 90-Min Score: `1-1 (11.23%)`

## Blind Test (Argentina vs Switzerland)
- Win Prob (Argentina): `57.97%` (90 mins)
- Win Prob (Switzerland): `20.01%` (90 mins)
- Push to Extra Time: `22.03%` (90 mins draw)
- Most Likely 90-Min Score: `1-1 (10.14%)`

## Still broken
- Teams data dictionary is completely static. Hardcoded group stage stats don't update automatically.
