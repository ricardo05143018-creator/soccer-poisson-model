# World Cup Poisson Predictor (v1.5.2)

L2 anchor got wiped out by the USA 1-4 blowout, dropping rho back to 0.0000. Added USA and Belgium to fix KeyErrors.

## What changed
- Added USA and Belgium to teams_data
- Tossed 2 more knockout matches into the training pool (12 total)

## MLE Calibration Log
- rho: `0.0000` (trapped on the wall again)
- decay: `0.1576`
- brier: `0.4337 -> 0.4263`

## Blind Test (Argentina vs Egypt)
- Win Prob (Argentina): `68.31%` (90 mins)
- Win Prob (Egypt): `13.07%` (90 mins)
- Push to Extra Time: `18.62%` (90 mins draw)
- Most Likely 90-Min Score: `2-0 (11.58%)`

## Still broken
- Teams data dictionary is completely static. Hardcoded group stage stats don't update automatically.
