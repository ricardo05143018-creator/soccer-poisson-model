# V2 Data Notes

The match table in `data/processed/world_cup_2026_90min_results.csv` starts from the 102-match list embedded in `soccer_v1_6_5_mle_optimization_diagnostics.py`.

Because `v1.6.5` missed the opening game (`Mexico 2-0 South Africa`) and was frozen right before the final, V2 inserts the opener at `match_no = 1` and adds the final (`Spain 0-0 Argentina` after 90 minutes) at `match_no = 104`. Every row follows the same 90-minute rule used in V1, excluding extra time and penalties.

The `in_frozen_v1` column marks the exact 102 rows from `v1.6.5` (matches 2 through 103) so Phase A can reproduce the frozen forecast from the same file.

`data/processed/target_fixtures.csv` fixes the 16 Round of 32 matchups (matches 73 through 88) at the end of the group stage (`cutoff_match = 72`), each with equal weight `0.0625`.
