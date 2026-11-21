# Soccer Model V2

V2 extends the frozen V1 tournament scripts (`v1.0` through `v1.6.5` in the repository root) in two stages:

1. Reconstructing the complete 104-match 90-minute dataset to audit V1 with a chronological rolling out-of-sample backtest.
2. Adding team-strength uncertainty after the group stage (`cutoff_match = 72`) to evaluate which extra unplayed match would be most informative before the Round of 32.

## Data Reconstruction

All match results follow V1's 90-minute regulation-time rule (ignoring extra time and penalty shootouts). See [`data/sources.md`](data/sources.md) for reconstruction notes.

- [`data/processed/world_cup_2026_90min_results.csv`](data/processed/world_cup_2026_90min_results.csv): Complete 104-match chronological table. Row 1 restores the opening match (`Mexico 2-0 South Africa`) omitted from `v1.6.5`, rows 2–103 preserve the exact 102 frozen V1 matches (`in_frozen_v1 = 1`), and row 104 adds the final (`Spain 0-0 Argentina` after 90 minutes).
- [`data/processed/target_fixtures.csv`](data/processed/target_fixtures.csv): Fixes the 16 Round of 32 matchups (matches 73–88) at the group-stage boundary (`cutoff_match = 72`) with equal weights (`1/16 = 0.0625`).