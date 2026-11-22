"""
Soccer model v2 data loader.
Reads the corrected 104-match 90-min dataset and fixed target fixtures.
Date: November 2026
"""

import csv
import os


V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_DATA_PATH = os.path.join(
    V2_DIR,
    "data",
    "processed",
    "world_cup_2026_90min_results.csv"
)
TARGET_DATA_PATH = os.path.join(
    V2_DIR,
    "data",
    "processed",
    "target_fixtures.csv"
)


def load_matches(path=MATCH_DATA_PATH):
    matches = []

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        # cast csv text once so later scripts stay simple
        for row in reader:
            matches.append({
                "match_no": int(row["match_no"]),
                "stage": row["stage"],
                "home": row["home"],
                "away": row["away"],
                "h_goals": int(row["h_goals"]),
                "a_goals": int(row["a_goals"]),
                "result": row["result"],
                "in_frozen_v1": int(row["in_frozen_v1"])
            })

    return matches


def load_target_fixtures(path=TARGET_DATA_PATH):
    fixtures = []

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            fixtures.append({
                "snapshot": row["snapshot"],
                "cutoff_match": int(row["cutoff_match"]),
                "target_match_no": int(row["target_match_no"]),
                "home": row["home"],
                "away": row["away"],
                "weight": float(row["weight"])
            })

    return fixtures


def frozen_v1_matches(matches):
    return [m for m in matches if m["in_frozen_v1"] == 1]


def matches_before(matches, cutoff_match):
    return [m for m in matches if m["match_no"] <= cutoff_match]
