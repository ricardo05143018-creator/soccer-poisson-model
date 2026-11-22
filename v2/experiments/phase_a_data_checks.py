"""
Soccer model v2 Phase A data checks.
Checks the corrected 104-match sequence and the 16 Round of 32 targets.
Date: November 2026
"""

from v2.src.load_data import load_matches, load_target_fixtures, frozen_v1_matches


GROUP_STAGE_END = 72


def result_from_score(h_goals, a_goals):
    if h_goals > a_goals:
        return "H"
    if h_goals < a_goals:
        return "A"
    return "D"


def expected_stage(match_no):
    if match_no <= 72:
        return "group"
    if match_no <= 88:
        return "round_of_32"
    if match_no <= 96:
        return "round_of_16"
    if match_no <= 100:
        return "quarterfinal"
    if match_no <= 102:
        return "semifinal"
    if match_no == 103:
        return "third_place"
    return "final"


def check_matches(matches):
    assert len(matches) == 104
    assert [m["match_no"] for m in matches] == list(range(1, 105))

    teams = sorted(set([m["home"] for m in matches] + [m["away"] for m in matches]))
    assert len(teams) == 48

    for m in matches:
        assert m["home"] != m["away"]
        assert m["h_goals"] >= 0 and m["a_goals"] >= 0
        assert m["result"] == result_from_score(m["h_goals"], m["a_goals"])
        assert m["stage"] == expected_stage(m["match_no"])

    # v1.6.5 omitted the opening match and stopped right before the final
    opener = matches[0]
    assert (opener["home"], opener["away"], opener["h_goals"], opener["a_goals"]) == ("Mexico", "South Africa", 2, 0)
    final = matches[-1]
    assert (final["home"], final["away"], final["h_goals"], final["a_goals"]) == ("Spain", "Argentina", 0, 0)

    frozen = frozen_v1_matches(matches)
    assert len(frozen) == 102
    assert [m["match_no"] for m in frozen] == list(range(2, 104))
    return teams


def check_targets(matches, fixtures):
    assert len(fixtures) == 16
    assert abs(sum(f["weight"] for f in fixtures) - 1.0) < 1e-12

    seen_teams = set([m["home"] for m in matches[:GROUP_STAGE_END]] + [m["away"] for m in matches[:GROUP_STAGE_END]])
    for f in fixtures:
        assert f["cutoff_match"] == GROUP_STAGE_END
        assert 73 <= f["target_match_no"] <= 88
        assert f["home"] in seen_teams and f["away"] in seen_teams
        actual = matches[f["target_match_no"] - 1]
        assert (f["home"], f["away"]) == (actual["home"], actual["away"])


if __name__ == "__main__":
    print("=== Phase A: Data Checks ===")

    matches = load_matches()
    fixtures = load_target_fixtures()
    teams = check_matches(matches)
    check_targets(matches, fixtures)

    print(f"  matches: {len(matches)} ({len(frozen_v1_matches(matches))} in frozen v1.6.5)")
    print(f"  teams:   {len(teams)}")
    print(f"  targets: {len(fixtures)} Round of 32 fixtures (cutoff match {GROUP_STAGE_END})")
    print("  all data checks passed.")
