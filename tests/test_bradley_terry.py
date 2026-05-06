"""Unit tests for W7.3 K.5 Bradley-Terry MLE."""

from __future__ import annotations

import math

from src.cdss.reasoning.bradley_terry import (
    aggregate_judge_votes,
    bt_mle,
    tournament_rank,
    win_probability,
)


def test_two_player_dominant_wins():
    # A beats B 9 of 10 times → s_A >> s_B.
    wins = {("A", "B"): 9, ("B", "A"): 1}
    s = bt_mle(wins, iters=100)
    assert s["A"] == 1.0
    assert s["A"] > s["B"] * 5
    # Closed-form check: s_A / s_B should ≈ 9/1 = 9.
    ratio = s["A"] / max(1e-9, s["B"])
    assert 5.0 < ratio < 15.0


def test_two_player_balanced_strengths_equal():
    wins = {("A", "B"): 5, ("B", "A"): 5}
    s = bt_mle(wins, iters=100)
    assert abs(s["A"] - s["B"]) < 1e-3


def test_three_player_consistent_ranking():
    # A > B > C, transitive
    wins = {
        ("A", "B"): 7, ("B", "A"): 3,
        ("B", "C"): 7, ("C", "B"): 3,
        ("A", "C"): 9, ("C", "A"): 1,
    }
    s = bt_mle(wins, iters=200)
    assert s["A"] > s["B"] > s["C"]


def test_tournament_rank_descending():
    s = {"A": 0.3, "B": 1.0, "C": 0.5}
    ranked = tournament_rank(s)
    assert [k for k, _ in ranked] == ["B", "C", "A"]


def test_win_probability_symmetry():
    p_ij = win_probability(2.0, 1.0)
    p_ji = win_probability(1.0, 2.0)
    assert abs(p_ij + p_ji - 1.0) < 1e-9
    assert abs(p_ij - 2.0 / 3.0) < 1e-9


def test_aggregate_votes_sums():
    judges = [
        {("A", "B"): 1, ("B", "A"): 0},
        {("A", "B"): 1, ("B", "A"): 0},
        {("A", "B"): 0, ("B", "A"): 1},
    ]
    out = aggregate_judge_votes(judges)
    assert out[("A", "B")] == 2.0
    assert out[("B", "A")] == 1.0


def test_empty_input_safe():
    assert bt_mle({}) == {}


def test_single_player_input():
    s = bt_mle({("X", "X"): 0})
    # only one participant — degenerate, returns {X: 1.0}
    assert s == {"X": 1.0}


def test_bt_floor_protects_zero_wins():
    # C has zero wins — ensure it doesn't collapse to literal zero.
    wins = {("A", "B"): 5, ("B", "A"): 5, ("A", "C"): 5, ("B", "C"): 5}
    s = bt_mle(wins, iters=100)
    assert s["C"] > 0.0
    assert s["A"] > s["C"]
    assert s["B"] > s["C"]
