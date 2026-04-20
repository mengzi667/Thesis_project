from __future__ import annotations

import random

import numpy as np

from simulation.user_choice_model import UserChoiceModel


def test_single_layer_probs_with_offer_sum_to_one() -> None:
    m = UserChoiceModel(rng=random.Random(123))
    p = m.choice_probabilities(
        has_offer=True,
        incentive_amount=1.0,
        walk_offer_min=3.0,
        walk_base_min=0.0,
        rt_offer_min=8.0,
        rt_base_min=6.0,
        battery_offer=80.0,
        battery_base=80.0,
        user_type="normal",
    )
    assert set(p.keys()) == {"offer", "base", "opt_out"}
    assert all(v >= 0.0 for v in p.values())
    assert np.isclose(sum(p.values()), 1.0)


def test_single_layer_probs_no_offer_offer_is_zero() -> None:
    m = UserChoiceModel(rng=random.Random(7))
    p = m.choice_probabilities(
        has_offer=False,
        walk_base_min=0.0,
        rt_base_min=7.0,
        battery_base=70.0,
        user_type="normal",
    )
    assert np.isclose(p["offer"], 0.0)
    assert p["base"] >= 0.0 and p["opt_out"] >= 0.0
    assert np.isclose(p["base"] + p["opt_out"], 1.0)


def test_single_layer_action_sampling_reproducible_by_seed() -> None:
    m1 = UserChoiceModel(rng=random.Random(2026))
    m2 = UserChoiceModel(rng=random.Random(2026))

    seq1 = [
        m1.decide_trip_action(
            has_offer=True,
            incentive_amount=1.0,
            walk_offer_min=2.0,
            rt_offer_min=7.0,
            rt_base_min=6.0,
            battery_offer=75.0,
            battery_base=75.0,
        )
        for _ in range(30)
    ]
    seq2 = [
        m2.decide_trip_action(
            has_offer=True,
            incentive_amount=1.0,
            walk_offer_min=2.0,
            rt_offer_min=7.0,
            rt_base_min=6.0,
            battery_offer=75.0,
            battery_base=75.0,
        )
        for _ in range(30)
    ]

    assert seq1 == seq2


def test_no_offer_action_never_returns_offer() -> None:
    m = UserChoiceModel(rng=random.Random(99))
    acts = [
        m.decide_trip_action(
            has_offer=False,
            rt_base_min=6.0,
            battery_base=60.0,
        )
        for _ in range(100)
    ]
    assert all(a in {"base", "opt_out"} for a in acts)
