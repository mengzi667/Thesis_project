from __future__ import annotations

import numpy as np

from rl.runtime import DecisionContext, Scenario1FeatureBuilder


def test_state_builder_shape_and_ordering() -> None:
    fb = Scenario1FeatureBuilder(zone_ids=[1, 2, 3], zone_capacity=10, planning_period=15, episode_minutes=120)
    ctx = DecisionContext(
        request_time=30.0,
        planning_period=15.0,
        episode_minutes=120.0,
        origin=1,
        destination=2,
        recommended=3,
        rt_base_min=5.0,
        rt_offer_min=7.0,
        walk_extra_min=2.0,
        incentive_amount=1.0,
        offered=True,
        accepted=False,
        rejected=True,
        quota_remaining=2.0,
        budget_remaining=100.0,
        zone_state_before={1: (0, 4, 5), 2: (1, 3, 6), 3: (2, 2, 6)},
        zone_state_after={1: (0, 4, 5), 2: (1, 3, 6), 3: (2, 2, 6)},
    )
    edl = {1: 0.5, 2: 1.0, 3: 2.0}
    s = fb.build(ctx, edl)

    assert s.shape == (22,)
    # Check stable ordering at key indices.
    assert np.isclose(s[0], 30.0 / 120.0)  # slot_norm
    assert np.isclose(s[1], 1 / 3)         # o id norm
    assert np.isclose(s[2], 2 / 3)         # d id norm
    assert np.isclose(s[3], 3 / 3)         # i id norm
    assert np.isclose(s[13], 0.5)          # edl_o
    assert np.isclose(s[14], 1.0)          # edl_d
    assert np.isclose(s[15], 2.0)          # edl_i


def test_state_builder_normalization_nonnegative() -> None:
    fb = Scenario1FeatureBuilder(zone_ids=[1, 2], zone_capacity=10, planning_period=15, episode_minutes=120)
    ctx = DecisionContext(
        request_time=0.0,
        planning_period=15.0,
        episode_minutes=120.0,
        origin=1,
        destination=2,
        recommended=2,
        rt_base_min=0.0,
        rt_offer_min=0.0,
        walk_extra_min=0.0,
        incentive_amount=1.0,
        offered=False,
        accepted=False,
        rejected=False,
        quota_remaining=0.0,
        budget_remaining=0.0,
        zone_state_before={1: (0, 0, 0), 2: (0, 0, 0)},
        zone_state_after={1: (0, 0, 0), 2: (0, 0, 0)},
    )
    s = fb.build(ctx, {1: 0.0, 2: 0.0})
    assert np.all(s >= 0.0)
