from __future__ import annotations

import numpy as np

from rl.runtime import reward_hybrid


def test_reward_positive_delta_edl_case() -> None:
    r = reward_hybrid(
        reward_lambda=0.7,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=1.0,
        cost_term=0.0,
        reject_flag=False,
    )
    # 0.7 * 0 + 0.3 * 1 = 0.3
    assert np.isclose(r, 0.3)


def test_reward_reject_penalty_case() -> None:
    r = reward_hybrid(
        reward_lambda=0.7,
        beta_c=0.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        reject_flag=True,
    )
    assert np.isclose(r, -0.1)


def test_reward_no_offer_case_zero() -> None:
    r = reward_hybrid(
        reward_lambda=0.7,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        reject_flag=False,
    )
    assert np.isclose(r, 0.0)


def test_reward_clip_and_cost_case() -> None:
    r = reward_hybrid(
        reward_lambda=0.7,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=2.0,  # clipped to 1.0
        delta_edl=3.0,      # clipped to 1.0
        cost_term=1.0,
        reject_flag=False,
    )
    # -0.7 + 0.3 - 1.0 = -1.4
    assert np.isclose(r, -1.4)
