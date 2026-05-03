from __future__ import annotations

import numpy as np

from rl.runtime import reward_hybrid


def test_reward_positive_delta_edl_case() -> None:
    r = reward_hybrid(
        w_l=0.7,
        w_e=0.3,
        beta_a=0.0,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=1.0,
        cost_term=0.0,
        accept_flag=False,
        reject_flag=False,
    )
    # 0.7 * 0 + 0.3 * 1 = 0.3
    assert np.isclose(r, 0.3)


def test_reward_reject_penalty_case() -> None:
    r = reward_hybrid(
        w_l=0.7,
        w_e=0.3,
        beta_a=0.0,
        beta_c=0.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        accept_flag=False,
        reject_flag=True,
    )
    assert np.isclose(r, -0.1)


def test_reward_no_offer_case_zero() -> None:
    r = reward_hybrid(
        w_l=0.7,
        w_e=0.3,
        beta_a=0.0,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        accept_flag=False,
        reject_flag=False,
    )
    assert np.isclose(r, 0.0)


def test_reward_clip_and_cost_case() -> None:
    r = reward_hybrid(
        w_l=0.7,
        w_e=0.3,
        beta_a=0.0,
        beta_c=1.0,
        beta_r=0.1,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=2.0,  # clipped to 1.0
        delta_edl=3.0,      # clipped to 1.0
        cost_term=1.0,
        accept_flag=False,
        reject_flag=False,
    )
    # -0.7 + 0.3 - 1.0 = -1.4
    assert np.isclose(r, -1.4)


def test_reward_accept_bonus_increment() -> None:
    base = reward_hybrid(
        w_l=0.5,
        w_e=0.5,
        beta_a=0.2,
        beta_c=0.3,
        beta_r=0.02,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        accept_flag=False,
        reject_flag=False,
    )
    accepted = reward_hybrid(
        w_l=0.5,
        w_e=0.5,
        beta_a=0.2,
        beta_c=0.3,
        beta_r=0.02,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=0.0,
        cost_term=0.0,
        accept_flag=True,
        reject_flag=False,
    )
    assert np.isclose(accepted - base, 0.2)
