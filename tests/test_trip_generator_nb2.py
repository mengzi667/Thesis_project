from __future__ import annotations

import random

import numpy as np

from simulation.trip_generator import OmegaODTripGenerator


def _sample_counts(
    *,
    arrival_dist: str,
    phi_mode: str,
    phi_global: float,
    phi_table: dict[tuple[int, int], float] | None = None,
    n_rep: int = 2000,
    mu: float = 2.0,
) -> np.ndarray:
    counts = []
    for k in range(n_rep):
        gen = OmegaODTripGenerator(
            od_slot_expected={(1, 2): {0: mu}},
            planning_period=15.0,
            rng=random.Random(1000 + k),
            arrival_dist=arrival_dist,
            phi_mode=phi_mode,
            phi_global=phi_global,
            phi_min=0.05,
            phi_max=5.0,
            phi_table=phi_table or {},
            slot_to_hour_fn=lambda s: 6,
            is_weekend=1,
        )
        trips = gen.generate_trips(sim_duration=15.0)
        counts.append(len(trips))
    return np.asarray(counts, dtype=np.float64)


def test_nb2_has_higher_dispersion_than_poisson() -> None:
    c_p = _sample_counts(arrival_dist="poisson", phi_mode="global", phi_global=0.8)
    c_n = _sample_counts(arrival_dist="nb2", phi_mode="global", phi_global=1.0)

    mean_p = float(c_p.mean())
    var_p = float(c_p.var(ddof=1))
    mean_n = float(c_n.mean())
    var_n = float(c_n.var(ddof=1))

    assert mean_p > 0.0 and mean_n > 0.0
    assert var_p / mean_p < 1.25
    assert var_n / mean_n > 1.6


def test_nb2_phi_zero_falls_back_to_poisson() -> None:
    c = _sample_counts(arrival_dist="nb2", phi_mode="global", phi_global=0.0)
    mean_c = float(c.mean())
    var_c = float(c.var(ddof=1))
    assert mean_c > 0.0
    assert var_c / mean_c < 1.25


def test_phi_routing_by_hour() -> None:
    gen = OmegaODTripGenerator(
        od_slot_expected={(1, 2): {0: 1.0}},
        planning_period=15.0,
        rng=random.Random(7),
        arrival_dist="nb2",
        phi_mode="by_hour",
        phi_global=0.9,
        phi_min=0.05,
        phi_max=5.0,
        phi_table={(-1, 6): 1.5, (1, 6): 0.6},
        slot_to_hour_fn=lambda s: 6,
        is_weekend=1,
    )
    # by_hour should prioritize (-1, hour) key over (is_weekend, hour)
    assert abs(gen._resolve_phi(slot=0) - 1.5) < 1e-12


def test_phi_routing_by_hour_weektype() -> None:
    gen = OmegaODTripGenerator(
        od_slot_expected={(1, 2): {0: 1.0}},
        planning_period=15.0,
        rng=random.Random(7),
        arrival_dist="nb2",
        phi_mode="by_hour_weektype",
        phi_global=0.9,
        phi_min=0.05,
        phi_max=5.0,
        phi_table={(1, 6): 1.2},
        slot_to_hour_fn=lambda s: 6,
        is_weekend=1,
    )
    assert abs(gen._resolve_phi(slot=0) - 1.2) < 1e-12

