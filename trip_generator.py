# trip_generator.py
# Stochastic trip request generation.
#
# The generation mechanism (Poisson / historical replay / synthetic OD matrix)
# is encapsulated here so it can be swapped without touching the simulation loop.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from config import (
    TRIP_ARRIVAL_RATE,
    TRIP_DURATION_MEAN,
    TRIP_DURATION_STD,
    TRIP_DISTANCE_MEAN,
    TRIP_DISTANCE_STD,
    USER_TYPES,
    USER_TYPE_WEIGHTS,
)


@dataclass
class TripRequest:
    """
    A single trip request: τ = (origin, destination, request_time, duration, user).
    """
    request_id:       int
    origin_zone:      int
    destination_zone: int
    request_time:     float   # minutes since simulation start
    trip_duration:    float   # minutes
    trip_distance:    float   # km
    user_type:        str

    def __repr__(self) -> str:
        return (
            f"TripRequest(id={self.request_id}, "
            f"o={self.origin_zone}→d={self.destination_zone}, "
            f"t={self.request_time:.1f}min, user={self.user_type})"
        )


class PoissonTripGenerator:
    """
    Generates trip requests following a homogeneous Poisson process (rate λ).

    Origins and destinations are sampled uniformly from the provided zone list.
    Trip duration and distance are drawn from Gaussian distributions.
    The generator is stateless between calls: pass the same rng + seed for
    reproducibility.
    """

    def __init__(
        self,
        zone_ids: List[int],
        arrival_rate: float = TRIP_ARRIVAL_RATE,
        rng: Optional[random.Random] = None,
    ) -> None:
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive.")
        self.zone_ids = list(zone_ids)
        self.arrival_rate = arrival_rate   # trips per minute
        self.rng = rng or random.Random()
        self._counter: int = 0

    def generate_trips(self, sim_duration: float) -> List[TripRequest]:
        """
        Pre-generate all trip requests for [0, sim_duration] using exponential
        inter-arrival times (Poisson process).
        Returns list sorted by request_time.
        """
        trips: List[TripRequest] = []
        current_time: float = 0.0

        while True:
            inter_arrival = self.rng.expovariate(self.arrival_rate)
            current_time += inter_arrival
            if current_time > sim_duration:
                break
            trips.append(self._sample_trip(current_time))

        return trips

    def _sample_trip(self, request_time: float) -> TripRequest:
        origin = self.rng.choice(self.zone_ids)
        dest   = self.rng.choice(self.zone_ids)
        duration = max(1.0, self.rng.gauss(TRIP_DURATION_MEAN, TRIP_DURATION_STD))
        distance = max(0.1, self.rng.gauss(TRIP_DISTANCE_MEAN, TRIP_DISTANCE_STD))
        user_type = self.rng.choices(USER_TYPES, weights=USER_TYPE_WEIGHTS, k=1)[0]
        self._counter += 1
        return TripRequest(
            request_id=self._counter,
            origin_zone=origin,
            destination_zone=dest,
            request_time=request_time,
            trip_duration=duration,
            trip_distance=distance,
            user_type=user_type,
        )
