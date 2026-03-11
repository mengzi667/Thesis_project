# trip_generator.py
# Stochastic trip request generation.
#
# The generation mechanism (Poisson / historical replay / synthetic OD matrix)
# is encapsulated here so it can be swapped without touching the simulation loop.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from config import (
    PLANNING_PERIOD,
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


# ── Demand profile ────────────────────────────────────────────────────────────

class DemandProfile:
    """
    Per-zone, per-planning-period demand specification.

    Captures area-level time-varying pickup/drop-off rates and OD flow
    weights so the simulation can model heterogeneous demand instead of a
    single global Poisson process.

    Attributes
    ----------
    planning_period : slot width in minutes — must equal ORInterface.planning_interval
    zone_rates      : zone_id  -> slot_index -> departure rate (trips / min)
    od_weights      : (origin, dest) -> slot_index -> unnormalised flow weight
    """

    def __init__(
        self,
        planning_period: float,
        zone_rates: Dict[int, Dict[int, float]],
        od_weights: Dict[Tuple[int, int], Dict[int, float]],
    ) -> None:
        self.planning_period = planning_period
        self.zone_rates = zone_rates
        self.od_weights = od_weights

    def rate_for(self, zone_id: int, slot: int) -> float:
        """Departure rate at *zone_id* during planning slot *slot*."""
        return self.zone_rates.get(zone_id, {}).get(slot, 0.0)

    def total_rate(self, slot: int) -> float:
        """Sum of departure rates across all zones in *slot*."""
        return sum(rates.get(slot, 0.0) for rates in self.zone_rates.values())

    def sample_destination(
        self,
        origin: int,
        slot: int,
        zone_ids: List[int],
        rng: random.Random,
    ) -> int:
        """Sample a destination from OD flow weights for (origin, slot)."""
        weights = [
            max(self.od_weights.get((origin, d), {}).get(slot, 1.0), 0.0)
            for d in zone_ids
        ]
        total_w = sum(weights)
        if total_w == 0.0:
            return rng.choice(zone_ids)
        return rng.choices(zone_ids, weights=weights, k=1)[0]

    def dest_arrival_rate(self, dest: int, slot: int, zone_ids: List[int]) -> float:
        """
        Expected scooter arrival rate at zone *dest* in *slot*.
        Accounts for each origin's departure rate and its OD probability toward dest.
        """
        total = 0.0
        for o in zone_ids:
            denom = sum(
                max(self.od_weights.get((o, d2), {}).get(slot, 1.0), 0.0)
                for d2 in zone_ids
            )
            if denom == 0.0:
                continue
            od_prob = max(self.od_weights.get((o, dest), {}).get(slot, 1.0), 0.0) / denom
            total += self.rate_for(o, slot) * od_prob
        return total


# ── Heterogeneous trip generator ──────────────────────────────────────────────

class HeterogeneousTripGenerator:
    """
    Generates trips from a zone-time heterogeneous demand profile.

    For each planning period the per-zone departure rates from *demand_profile*
    drive a piece-wise Poisson process.  Origins are drawn proportional to zone
    rates; destinations are drawn from OD flow weights.

    External interface is identical to PoissonTripGenerator:
      call generate_trips(sim_duration) -> List[TripRequest].
    """

    def __init__(
        self,
        zone_ids: List[int],
        demand_profile: DemandProfile,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.zone_ids = list(zone_ids)
        self.profile  = demand_profile
        self.rng      = rng or random.Random()
        self._counter: int = 0

    def generate_trips(self, sim_duration: float) -> List[TripRequest]:
        """
        Pre-generate all trips for [0, sim_duration] using piece-wise Poisson
        arrivals per planning slot.  Returns list sorted by request_time.
        """
        trips: List[TripRequest] = []
        period    = self.profile.planning_period
        num_slots = max(1, int(sim_duration // period) + 1)

        for slot in range(num_slots):
            slot_start = slot * period
            slot_end   = min((slot + 1) * period, sim_duration)
            if slot_start >= sim_duration:
                break

            zone_rates = [self.profile.rate_for(z, slot) for z in self.zone_ids]
            total_rate = sum(zone_rates)
            if total_rate == 0.0:
                continue

            # Superposition of zone-level Poisson streams: generate arrivals with
            # the combined rate, then assign origin proportional to zone rates.
            t = slot_start
            while True:
                t += self.rng.expovariate(total_rate)
                if t >= slot_end:
                    break
                origin = self.rng.choices(self.zone_ids, weights=zone_rates, k=1)[0]
                dest   = self.profile.sample_destination(
                    origin, slot, self.zone_ids, self.rng
                )
                trips.append(self._make_trip(t, origin, dest))

        return sorted(trips, key=lambda r: r.request_time)

    def _make_trip(self, request_time: float, origin: int, dest: int) -> TripRequest:
        duration  = max(1.0, self.rng.gauss(TRIP_DURATION_MEAN, TRIP_DURATION_STD))
        distance  = max(0.1, self.rng.gauss(TRIP_DISTANCE_MEAN, TRIP_DISTANCE_STD))
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


# Shared type alias — both generators satisfy this interface
TripGenerator = Union[PoissonTripGenerator, HeterogeneousTripGenerator]


# ── Synthetic demand profile builder ─────────────────────────────────────────

def build_synthetic_demand_profile(
    zone_ids: List[int],
    sim_duration: float,
    planning_period: float = PLANNING_PERIOD,
    total_rate: float = TRIP_ARRIVAL_RATE,
    rng: Optional[random.Random] = None,
) -> DemandProfile:
    """
    Build a synthetic zone-time heterogeneous demand profile.

    Time-shape (slot multipliers on base rate per zone):
      warm-up      (t = 0-30 min)  : 0.6x
      morning peak (t = 30-90 min) : 1.8x   -- AM commute
      midday lull  (t = 90-180min) : 0.8x
      afternoon    (t = 180-240)   : 1.2x
      PM peak      (t = 240-300)   : 1.5x   -- PM commute
      evening      (t = 300+ min)  : 0.7x

    Zone roles (zones split into three equal groups):
      Generator zones (first 1/3): residential / offices generating strong AM departures.
      Neutral  zones  (middle)   : balanced profile.
      Attractor zones (last 1/3) : destinations accumulating supply in AM,
                                   generating departures in PM.

    OD structure:
      AM peak  : generator -> attractor flows boosted x3;  counter-flow x0.3
      PM peak  : attractor -> generator flows boosted x3;  counter-flow x0.3
    """
    if rng is None:
        rng = random.Random()

    num_slots = max(1, int(sim_duration // planning_period) + 1)
    base_rate = total_rate / max(len(zone_ids), 1)

    # ── Time-shape multipliers ──────────────────────────────────────────────
    def _time_mult(slot: int) -> float:
        t = slot * planning_period
        if t < 30:   return 0.6
        if t < 90:   return 1.8
        if t < 180:  return 0.8
        if t < 240:  return 1.2
        if t < 300:  return 1.5
        return 0.7

    # ── Zone role assignment ────────────────────────────────────────────────
    n_grp      = max(1, len(zone_ids) // 3)
    generators = set(zone_ids[:n_grp])
    attractors = set(zone_ids[len(zone_ids) - n_grp:])

    def _zone_mult(zone_id: int, slot: int) -> float:
        t = slot * planning_period
        if zone_id in generators:
            if 30 <= t < 90:   return 2.0   # strong AM origin
            if 240 <= t < 300: return 0.8
            return 1.0
        if zone_id in attractors:
            if 30 <= t < 90:   return 0.5   # few AM departures
            if 240 <= t < 300: return 1.8   # strong PM origin
            return 1.0
        return 1.0  # neutral

    # ── Build zone_rates ────────────────────────────────────────────────────
    zone_rates: Dict[int, Dict[int, float]] = {z: {} for z in zone_ids}
    for z in zone_ids:
        for s in range(num_slots):
            zone_rates[z][s] = round(base_rate * _time_mult(s) * _zone_mult(z, s), 6)

    # ── Build OD flow weights ───────────────────────────────────────────────
    od_weights: Dict[Tuple[int, int], Dict[int, float]] = {}
    for o in zone_ids:
        for d in zone_ids:
            od_weights[(o, d)] = {}
            for s in range(num_slots):
                t = s * planning_period
                w = 1.0
                if 30 <= t < 90:
                    if o in generators and d in attractors:   w = 3.0
                    elif o in attractors and d in generators: w = 0.3
                elif 240 <= t < 300:
                    if o in attractors and d in generators:   w = 3.0
                    elif o in generators and d in attractors: w = 0.3
                od_weights[(o, d)][s] = w

    return DemandProfile(
        planning_period=planning_period,
        zone_rates=zone_rates,
        od_weights=od_weights,
    )
