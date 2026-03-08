# simulation_engine.py
# Event-driven simulation engine.
#
# Implements the 8-step simulation loop from the requirements:
#
#   Step 1  Trip arrives
#   Step 2  Query OR interface for relocation opportunity
#   Step 3  Retrieve recommended target zone (if any)
#   Step 4  Simulate user relocation decision
#   Step 5  Select scooter and execute trip
#   Step 6  Update battery state
#   Step 7  Update zone inventories
#   Step 8  Record metrics
#
# Decision logic (Steps 4-5) is cleanly separated from environment dynamics
# (Steps 6-7) to allow future RL agent integration with minimal changes.

from __future__ import annotations

from typing import List, Optional

from config import SIM_DURATION, SNAPSHOT_INTERVAL
from spatial_system import SpatialSystem
from fleet_manager import FleetManager, Scooter
from trip_generator import TripRequest, PoissonTripGenerator
from user_choice_model import UserChoiceModel
from or_interface import ORInterface, RelocationOpportunity
from metrics_logger import MetricsLogger, TripRecord


class SimulationEngine:
    """
    Orchestrates all simulator components and runs the event-driven loop.

    The engine exposes:
      - run()          : execute a full simulation and return the MetricsLogger.
      - current_time   : read-only property for the current simulation clock.

    Future RL integration:
      Replace ``_decide_scooter()`` and the relocation branch with
      ``agent.select_action(state)`` — the environment dynamics (Steps 6-8)
      require no changes.
    """

    def __init__(
        self,
        spatial: SpatialSystem,
        fleet: FleetManager,
        trip_gen: PoissonTripGenerator,
        user_model: UserChoiceModel,
        or_interface: ORInterface,
        logger: MetricsLogger,
        snapshot_interval: float = SNAPSHOT_INTERVAL,
    ) -> None:
        self.spatial = spatial
        self.fleet = fleet
        self.trip_gen = trip_gen
        self.user_model = user_model
        self.or_interface = or_interface
        self.logger = logger
        self.snapshot_interval = snapshot_interval
        self._current_time: float = 0.0

    @property
    def current_time(self) -> float:
        return self._current_time

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, sim_duration: float = SIM_DURATION) -> MetricsLogger:
        """
        Pre-generate all trip requests then process them chronologically.
        Returns the populated MetricsLogger for inspection or export.
        """
        trips = self.trip_gen.generate_trips(sim_duration)
        next_snapshot = self.snapshot_interval

        for trip in trips:
            self._current_time = trip.request_time

            # Periodic inventory snapshot
            while self._current_time >= next_snapshot:
                self.logger.snapshot_inventories(
                    time=next_snapshot,
                    zone_state=self.spatial.get_state_snapshot(),
                )
                next_snapshot += self.snapshot_interval

            self._process_trip(trip)

        # Final snapshot at end of simulation
        self.logger.snapshot_inventories(
            time=self._current_time,
            zone_state=self.spatial.get_state_snapshot(),
        )
        return self.logger

    # ── Core simulation loop ─────────────────────────────────────────────────

    def _process_trip(self, trip: TripRequest) -> None:
        """
        Execute the full 8-step loop for a single trip request.
        """
        # ── Step 2 & 3: Query OR interface ────────────────────────────────────
        reloc_opp: Optional[RelocationOpportunity] = self.or_interface.query(
            origin=trip.origin_zone,
            destination=trip.destination_zone,
            request_time=trip.request_time,
        )

        # ── Step 4: User relocation decision ─────────────────────────────────
        relocation_offered = reloc_opp is not None
        relocation_accepted = False
        effective_dest = trip.destination_zone

        if reloc_opp is not None:
            extra_walk = self.spatial.distance_between(
                trip.destination_zone, reloc_opp.recommended_dest
            )
            relocation_accepted = self.user_model.accept_relocation(
                incentive_amount=reloc_opp.incentive_amount,
                extra_walking_meters=extra_walk,
                user_type=trip.user_type,
            )
            if relocation_accepted:
                effective_dest = reloc_opp.recommended_dest

        # ── Step 5: Scooter selection — default logic: high-battery first ─────
        available = self.fleet.get_available_scooters(
            trip.origin_zone, current_time=trip.request_time
        )
        chosen: Optional[Scooter] = self._decide_scooter(available, trip)

        if chosen is None:
            # No rentable scooter or user opted out — record as unserved
            self.logger.log_trip(TripRecord(
                request_id=trip.request_id,
                origin_zone=trip.origin_zone,
                effective_dest=effective_dest,
                request_time=trip.request_time,
                trip_duration=trip.trip_duration,
                trip_distance=trip.trip_distance,
                user_type=trip.user_type,
                served=False,
                relocation_offered=relocation_offered,
                relocation_accepted=relocation_accepted,
                scooter_id=None,
            ))
            return

        # ── Step 5 (cont.): Pickup ────────────────────────────────────────────
        self.fleet.pickup_scooter(chosen)

        # ── Steps 6 & 7: Drop-off — updates battery + zone inventories ────────
        arrival_time = trip.request_time + trip.trip_duration
        self.fleet.dropoff_scooter(
            scooter=chosen,
            dest_zone=effective_dest,
            distance_km=trip.trip_distance,
            arrival_time=arrival_time,
        )

        # ── Step 8: Record metrics ────────────────────────────────────────────
        self.logger.log_trip(TripRecord(
            request_id=trip.request_id,
            origin_zone=trip.origin_zone,
            effective_dest=effective_dest,
            request_time=trip.request_time,
            trip_duration=trip.trip_duration,
            trip_distance=trip.trip_distance,
            user_type=trip.user_type,
            served=True,
            relocation_offered=relocation_offered,
            relocation_accepted=relocation_accepted,
            scooter_id=chosen.scooter_id,
        ))

    # ── Decision hook ─────────────────────────────────────────────────────────

    def _decide_scooter(
        self, available: List[Scooter], trip: TripRequest
    ) -> Optional[Scooter]:
        """
        Default scooter selection: delegate to the user choice model.
        FleetManager already returns scooters in high-battery-first order.

        Override or replace this method to plug in an RL agent:
            return agent.select_action(self._build_state(trip))
        """
        if not available:
            return None
        return self.user_model.choose_scooter(available, user_type=trip.user_type)
