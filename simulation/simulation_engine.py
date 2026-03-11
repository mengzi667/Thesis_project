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
from simulation.spatial_system import SpatialSystem
from simulation.fleet_manager import FleetManager, Scooter
from simulation.trip_generator import TripRequest, PoissonTripGenerator
from simulation.user_choice_model import UserChoiceModel
from or_model.or_interface import ORInterface, RelocationOpportunity
from simulation.metrics_logger import MetricsLogger, TripRecord, UNSERVED_NO_SUPPLY, UNSERVED_OPT_OUT


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
        verbose: bool = False,
    ) -> None:
        self.spatial = spatial
        self.fleet = fleet
        self.trip_gen = trip_gen
        self.user_model = user_model
        self.or_interface = or_interface
        self.logger = logger
        self.snapshot_interval = snapshot_interval
        self.verbose = verbose
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
                zone_state = self.spatial.get_state_snapshot()
                self.logger.snapshot_inventories(
                    time=next_snapshot,
                    zone_state=zone_state,
                )
                self._print_snapshot(next_snapshot, zone_state)
                next_snapshot += self.snapshot_interval

            self._process_trip(trip)

        # Final snapshot at end of simulation
        zone_state = self.spatial.get_state_snapshot()
        self.logger.snapshot_inventories(
            time=self._current_time,
            zone_state=zone_state,
        )
        self._print_snapshot(self._current_time, zone_state, final=True)
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
        extra_walk: float = 0.0

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

        # Distinguish supply shortage from user opt-out *before* calling the
        # choice model, so KPI / EDL analysis can treat them separately.
        if not available:
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
                unserved_reason=UNSERVED_NO_SUPPLY,
            ))
            if self.verbose:
                self._print_trip_event(
                    trip, reloc_opp, extra_walk, relocation_accepted,
                    chosen=None, effective_dest=effective_dest,
                    outcome=UNSERVED_NO_SUPPLY,
                )
            return

        chosen: Optional[Scooter] = self._decide_scooter(available, trip)

        if chosen is None:
            # Scooters were available but user chose not to rent (opt-out).
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
                unserved_reason=UNSERVED_OPT_OUT,
            ))
            if self.verbose:
                self._print_trip_event(
                    trip, reloc_opp, extra_walk, relocation_accepted,
                    chosen=None, effective_dest=effective_dest,
                    outcome=UNSERVED_OPT_OUT,
                )
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
            unserved_reason=None,
        ))
        if self.verbose:
            self._print_trip_event(
                trip, reloc_opp, extra_walk, relocation_accepted,
                chosen=chosen, effective_dest=effective_dest,
                outcome="served",
            )

    # ── Decision hook ─────────────────────────────────────────────────────────

    # ── Output helpers ────────────────────────────────────────────────────────

    def _print_snapshot(
        self,
        t: float,
        zone_state: dict,
        final: bool = False,
    ) -> None:
        """
        Print a tabular inventory snapshot at simulation time *t*.
        Always printed (not gated on verbose) — snapshots are key milestones.
        """
        W = 62
        BAR = "─" * W
        stats = self.logger.running_stats()
        if final:
            label = "FINAL SNAPSHOT"
        else:
            period_idx = int(round(t / self.snapshot_interval))
            label = f"PERIOD d={period_idx:02d}  |  t={t:.1f} min"
        print(f"\n{BAR}")
        print(f"  {label}")
        srate = stats['served'] / stats['total'] if stats['total'] else 0.0
        print(
            f"  Progress : {stats['total']:,} requests | "
            f"{stats['served']:,} served ({srate:.1%}) | "
            f"{stats['no_supply']:,} no-supply | "
            f"{stats['opt_out']:,} opt-out"
        )
        # Zone table
        hdr = f"  {'Zone':>4}  {'Inactive':>8}  {'Low':>5}  {'High':>6}  {'Total':>6}  {'Rentable':>8}"
        print(f"  {'':─<55}")
        print(hdr)
        print(f"  {'':─<55}")
        for zone_id in sorted(zone_state):
            inactive, low, high = zone_state[zone_id]
            total_inv = inactive + low + high
            rentable  = low + high
            print(
                f"  {zone_id:>4}  {inactive:>8}  {low:>5}  {high:>6}"
                f"  {total_inv:>6}  {rentable:>8}"
            )
        print(BAR)

    def _print_trip_event(
        self,
        trip: TripRequest,
        reloc_opp,
        extra_walk: float,
        relocation_accepted: bool,
        chosen,
        effective_dest: int,
        outcome: str,
    ) -> None:
        """Print a one-trip event block (verbose mode only)."""
        IND = " " * 14   # indent to align sub-lines under trip header
        # ── Header line ───────────────────────────────────────────────────
        print(
            f"[t={trip.request_time:6.1f} min]  "
            f"#{trip.request_id:04d}  "
            f"Z{trip.origin_zone}\u2192Z{trip.destination_zone}  "
            f"({trip.user_type})"
        )
        # ── OR offer line ─────────────────────────────────────────────────
        if reloc_opp is not None:
            decision = "\u2713 ACCEPTED" if relocation_accepted else "\u2717 REJECTED"
            print(
                f"{IND}OR offer : Z{reloc_opp.original_dest}\u2192Z{reloc_opp.recommended_dest}  "
                f"+${reloc_opp.incentive_amount:.2f}  extra={extra_walk:.0f}m  \u2192 {decision}"
            )
            if relocation_accepted:
                print(f"{IND}           eff. dest\u2192Z{effective_dest}")
        else:
            print(f"{IND}no OR offer")
        # ── Outcome line ──────────────────────────────────────────────────
        if outcome == "served" and chosen is not None:
            print(
                f"{IND}Scooter #{chosen.scooter_id}  "
                f"SOC={chosen.battery_level:.0%}  [{chosen.battery_category}]  "
                f"\u2713 SERVED  drop-off Z{effective_dest}"
            )
        elif outcome == UNSERVED_NO_SUPPLY:
            print(f"{IND}\u2717 UNSERVED  [no supply at Z{trip.origin_zone}]")
        else:
            print(f"{IND}\u2717 UNSERVED  [user opt-out]")

    def _decide_scooter(
        self, available: List[Scooter], trip: TripRequest
    ) -> Optional[Scooter]:
        """
        Deterministic system rule (PRD §14): always assign the
        highest-battery available scooter.

        FleetManager.get_available_scooters() sorts candidates with
        high-battery first and descending battery level, so available[0] is
        always the best choice.

        UserChoiceModel is NOT involved here — it handles only relocation
        acceptance (accept_relocation).  For future RL integration, replace
        this method body with agent.select_action(state).
        """
        if not available:
            return None
        return available[0]   # highest-battery scooter, strict priority (PRD §14)
