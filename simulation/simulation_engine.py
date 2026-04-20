# simulation_engine.py
# Event-driven simulation engine.
#
# Implements the 8-step simulation loop from the requirements:
#
#   Step 1  Trip arrives
#   Step 2  Query OR interface for relocation opportunity
#   Step 3  Retrieve recommended target zone (if any)
#   Step 4  Single-layer user decision (offer/base/opt_out)
#   Step 5  Select scooter and execute trip
#   Step 6  Update battery state
#   Step 7  Update zone inventories
#   Step 8  Record metrics
#
# Decision logic (Steps 4-5) is cleanly separated from environment dynamics
# (Steps 6-7) to allow future RL agent integration with minimal changes.

from __future__ import annotations

from typing import Dict, List, Optional

from config import (
    SARA_RIDE_SPEED_KMH,
    SARA_WALK_SPEED_KMH,
    SIM_DURATION,
    SNAPSHOT_INTERVAL,
)
from simulation.spatial_system import SpatialSystem
from simulation.fleet_manager import FleetManager, Scooter
from simulation.trip_generator import TripRequest, TripGenerator
from simulation.user_choice_model import UserChoiceModel
from or_model.or_interface import ORInterface, RelocationOpportunity
from simulation.metrics_logger import MetricsLogger, TripRecord, UNSERVED_NO_SUPPLY, UNSERVED_OPT_OUT
from rl.runtime import (
    DecisionContext,
    Scenario1FeatureBuilder,
    TransitionLogger,
    estimate_zone_edl,
    reward_hybrid,
)


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
        trip_gen: TripGenerator,
        user_model: UserChoiceModel,
        or_interface: ORInterface,
        logger: MetricsLogger,
        demand_profile=None,
        rl_policy=None,
        rl_feature_builder: Optional[Scenario1FeatureBuilder] = None,
        rl_transition_logger: Optional[TransitionLogger] = None,
        rl_reward_cfg: Optional[Dict[str, float]] = None,
        episode_minutes: float = 120.0,
        budget_remaining: float = 0.0,
        edl_model=None,
        ride_time_minutes: Optional[Dict[int, Dict[int, float]]] = None,
        snapshot_interval: float = SNAPSHOT_INTERVAL,
        verbose: bool = False,
        print_snapshots: bool = True,
    ) -> None:
        self.spatial = spatial
        self.fleet = fleet
        self.trip_gen = trip_gen
        self.user_model = user_model
        self.or_interface = or_interface
        self.logger = logger
        self.demand_profile = demand_profile
        self.rl_policy = rl_policy
        self.rl_feature_builder = rl_feature_builder
        self.rl_transition_logger = rl_transition_logger
        self.rl_reward_cfg = rl_reward_cfg or {"reward_lambda": 0.7, "beta_c": 1.0, "beta_r": 0.1, "l_ref": 1.0, "e_ref": 1.0}
        self.episode_minutes = float(episode_minutes)
        self.budget_remaining = float(budget_remaining)
        self.zone_ids = sorted(self.spatial.all_zone_ids())
        self.edl_model = edl_model
        self.ride_time_minutes = ride_time_minutes or {}
        self.snapshot_interval = snapshot_interval
        self.verbose = verbose
        self.print_snapshots = bool(print_snapshots)
        self._current_time: float = 0.0
        # Delayed reward bookkeeping (Rina-style):
        # accumulate realized loss between two RL decision points.
        self._pending_rl_transition: Optional[dict] = None
        self._pending_realized_loss: float = 0.0

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
        self._pending_rl_transition = None
        self._pending_realized_loss = 0.0

        for trip in trips:
            self._current_time = trip.request_time

            # Periodic inventory snapshot
            while self._current_time >= next_snapshot:
                zone_state = self.spatial.get_state_snapshot()
                self.logger.snapshot_inventories(
                    time=next_snapshot,
                    zone_state=zone_state,
                )
                if self.print_snapshots:
                    self._print_snapshot(next_snapshot, zone_state)
                next_snapshot += self.snapshot_interval

            self._process_trip(trip)

        # Flush last pending RL transition at episode end.
        if self._pending_rl_transition is not None:
            self._finalize_pending_transition(
                next_state=self._pending_rl_transition["state"],
                done=1.0,
            )

        # Final snapshot at end of simulation
        zone_state = self.spatial.get_state_snapshot()
        self.logger.snapshot_inventories(
            time=self._current_time,
            zone_state=zone_state,
        )
        if self.print_snapshots:
            self._print_snapshot(self._current_time, zone_state, final=True)
        self.logger.set_or_injection_metrics(self.or_interface.stats())
        return self.logger

    # ── Core simulation loop ─────────────────────────────────────────────────

    def _process_trip(self, trip: TripRequest) -> None:
        """
        Execute the full loop for a single trip request.
        """
        # ── Step 2 & 3: Query OR interface ────────────────────────────────────
        reloc_opp: Optional[RelocationOpportunity] = self.or_interface.query(
            origin=trip.origin_zone,
            destination=trip.destination_zone,
            request_time=trip.request_time,
        )

        # ── Step 4: Single-layer decision (offer/base/opt_out) ──────────────
        relocation_offered = reloc_opp is not None
        relocation_accepted = False
        effective_dest = trip.destination_zone
        extra_walk: float = 0.0
        user_choice: Optional[str] = None
        rl_state = None
        rl_action = None
        rl_edl_before: Dict[int, float] = {}
        rl_slot_d = None
        rl_slot_i = None
        rt_base_min = 0.0
        rt_offer_min = 0.0
        walk_offer_min = 0.0

        if reloc_opp is not None:
            extra_walk = self.spatial.distance_between(
                trip.destination_zone, reloc_opp.recommended_dest
            )

        # ── Step 5: Scooter supply check ─────────────────────────────────────
        available = self.fleet.get_available_scooters(
            trip.origin_zone, current_time=trip.request_time
        )

        # Distinguish supply shortage from user opt-out *before* calling the
        # choice model, so KPI / EDL analysis can treat them separately.
        if not available:
            if reloc_opp is not None:
                self.or_interface.consume_after_decision(reloc_opp, accepted=False)
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
                user_choice=None,
                unserved_reason=UNSERVED_NO_SUPPLY,
            ))
            if self.verbose:
                self._print_trip_event(
                    trip, reloc_opp, extra_walk, relocation_accepted,
                    chosen=None, effective_dest=effective_dest,
                    outcome=UNSERVED_NO_SUPPLY,
                )
            self._record_realized_loss(1.0)
            return

        # ── Step 4 (cont.): compute trip/offer attributes ────────────────────
        if reloc_opp is not None:
            walk_offer_min = self._meters_to_minutes(extra_walk, SARA_WALK_SPEED_KMH)
            rt_base_min = self._ride_minutes_between(
                trip.origin_zone, trip.destination_zone, SARA_RIDE_SPEED_KMH
            )
            rt_offer_min = self._ride_minutes_between(
                trip.origin_zone, reloc_opp.recommended_dest, SARA_RIDE_SPEED_KMH
            )

            # Optional RL decision hook at offer opportunities
            if self.rl_policy is not None and self.rl_feature_builder is not None:
                zone_state_before = self.spatial.get_state_snapshot()
                slot_req = int(trip.request_time // self.snapshot_interval)
                rl_slot_d = int((trip.request_time + rt_base_min) // self.snapshot_interval)
                rl_slot_i = int((trip.request_time + rt_offer_min) // self.snapshot_interval)
                for z in (trip.origin_zone, trip.destination_zone, reloc_opp.recommended_dest):
                    rl_edl_before[z] = estimate_zone_edl(
                        zone_id=z,
                        slot_idx=slot_req,
                        zone_state=zone_state_before,
                        planning_period=self.snapshot_interval,
                        demand_profile=self.demand_profile,
                        zone_ids=self.zone_ids,
                        edl_model=self.edl_model,
                    )
                ctx = DecisionContext(
                    request_time=trip.request_time,
                    planning_period=self.snapshot_interval,
                    episode_minutes=self.episode_minutes,
                    origin=trip.origin_zone,
                    destination=trip.destination_zone,
                    recommended=reloc_opp.recommended_dest,
                    rt_base_min=rt_base_min,
                    rt_offer_min=rt_offer_min,
                    walk_extra_min=walk_offer_min,
                    incentive_amount=reloc_opp.incentive_amount,
                    offered=True,
                    accepted=False,
                    rejected=False,
                    quota_remaining=float(reloc_opp.quota_remaining),
                    budget_remaining=float(self.budget_remaining),
                    zone_state_before=zone_state_before,
                    zone_state_after=zone_state_before,
                )
                rl_state = self.rl_feature_builder.build(ctx, rl_edl_before)
                # Delayed reward: finalize previous decision when new decision state arrives.
                self._finalize_pending_transition(next_state=rl_state, done=0.0)
                rl_action = int(self.rl_policy.act(rl_state))
                if rl_action == 0:
                    relocation_offered = False
                else:
                    relocation_offered = True

        # Single-layer user decision.
        # In Scenario 1 the scooter assignment is deterministic (highest battery),
        # so both offer/base branches observe the same origin scooter quality.
        best_battery_pct = max(float(s.battery_level) for s in available) * 100.0
        user_choice = self.user_model.decide_trip_action(
            has_offer=bool(relocation_offered),
            incentive_amount=float(reloc_opp.incentive_amount) if (reloc_opp and relocation_offered) else 0.0,
            walk_offer_min=walk_offer_min,
            walk_base_min=0.0,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=best_battery_pct,
            battery_base=best_battery_pct,
            user_type=trip.user_type,
        )

        if user_choice == "opt_out":
            relocation_accepted = False
            if reloc_opp is not None:
                self.or_interface.consume_after_decision(reloc_opp, accepted=False)
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
                relocation_accepted=False,
                scooter_id=None,
                user_choice="opt_out",
                unserved_reason=UNSERVED_OPT_OUT,
            ))
            self._queue_pending_rl_transition(
                trip=trip,
                reloc_opp=reloc_opp,
                rl_state=rl_state,
                rl_action=rl_action,
                rl_edl_before=rl_edl_before,
                rl_slot_d=rl_slot_d,
                rl_slot_i=rl_slot_i,
                relocation_offered=relocation_offered,
                relocation_accepted=False,
                chosen=None,
                zone_state_after=self.spatial.get_state_snapshot(),
            )
            if self.verbose:
                self._print_trip_event(
                    trip, reloc_opp, extra_walk, relocation_accepted,
                    chosen=None, effective_dest=effective_dest,
                    outcome=UNSERVED_OPT_OUT,
                )
            return

        relocation_accepted = bool(
            user_choice == "offer" and reloc_opp is not None and relocation_offered
        )
        if relocation_accepted and reloc_opp is not None:
            effective_dest = reloc_opp.recommended_dest

        if reloc_opp is not None:
            self.or_interface.consume_after_decision(reloc_opp, accepted=relocation_accepted)

        # ── Step 5 (cont.): Scooter assignment (Scenario 1 rule) ────────────
        chosen: Optional[Scooter] = self._decide_scooter(available, trip)

        if chosen is None:
            # Defensive guard: should not happen if 'available' is non-empty.
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
                user_choice=user_choice,
                unserved_reason=UNSERVED_NO_SUPPLY,
            ))
            if self.verbose:
                self._print_trip_event(
                    trip, reloc_opp, extra_walk, relocation_accepted,
                    chosen=None, effective_dest=effective_dest,
                    outcome=UNSERVED_NO_SUPPLY,
                )
            self._record_realized_loss(1.0)
            return

        # ── Step 5 (cont.): Pickup ───────────────────────────────────────────
        self.fleet.pickup_scooter(chosen)

        # ── Steps 6 & 7: Drop-off — battery + zone inventories ──────────────
        arrival_time = trip.request_time + trip.trip_duration
        self.fleet.dropoff_scooter(
            scooter=chosen,
            dest_zone=effective_dest,
            arrival_time=arrival_time,
        )

        # ── Step 8: Record metrics ───────────────────────────────────────────
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
            user_choice=user_choice,
            unserved_reason=None,
        ))

        self._queue_pending_rl_transition(
            trip=trip,
            reloc_opp=reloc_opp,
            rl_state=rl_state,
            rl_action=rl_action,
            rl_edl_before=rl_edl_before,
            rl_slot_d=rl_slot_d,
            rl_slot_i=rl_slot_i,
            relocation_offered=relocation_offered,
            relocation_accepted=relocation_accepted,
            chosen=chosen,
            zone_state_after=self.spatial.get_state_snapshot(),
        )
        if self.verbose:
            self._print_trip_event(
                trip, reloc_opp, extra_walk, relocation_accepted,
                chosen=chosen, effective_dest=effective_dest,
                outcome="served",
            )

    # ── Decision hook ─────────────────────────────────────────────────────────

    def _queue_pending_rl_transition(
        self,
        trip: TripRequest,
        reloc_opp: Optional[RelocationOpportunity],
        rl_state,
        rl_action,
        rl_edl_before: Dict[int, float],
        rl_slot_d,
        rl_slot_i,
        relocation_offered: bool,
        relocation_accepted: bool,
        chosen: Optional[Scooter],
        zone_state_after: Dict[int, tuple],
    ) -> None:
        if (
            self.rl_transition_logger is None
            or self.rl_feature_builder is None
            or rl_state is None
            or rl_action is None
            or reloc_opp is None
            or rl_slot_d is None
            or rl_slot_i is None
        ):
            return

        edl_after_d = estimate_zone_edl(
            zone_id=trip.destination_zone,
            slot_idx=rl_slot_d,
            zone_state=zone_state_after,
            planning_period=self.snapshot_interval,
            demand_profile=self.demand_profile,
            zone_ids=self.zone_ids,
            edl_model=self.edl_model,
        )
        edl_after_i = estimate_zone_edl(
            zone_id=reloc_opp.recommended_dest,
            slot_idx=rl_slot_i,
            zone_state=zone_state_after,
            planning_period=self.snapshot_interval,
            demand_profile=self.demand_profile,
            zone_ids=self.zone_ids,
            edl_model=self.edl_model,
        )
        reject_flag = bool(relocation_offered and (not relocation_accepted))
        # Reward cost accounting: real payout only when user accepts.
        cost_term = float(reloc_opp.incentive_amount) if relocation_accepted else 0.0

        if self.edl_model is not None:
            # Sara-aligned cumulative EDL comparison:
            # baseline(no-offer) vs actual action outcome, from t+RT to episode end.
            d = int(trip.destination_zone)
            i = int(reloc_opp.recommended_dest)
            t_end_slot = int(max(0, self.episode_minutes // self.snapshot_interval))

            actual_d = tuple(zone_state_after.get(d, (0, 0, 0)))
            actual_i = tuple(zone_state_after.get(i, (0, 0, 0)))
            base_d = actual_d
            base_i = actual_i

            # Build baseline counterfactual state from actual state.
            # If relocation was accepted, move one scooter of the realized
            # post-trip battery class back from i to d.
            if relocation_accepted and d != i and chosen is not None:
                cat_idx = {"inactive": 0, "low": 1, "high": 2}.get(str(chosen.battery_category), 2)
                bd = list(base_d)
                bi = list(base_i)
                if bi[cat_idx] > 0:
                    bi[cat_idx] -= 1
                    bd[cat_idx] += 1
                base_d = tuple(bd)
                base_i = tuple(bi)

            base_cum_edl = float(
                self.edl_model.station_cumulative_edl(d, int(rl_slot_d), t_end_slot, base_d)
                + self.edl_model.station_cumulative_edl(i, int(rl_slot_i), t_end_slot, base_i)
            )
            actual_cum_edl = float(
                self.edl_model.station_cumulative_edl(d, int(rl_slot_d), t_end_slot, actual_d)
                + self.edl_model.station_cumulative_edl(i, int(rl_slot_i), t_end_slot, actual_i)
            )
            delta_edl = float(base_cum_edl - actual_cum_edl)
        else:
            base_cum_edl = float(
                rl_edl_before.get(trip.destination_zone, 0.0)
                + rl_edl_before.get(reloc_opp.recommended_dest, 0.0)
            )
            actual_cum_edl = float(edl_after_d + edl_after_i)
            delta_edl = float(base_cum_edl - actual_cum_edl)

        # Delay reward until next decision: keep this as pending transition.
        self._pending_rl_transition = {
            "request_id": trip.request_id,
            "time_min": float(trip.request_time),
            "origin": trip.origin_zone,
            "destination": trip.destination_zone,
            "recommended_dest": reloc_opp.recommended_dest,
            "action": int(rl_action),
            "offered": int(bool(relocation_offered)),
            "accepted": int(bool(relocation_accepted)),
            "reject_flag": int(reject_flag),
            "delta_edl": float(delta_edl),
            "base_cum_edl": float(base_cum_edl),
            "actual_cum_edl": float(actual_cum_edl),
            "cost_term": float(cost_term),
            "state": rl_state.astype("float32"),
        }
        self._pending_realized_loss = 0.0

    def _record_realized_loss(self, value: float) -> None:
        if self._pending_rl_transition is None:
            return
        self._pending_realized_loss += float(max(0.0, value))

    def _finalize_pending_transition(self, next_state, done: float) -> None:
        if self._pending_rl_transition is None or self.rl_transition_logger is None:
            return
        p = self._pending_rl_transition
        realized_loss = float(self._pending_realized_loss)
        delta_edl = float(p["delta_edl"])
        cost_term = float(p["cost_term"])
        reject_flag = bool(p["reject_flag"])
        reward_lambda = float(self.rl_reward_cfg.get("reward_lambda", 0.7))
        l_ref = max(1e-9, float(self.rl_reward_cfg.get("l_ref", 1.0)))
        e_ref = max(1e-9, float(self.rl_reward_cfg.get("e_ref", 1.0)))
        realized_norm = min(max(realized_loss / l_ref, 0.0), 1.0)
        delta_edl_norm = min(max(delta_edl / e_ref, -1.0), 1.0)
        reward_realized_term = reward_lambda * (-realized_norm)
        reward_edl_term = (1.0 - reward_lambda) * delta_edl_norm
        reward = reward_hybrid(
            reward_lambda=reward_lambda,
            beta_c=float(self.rl_reward_cfg.get("beta_c", 1.0)),
            beta_r=float(self.rl_reward_cfg.get("beta_r", 0.1)),
            l_ref=l_ref,
            e_ref=e_ref,
            realized_loss=realized_loss,
            delta_edl=delta_edl,
            cost_term=cost_term,
            reject_flag=reject_flag,
        )
        self.rl_transition_logger.append(
            {
                "request_id": int(p["request_id"]),
                "time_min": float(p["time_min"]),
                "origin": int(p["origin"]),
                "destination": int(p["destination"]),
                "recommended_dest": int(p["recommended_dest"]),
                "action": int(p["action"]),
                "offered": int(p["offered"]),
                "accepted": int(p["accepted"]),
                "reject_flag": int(p["reject_flag"]),
                "delta_edl": float(delta_edl),
                "base_cum_edl": float(p.get("base_cum_edl", 0.0)),
                "actual_cum_edl": float(p.get("actual_cum_edl", 0.0)),
                "realized_loss": float(realized_loss),
                "cost_term": float(cost_term),
                "reward_realized_term": float(reward_realized_term),
                "reward_edl_term": float(reward_edl_term),
                "reward": float(reward),
                "done": float(done),
                "state": p["state"],
                "next_state": next_state.astype("float32"),
            }
        )
        self._pending_rl_transition = None
        self._pending_realized_loss = 0.0

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

        UserChoiceModel is NOT involved here — user action selection is handled earlier in the single-layer decision step. For future RL integration, replace
        this method body with agent.select_action(state).
        """
        if not available:
            return None
        return available[0]   # highest-battery scooter, strict priority (PRD §14)

    # ── Acceptance utility helpers ──────────────────────────────────────────

    def _meters_to_minutes(self, distance_m: float, speed_kmh: float) -> float:
        """Convert meter-scale distance to minutes under constant speed."""
        if speed_kmh <= 0:
            return 0.0
        speed_m_per_min = (speed_kmh * 1000.0) / 60.0
        if speed_m_per_min <= 0:
            return 0.0
        return max(0.0, float(distance_m)) / speed_m_per_min

    def _ride_minutes_between(self, zone_a: int, zone_b: int, speed_kmh: float) -> float:
        """
        Ride-time input for acceptance model.
        Priority:
          1) Sara minute_RT_ij matrix (if provided)
          2) Fallback distance/speed conversion
        """
        if self.ride_time_minutes:
            row = self.ride_time_minutes.get(int(zone_a))
            if row is not None and int(zone_b) in row:
                return max(0.0, float(row[int(zone_b)]))
        distance_m = self.spatial.distance_between(zone_a, zone_b)
        return self._meters_to_minutes(distance_m, speed_kmh)
