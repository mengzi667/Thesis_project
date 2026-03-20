# main.py
# Entry point for the shared e-scooter simulation.
#
# Usage:
#   python main.py
#
# All parameters are controlled via config.py.

from __future__ import annotations

import os
import random

from config import (
    OR_FIXED_INCENTIVE_EUR,
    OR_FORCE_FIXED_INCENTIVE,
    OR_INPUT_FORMAT,
    OR_INPUT_PATH,
    OR_QUOTA_CONSUME_POLICY,
    OR_SLOT_MINUTES,
    OR_VALIDATION_MODE,
    PLANNING_PERIOD,
    RANDOM_SEED,
    SARA_DATA_DIR,
    SARA_INIT_INVENTORY_CSV,
    SARA_INIT_UNIFORM_H,
    SARA_INIT_UNIFORM_L,
    SARA_INIT_UNIFORM_N,
    SARA_IS_WEEKEND,
    SARA_SLOT0_HOUR,
    SARA_ZONE_CAPACITY,
    WALKING_THRESHOLD,
    ROLLING_HORIZON,
    LOOKAHEAD_HORIZON,
    SIM_DURATION,
)
from simulation.fleet_manager import FleetManager
from simulation.trip_generator import (
    HeterogeneousTripGenerator,
)
from simulation.sara_environment import (
    build_sara_demand_profile,
    build_sara_spatial_system,
    build_uniform_zone_state,
    load_zone_state_from_csv,
)
from simulation.user_choice_model import UserChoiceModel
from or_model.or_interface import ORInterface, build_demand_informed_table
from or_model.sara_adapter import convert_sara_output_to_uodit

from simulation.metrics_logger import MetricsLogger
from simulation.simulation_engine import SimulationEngine


def build_simulation(seed: int = RANDOM_SEED, verbose: bool = True) -> SimulationEngine:
    """
    Wire up all components and return a ready-to-run SimulationEngine.
    A single Random instance is shared across modules for reproducibility.
    """
    rng = random.Random(seed)

    # 1. Spatial system — Sara-aligned H3 stations (IDs 1..N)
    spatial = build_sara_spatial_system(
        data_dir=SARA_DATA_DIR,
        zone_capacity=SARA_ZONE_CAPACITY,
        walking_threshold=WALKING_THRESHOLD,
    )

    # 2. Fleet — Sara-aligned initial zone state
    fleet = FleetManager(spatial, rng=rng)
    zone_ids = spatial.all_zone_ids()
    default_state = (SARA_INIT_UNIFORM_N, SARA_INIT_UNIFORM_L, SARA_INIT_UNIFORM_H)
    if SARA_INIT_INVENTORY_CSV and os.path.isfile(SARA_INIT_INVENTORY_CSV):
        zone_state = load_zone_state_from_csv(
            csv_path=SARA_INIT_INVENTORY_CSV,
            zone_ids=zone_ids,
            default_state=default_state,
        )
    else:
        zone_state = build_uniform_zone_state(
            zone_ids=zone_ids,
            n_count=SARA_INIT_UNIFORM_N,
            l_count=SARA_INIT_UNIFORM_L,
            h_count=SARA_INIT_UNIFORM_H,
        )
    fleet.initialize_fleet_from_zone_state(zone_state)

    # 3. Demand profile — Sara pickup + omega data
    demand_profile = build_sara_demand_profile(
        data_dir=SARA_DATA_DIR,
        zone_ids=zone_ids,
        sim_duration=SIM_DURATION,
        planning_period=PLANNING_PERIOD,
        is_weekend=SARA_IS_WEEKEND,
        slot0_hour=SARA_SLOT0_HOUR,
    )

    # 4. Trip generator (piece-wise Poisson, one Poisson rate per zone per slot)
    trip_gen = HeterogeneousTripGenerator(
        zone_ids=spatial.all_zone_ids(),
        demand_profile=demand_profile,
        rng=rng,
    )

    # 5. User choice model
    user_model = UserChoiceModel(rng=rng)

    # 6. OR interface
    #    Preferred path: load standard U_odit from configured file.
    #    If configured source is Sara-native output, auto-translate once to U_odit.
    max_time_slot = int(SIM_DURATION // PLANNING_PERIOD)
    zone_id_set = set(spatial.all_zone_ids())

    if os.path.isfile(OR_INPUT_PATH):
        try:
            if OR_INPUT_FORMAT.lower() == "json":
                or_interface = ORInterface.load_from_json(
                    OR_INPUT_PATH,
                    planning_interval=PLANNING_PERIOD,
                    quota_consume_policy=OR_QUOTA_CONSUME_POLICY,
                    force_fixed_incentive=OR_FORCE_FIXED_INCENTIVE,
                    fixed_incentive_eur=OR_FIXED_INCENTIVE_EUR,
                    validation_mode=OR_VALIDATION_MODE,
                    valid_zone_ids=zone_id_set,
                    max_time_slot=max_time_slot,
                )
            else:
                or_interface = ORInterface.load_from_csv(
                    OR_INPUT_PATH,
                    planning_interval=PLANNING_PERIOD,
                    quota_consume_policy=OR_QUOTA_CONSUME_POLICY,
                    force_fixed_incentive=OR_FORCE_FIXED_INCENTIVE,
                    fixed_incentive_eur=OR_FIXED_INCENTIVE_EUR,
                    validation_mode=OR_VALIDATION_MODE,
                    valid_zone_ids=zone_id_set,
                    max_time_slot=max_time_slot,
                )
            or_source = f"loaded from configured U_odit: {OR_INPUT_PATH}"
        except Exception:
            translated_path = "data/generated/u_odit_from_sara.csv"
            convert_sara_output_to_uodit(
                input_path=OR_INPUT_PATH,
                output_path=translated_path,
                slot_minutes=OR_SLOT_MINUTES,
            )
            or_interface = ORInterface.load_from_csv(
                translated_path,
                planning_interval=PLANNING_PERIOD,
                quota_consume_policy=OR_QUOTA_CONSUME_POLICY,
                force_fixed_incentive=OR_FORCE_FIXED_INCENTIVE,
                fixed_incentive_eur=OR_FIXED_INCENTIVE_EUR,
                validation_mode=OR_VALIDATION_MODE,
                valid_zone_ids=zone_id_set,
                max_time_slot=max_time_slot,
            )
            or_source = f"translated Sara output -> {translated_path}"
    else:
        u_odit_table = build_demand_informed_table(
            demand_profile=demand_profile,
            zone_ids=spatial.all_zone_ids(),
            sim_duration=SIM_DURATION,
            planning_period=PLANNING_PERIOD,
            rng=rng,
        )
        or_interface = ORInterface(
            u_odit_table,
            planning_interval=PLANNING_PERIOD,
            quota_consume_policy=OR_QUOTA_CONSUME_POLICY,
        )
        or_source = "generated (synthetic placeholder)"

    # 7. Metrics logger
    logger = MetricsLogger()

    print(f"  OR source  : {or_source}")
    return SimulationEngine(
        spatial=spatial,
        fleet=fleet,
        trip_gen=trip_gen,
        user_model=user_model,
        or_interface=or_interface,
        logger=logger,
        verbose=verbose,
    )


def main() -> None:
    W = 62
    SEP = "=" * W
    print(SEP)
    print("  EV Scooter Sharing Simulation".center(W))
    print(SEP)
    print(f"  Seed             : {RANDOM_SEED}")
    print(f"  Duration         : {SIM_DURATION:.0f} min")
    print(f"  Env mode         : sara_aligned")
    print(f"  Sara data dir    : {SARA_DATA_DIR}")
    print(f"  Planning period  : {PLANNING_PERIOD:.0f} min / period")
    print(
        f"  Rolling horizon  : {ROLLING_HORIZON:.0f} min  "
        f"({int(ROLLING_HORIZON / PLANNING_PERIOD)} periods)"
    )
    print(
        f"  Look-ahead       : {LOOKAHEAD_HORIZON:.0f} min  "
        f"({int(LOOKAHEAD_HORIZON / PLANNING_PERIOD)} periods)"
    )
    print(SEP)

    engine = build_simulation(verbose=True)
    total_periods = int(SIM_DURATION / PLANNING_PERIOD) + 1
    print(
        f"  U_odit table loaded  ({len(engine.or_interface)} entries  "
        f"across {total_periods} planning periods)"
    )
    print("  Demand profile: Sara data-driven (pickup + omega)")
    print("  Starting event loop ...\n")

    logger = engine.run(sim_duration=SIM_DURATION)

    logger.print_summary()


if __name__ == "__main__":
    main()
