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
    PLANNING_PERIOD,
    RANDOM_SEED,
    ROLLING_HORIZON,
    LOOKAHEAD_HORIZON,
    SIM_DURATION,
    NUM_ZONES,
    FLEET_SIZE,
    TRIP_ARRIVAL_RATE,
)
from simulation.spatial_system import SpatialSystem
from simulation.fleet_manager import FleetManager
from simulation.trip_generator import (
    HeterogeneousTripGenerator,
    build_synthetic_demand_profile,
)
from simulation.user_choice_model import UserChoiceModel
from or_model.or_interface import ORInterface, build_demand_informed_table

# OR data file produced by: python -m or_model.generate_or_output
OR_DATA_FILE = "data/or_output.csv"
from simulation.metrics_logger import MetricsLogger
from simulation.simulation_engine import SimulationEngine


def build_simulation(seed: int = RANDOM_SEED, verbose: bool = True) -> SimulationEngine:
    """
    Wire up all components and return a ready-to-run SimulationEngine.
    A single Random instance is shared across modules for reproducibility.
    """
    rng = random.Random(seed)

    # 1. Spatial system — synthetic 10-zone grid
    spatial = SpatialSystem.create_grid(num_zones=NUM_ZONES)

    # 2. Fleet — scooters distributed across zones with random battery levels
    fleet = FleetManager(spatial)
    fleet.initialize_fleet(fleet_size=FLEET_SIZE, rng=rng)

    # 3. Demand profile — zone-time heterogeneous (generator / neutral / attractor zones)
    #    To use real historical data: replace build_synthetic_demand_profile() with
    #    a loader that reads per-zone per-slot rates from a CSV or JSON file.
    demand_profile = build_synthetic_demand_profile(
        zone_ids=spatial.all_zone_ids(),
        sim_duration=SIM_DURATION,
        planning_period=PLANNING_PERIOD,
        total_rate=TRIP_ARRIVAL_RATE,
        rng=rng,
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
    #    If data/or_output.csv exists (run `python -m or_model.generate_or_output` first),
    #    load it directly.  Otherwise fall back to the demand-informed synthetic table.
    if os.path.isfile(OR_DATA_FILE):
        or_interface = ORInterface.load_from_csv(OR_DATA_FILE, planning_interval=PLANNING_PERIOD)
        or_source = f"loaded from {OR_DATA_FILE}"
    else:
        u_odit_table = build_demand_informed_table(
            demand_profile=demand_profile,
            zone_ids=spatial.all_zone_ids(),
            sim_duration=SIM_DURATION,
            planning_period=PLANNING_PERIOD,
            rng=rng,
        )
        or_interface = ORInterface(u_odit_table, planning_interval=PLANNING_PERIOD)
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
    print(f"  Zones            : {NUM_ZONES}")
    print(f"  Fleet            : {FLEET_SIZE} scooters")
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
    print("  Demand profile: zone-time heterogeneous  "
          f"(generator / neutral / attractor zones)")
    print("  Starting event loop ...\n")

    logger = engine.run(sim_duration=SIM_DURATION)

    logger.print_summary()


if __name__ == "__main__":
    main()
