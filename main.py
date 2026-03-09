# main.py
# Entry point for the shared e-scooter simulation.
#
# Usage:
#   python main.py
#
# All parameters are controlled via config.py.

from __future__ import annotations

import random

from config import (
    RANDOM_SEED,
    SIM_DURATION,
    NUM_ZONES,
    FLEET_SIZE,
)
from spatial_system import SpatialSystem
from fleet_manager import FleetManager
from trip_generator import PoissonTripGenerator
from user_choice_model import UserChoiceModel
from or_interface import ORInterface, generate_synthetic_table
from metrics_logger import MetricsLogger
from simulation_engine import SimulationEngine


def build_simulation(seed: int = RANDOM_SEED) -> SimulationEngine:
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

    # 3. Trip generator (Poisson process)
    trip_gen = PoissonTripGenerator(zone_ids=spatial.all_zone_ids(), rng=rng)

    # 4. User choice model
    user_model = UserChoiceModel(rng=rng)

    # 5. OR interface — build a structured placeholder U_odit table once,
    #    then inject it into ORInterface as an external structured input.
    #    To switch to real OR outputs: replace generate_synthetic_table() with
    #    ORInterface.load_from_csv("or_outputs.csv") or load_from_json().
    u_odit_table = generate_synthetic_table(
        zone_ids=spatial.all_zone_ids(),
        sim_duration=SIM_DURATION,
        rng=rng,
    )
    or_interface = ORInterface(u_odit_table)

    # 6. Metrics logger
    logger = MetricsLogger()

    return SimulationEngine(
        spatial=spatial,
        fleet=fleet,
        trip_gen=trip_gen,
        user_model=user_model,
        or_interface=or_interface,
        logger=logger,
    )


def main() -> None:
    print(f"Building simulation  [seed={RANDOM_SEED}, duration={SIM_DURATION} min, "
          f"zones={NUM_ZONES}, fleet={FLEET_SIZE}]")
    engine = build_simulation()
    print(f"U_odit table loaded  [{len(engine.or_interface)} entries]")

    print("Running simulation ...")
    logger = engine.run(sim_duration=SIM_DURATION)

    logger.print_summary()


if __name__ == "__main__":
    main()
