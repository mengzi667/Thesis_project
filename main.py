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
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from config import (
    TRIP_SOURCE,
    TRIP_REPLAY_PATH,
    TRIP_REPLAY_SHEET,
    TRIP_REPLAY_SLOT_MINUTES,
    TRIP_REPLAY_TIME_OFFSET_MIN,
    TRIP_ARRIVAL_RATE,
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
    OMEGA_GLOBAL_SCALE,
    OMEGA_WINDOW_START_SLOT,
    OMEGA_WINDOW_END_SLOT,
    OMEGA_WINDOW_SCALE,
    OMEGA_OD_TARGET_SCALE,
    OMEGA_ARRIVAL_DIST,
    OMEGA_NB_PHI_MODE,
    OMEGA_NB_PHI_GLOBAL,
    OMEGA_NB_PHI_CSV,
    OMEGA_NB_PHI_MIN,
    OMEGA_NB_PHI_MAX,
)
from simulation.fleet_manager import FleetManager
from simulation.trip_generator import (
    PoissonTripGenerator,
    HeterogeneousTripGenerator,
    OmegaODTripGenerator,
    ReplayTripGenerator,
)
from simulation.sara_environment import (
    build_sara_demand_profile,
    build_sara_minute_rt_matrix,
    build_sara_omega_slot_expected,
    build_sara_spatial_system,
    build_uniform_zone_state,
    load_zone_state_from_csv,
)
from simulation.user_choice_model import UserChoiceModel
from or_model.or_interface import ORInterface, build_demand_informed_table
from or_model.sara_adapter import convert_sara_output_to_uodit

from simulation.metrics_logger import MetricsLogger
from simulation.simulation_engine import SimulationEngine
from simulation.edl_markov import SaraMarkovEDL


def _slot_to_hour(slot: int, slot0_hour: int, planning_period: float) -> int:
    return int((slot0_hour + int((int(slot) * float(planning_period)) // 60)) % 24)


def _load_nb_phi_table(phi_csv: str) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    if not phi_csv or (not os.path.isfile(phi_csv)):
        return out
    try:
        df = pd.read_csv(phi_csv)
        required = {"is_weekend", "hour", "phi"}
        if not required.issubset(df.columns):
            return out
        for _, row in df.iterrows():
            k = (int(row["is_weekend"]), int(row["hour"]))
            out[k] = float(row["phi"])
    except Exception:
        return {}
    return out


def _load_target_ods_from_uodit(
    uodit_path: str,
    window_start_slot: int,
    window_end_slot: int,
) -> Set[Tuple[int, int]]:
    if not os.path.isfile(uodit_path):
        return set()
    try:
        df = pd.read_csv(uodit_path)
        req = {"origin", "original_dest", "time_slot", "quota"}
        if not req.issubset(df.columns):
            return set()
        sub = df[
            (df["time_slot"].astype(int) >= int(window_start_slot))
            & (df["time_slot"].astype(int) <= int(window_end_slot))
            & (df["quota"].astype(float) > 0)
        ]
        return {
            (int(r.origin), int(r.original_dest))
            for r in sub[["origin", "original_dest"]].drop_duplicates().itertuples(index=False)
        }
    except Exception:
        return set()


def _apply_omega_sampling_profile(
    od_slot_expected: Dict[Tuple[int, int], Dict[int, float]],
    global_scale: float,
    window_start_slot: int,
    window_end_slot: int,
    window_scale: float,
    od_target_scale: float,
    target_ods: Set[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[int, float]]:
    out: Dict[Tuple[int, int], Dict[int, float]] = {}
    ws = int(window_start_slot)
    we = int(window_end_slot)
    for (o, d), slot_map in od_slot_expected.items():
        new_map: Dict[int, float] = {}
        for slot, lam in slot_map.items():
            v = float(lam) * float(global_scale)
            if ws <= int(slot) <= we:
                v *= float(window_scale)
                if (int(o), int(d)) in target_ods:
                    v *= float(od_target_scale)
            new_map[int(slot)] = max(0.0, v)
        out[(int(o), int(d))] = new_map
    return out


def build_simulation(
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    omega_global_scale: Optional[float] = None,
    omega_window_start_slot: Optional[int] = None,
    omega_window_end_slot: Optional[int] = None,
    omega_window_scale: Optional[float] = None,
    omega_od_target_scale: Optional[float] = None,
    omega_arrival_dist: Optional[str] = None,
    omega_nb_phi_mode: Optional[str] = None,
    omega_nb_phi_global: Optional[float] = None,
    omega_nb_phi_csv: Optional[str] = None,
    omega_nb_phi_min: Optional[float] = None,
    omega_nb_phi_max: Optional[float] = None,
    or_input_path_override: Optional[str] = None,
) -> SimulationEngine:
    """
    Wire up all components and return a ready-to-run SimulationEngine.
    A single Random instance is shared across modules for reproducibility.
    """
    rng = random.Random(seed)

    # 0. Optional runtime overrides (used by RL train/eval scripts)
    or_input_path = str(or_input_path_override or OR_INPUT_PATH)
    g_scale = float(OMEGA_GLOBAL_SCALE if omega_global_scale is None else omega_global_scale)
    ws = int(OMEGA_WINDOW_START_SLOT if omega_window_start_slot is None else omega_window_start_slot)
    we = int(OMEGA_WINDOW_END_SLOT if omega_window_end_slot is None else omega_window_end_slot)
    w_scale = float(OMEGA_WINDOW_SCALE if omega_window_scale is None else omega_window_scale)
    od_scale = float(OMEGA_OD_TARGET_SCALE if omega_od_target_scale is None else omega_od_target_scale)
    arrival_dist = str(OMEGA_ARRIVAL_DIST if omega_arrival_dist is None else omega_arrival_dist).strip().lower()
    phi_mode = str(OMEGA_NB_PHI_MODE if omega_nb_phi_mode is None else omega_nb_phi_mode).strip().lower()
    phi_global = float(OMEGA_NB_PHI_GLOBAL if omega_nb_phi_global is None else omega_nb_phi_global)
    phi_csv = str(OMEGA_NB_PHI_CSV if omega_nb_phi_csv is None else omega_nb_phi_csv)
    phi_min = float(OMEGA_NB_PHI_MIN if omega_nb_phi_min is None else omega_nb_phi_min)
    phi_max = float(OMEGA_NB_PHI_MAX if omega_nb_phi_max is None else omega_nb_phi_max)

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
    minute_rt_matrix = build_sara_minute_rt_matrix(SARA_DATA_DIR)
    # Sara-aligned full Markov EDL model (for RL reward EDL term).
    edl_model = SaraMarkovEDL.from_sara_csv(
        data_dir=SARA_DATA_DIR,
        station_ids=zone_ids,
        capacity=SARA_ZONE_CAPACITY,
        is_weekend=SARA_IS_WEEKEND,
        dt=PLANNING_PERIOD / 60.0,
        slots_per_hour=max(1, int(round(60.0 / PLANNING_PERIOD))),
    )

    # 4. Trip generator (switchable source)
    source = TRIP_SOURCE.strip().lower()
    if source == "replay":
        trip_gen = ReplayTripGenerator(
            replay_path=TRIP_REPLAY_PATH,
            sim_duration=SIM_DURATION,
            rng=rng,
            sheet_name=TRIP_REPLAY_SHEET,
            slot_minutes=TRIP_REPLAY_SLOT_MINUTES,
            time_offset_min=TRIP_REPLAY_TIME_OFFSET_MIN,
        )
    elif source in {"omega", "omega_od"}:
        od_slot_expected_raw = build_sara_omega_slot_expected(
            data_dir=SARA_DATA_DIR,
            zone_ids=zone_ids,
            sim_duration=SIM_DURATION,
            planning_period=PLANNING_PERIOD,
            is_weekend=SARA_IS_WEEKEND,
            slot0_hour=SARA_SLOT0_HOUR,
        )
        target_ods = _load_target_ods_from_uodit(
            uodit_path=or_input_path,
            window_start_slot=ws,
            window_end_slot=we,
        )
        od_slot_expected = _apply_omega_sampling_profile(
            od_slot_expected=od_slot_expected_raw,
            global_scale=g_scale,
            window_start_slot=ws,
            window_end_slot=we,
            window_scale=w_scale,
            od_target_scale=od_scale,
            target_ods=target_ods,
        )
        phi_table = _load_nb_phi_table(phi_csv)
        phi_values = []
        for hour in range(24):
            if phi_mode == "by_hour_weektype":
                k = (int(SARA_IS_WEEKEND), int(hour))
                phi_values.append(float(phi_table.get(k, phi_global)))
            elif phi_mode == "by_hour":
                k1 = (-1, int(hour))
                k2 = (int(SARA_IS_WEEKEND), int(hour))
                phi_values.append(float(phi_table.get(k1, phi_table.get(k2, phi_global))))
            else:
                phi_values.append(float(phi_global))
        phi_series = pd.Series(phi_values, dtype=float)
        trip_gen = OmegaODTripGenerator(
            od_slot_expected=od_slot_expected,
            planning_period=PLANNING_PERIOD,
            rng=rng,
            arrival_dist=arrival_dist,
            phi_mode=phi_mode,
            phi_global=phi_global,
            phi_min=phi_min,
            phi_max=phi_max,
            phi_table=phi_table,
            slot_to_hour_fn=lambda s: _slot_to_hour(int(s), SARA_SLOT0_HOUR, PLANNING_PERIOD),
            is_weekend=int(SARA_IS_WEEKEND),
        )
        phi_source = phi_csv if phi_table else "global_fallback"
        if verbose:
            print(
                "  Trip arrival   : "
                f"{arrival_dist} | phi_mode={phi_mode} | phi_source={phi_source} | "
                f"phi(min/p50/max)=({phi_series.min():.3f}/{phi_series.median():.3f}/{phi_series.max():.3f})"
            )
    elif source == "poisson":
        trip_gen = PoissonTripGenerator(
            zone_ids=spatial.all_zone_ids(),
            arrival_rate=TRIP_ARRIVAL_RATE,
            rng=rng,
        )
    else:
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

    if os.path.isfile(or_input_path):
        try:
            if OR_INPUT_FORMAT.lower() == "json":
                or_interface = ORInterface.load_from_json(
                    or_input_path,
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
                    or_input_path,
                    planning_interval=PLANNING_PERIOD,
                    quota_consume_policy=OR_QUOTA_CONSUME_POLICY,
                    force_fixed_incentive=OR_FORCE_FIXED_INCENTIVE,
                    fixed_incentive_eur=OR_FIXED_INCENTIVE_EUR,
                    validation_mode=OR_VALIDATION_MODE,
                    valid_zone_ids=zone_id_set,
                    max_time_slot=max_time_slot,
                )
            or_source = f"loaded from configured U_odit: {or_input_path}"
        except Exception:
            translated_path = "data/generated/u_odit_from_sara.csv"
            convert_sara_output_to_uodit(
                input_path=or_input_path,
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

    if verbose:
        print(f"  OR source  : {or_source}")
    return SimulationEngine(
        spatial=spatial,
        fleet=fleet,
        trip_gen=trip_gen,
        user_model=user_model,
        or_interface=or_interface,
        logger=logger,
        demand_profile=demand_profile,
        episode_minutes=SIM_DURATION,
        edl_model=edl_model,
        ride_time_minutes=minute_rt_matrix,
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
    print(f"  Trip source      : {TRIP_SOURCE}")
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
