# config.py
# Central configuration for the shared e-scooter simulation.
# All key parameters are exposed here for easy modification and reproducibility.

# ── Simulation control ─────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
SIM_DURATION: float = 840.0          # minutes  (Sara: 06:00–20:00)

# ── Sara-aligned environment mode ────────────────────────────────────────────
# When enabled, spatial system + demand + initial fleet state are loaded from
# Sara data files instead of synthetic grid/profile.
SARA_DATA_DIR: str = "sara_repo/data"
SARA_IS_WEEKEND: int = 1
SARA_SLOT0_HOUR: int = 6
SARA_ZONE_CAPACITY: int = 10
SARA_INIT_INVENTORY_CSV: str = ""   # optional CSV (station,n,l,h)
SARA_INIT_UNIFORM_N: int = 0
SARA_INIT_UNIFORM_L: int = 4
SARA_INIT_UNIFORM_H: int = 5

# ── Spatial system ─────────────────────────────────────────────────────────────
NUM_ZONES: int = 10                  # initial prototype scale (10-20 recommended)
ZONE_CAPACITY: int = 20              # default scooter capacity per zone
GRID_SPACING: float = 400.0          # meters between zone-centre grid points
WALKING_THRESHOLD: float = 500.0     # meters — defines zone-adjacency (N_i)

# ── Fleet ──────────────────────────────────────────────────────────────────────
FLEET_SIZE: int = 50
INITIAL_BATTERY_MEAN: float = 0.65   # mean initial battery fraction
INITIAL_BATTERY_STD: float = 0.20    # std-dev of initial battery fraction

# ── Battery thresholds ─────────────────────────────────────────────────────────
BATTERY_INACTIVE_THRESHOLD: float = 0.10   # < 10%  → inactive  (not rentable)
BATTERY_LOW_THRESHOLD: float = 0.25        # 10-25% → low       (rentable)
                                           # > 25%  → high      (rentable)

# ── Battery Markov transition (Sara CSV-based) ───────────────────────────────
BATTERY_TRANSITION_CSV: str = "sara_repo/data/30sep-df_battery_decline_probs.csv"
# Policy for high -> inactive from CSV:
#   strict-paper: force P(high->inactive)=0 and renormalize high/low row
#   strict-data : keep CSV as-is
BATTERY_HIGH_TO_INACTIVE_POLICY: str = "strict-paper"
BATTERY_MIN_PRIMARY_N_FROM: int = 30
SIM_IS_WEEKEND: int = 1
SIM_START_HOUR: int = 6

# Legacy constant-rate Markov parameters kept for fallback compatibility.
PHI_HL: float = 0.18
PHI_LN: float = 0.22

# ── Trip generation (Poisson) ──────────────────────────────────────────────────
TRIP_ARRIVAL_RATE: float = 0.5       # mean trips per minute  (λ)
TRIP_DURATION_MEAN: float = 15.0     # minutes
TRIP_DURATION_STD: float = 5.0
TRIP_DISTANCE_MEAN: float = 2.0      # km
TRIP_DISTANCE_STD: float = 0.5

# ── Trip source switch ─────────────────────────────────────────────────────────
#   omega_od     : omega_h OD-slot stochastic generation (recommended for OR/RL)
#   sara_profile : Sara pickup+omega demand-driven stochastic generation
#   replay       : deterministic replay from a trip file (recommended for Sara alignment)
#   poisson      : homogeneous Poisson baseline
TRIP_SOURCE: str = "omega_od"
# Replay input file (CSV/XLSX), used only when TRIP_SOURCE == "replay".
TRIP_REPLAY_PATH: str = "data/input/sara_trip_replay.csv"
TRIP_REPLAY_SHEET: str = "Sheet1"    # for .xlsx
TRIP_REPLAY_SLOT_MINUTES: float = 15.0
TRIP_REPLAY_TIME_OFFSET_MIN: float = 0.0

# ── User types ─────────────────────────────────────────────────────────────────
USER_TYPES: list = ["price_sensitive", "time_sensitive", "normal"]
USER_TYPE_WEIGHTS: list = [0.3, 0.3, 0.4]

# ── Scenario 1 two-layer user behavior (Sara-aligned) ───────────────────────
# Layer 1 mode:
#   aggregated_prob : fixed aggregate participation probability (Sara-style)
#   realtime_choice : compute participation via Sara-consistent utility terms
FIRST_LAYER_MODE: str = "aggregated_prob"
# Layer 1 realization is fixed to stochastic Bernoulli(P(ride)) to align with
# Sara-style probabilistic participation handling.
SARA_PROB_H: float = 0.70
SARA_PROB_L: float = 0.18
SARA_PROB_OUT: float = 0.12
# Sara first-layer real-time choice defaults (compute_probs_for_class-consistent)
SARA_FIRST_LAYER_WALK_MIN: float = 2.0
SARA_FIRST_LAYER_UNLOCK_FEE: float = 1.0
SARA_FIRST_LAYER_RIDE_FEE_TERM: float = 0.30
SARA_FIRST_LAYER_PCT_HIGH: float = 50.0
SARA_FIRST_LAYER_PCT_LOW: float = 25.0

# ── Sara choice-model parameters (mean coefficients) ─────────────────────────
# Adopted from Sara/Burghardt setting (mean values) for labeled utilities.
SARA_BETA_ES: float = 10.514
SARA_BETA_WALK: float = -0.342
SARA_BETA_UNLOCK: float = -1.419
SARA_BETA_RIDE: float = -25.147
SARA_BETA_BATT: float = 0.27
SARA_BETA_TYPE: float = -1.02
SARA_BETA_PREV: float = -1.79
SARA_BETA_BIKE: float = -2.21
SARA_BETA_INCOME: float = -1.18
SARA_BETA_ALONE: float = 1.21
SARA_BETA_SHARED: float = -0.86
SARA_ETA_ATT: float = 0.82
SARA_ETA_RANGE: float = -0.58
SARA_RANGE_PER_PCT_KM: float = 0.6
SARA_WALK_SPEED_KMH: float = 4.8
SARA_RIDE_SPEED_KMH: float = 25.0
SARA_RIDE_PRICE_PER_MIN: float = 0.30

# Default labeled-alternative attributes used in simulation.
SARA_USER_VEHICLE_TYPE_25: int = 1
SARA_USER_PREVIOUS_USE: int = 1
SARA_USER_BIKE: int = 0
SARA_USER_INCOME_LOW: int = 0
SARA_USER_LIVING_ALONE: int = 0
SARA_USER_LIVING_SHARED: int = 0
SARA_USER_ATTITUDE: float = 0.0
SARA_USER_RANGE_ANXIETY: float = 0.0

# Acceptance model remains configurable for deterministic/stochastic realization.
# Acceptance decision mode:
#   deterministic -> accept iff P_offer > P_base
#   stochastic    -> draw z_t ~ Bernoulli(P_offer)
RELOCATION_ACCEPTANCE_MODE: str = "deterministic"

# ── Relocation incentive ───────────────────────────────────────────────────────
RELOCATION_INCENTIVE: float = 1.0    # EUR, fixed for controlled experiment

# ── Planning horizon ──────────────────────────────────────────────────────────
PLANNING_PERIOD: float   = 15.0      # minutes — OR slot width & re-planning cadence
ROLLING_HORIZON: float   = 60.0     # minutes — rolling planning window  (4 periods)
LOOKAHEAD_HORIZON: float = 240.0     # minutes — demand look-ahead         (16 periods)

# ── OR interface (placeholder) ─────────────────────────────────────────────────
RELOCATION_OFFER_PROB: float = 0.30  # used only by the legacy random synthetic table

# ── OR static injection (Sara Level 1 integration) ───────────────────────────
# Input format and source
OR_INPUT_FORMAT: str = "csv"  # csv | json
OR_INPUT_PATH: str = "data/input/u_odit.csv"

# Experimental incentive policy
OR_FORCE_FIXED_INCENTIVE: bool = True
OR_FIXED_INCENTIVE_EUR: float = 1.0

# Quota consumption policy
#   consume_on_accept: consume quota only when user accepts offer
#   consume_on_offer : consume quota immediately when offer is made
OR_QUOTA_CONSUME_POLICY: str = "consume_on_accept"

# Time-slot audit settings
OR_SLOT_MINUTES: int = 15
OR_TIME_BASE: str = "06:00->slot0"

# Data validation behavior
#   strict : raise error on dirty rows
#   lenient: skip dirty rows with warning
OR_VALIDATION_MODE: str = "strict"

# ── Metrics snapshot (aligned to planning period) ─────────────────────────────
SNAPSHOT_INTERVAL: float = PLANNING_PERIOD   # snapshot at every planning boundary
