# config.py
# Central configuration for the shared e-scooter simulation.
# All key parameters are exposed here for easy modification and reproducibility.

# ── Simulation control ─────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
SIM_DURATION: float = 480.0          # minutes  (e.g. 8-hour operating window)

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
BATTERY_TRANSITION_CSV: str = "data/30sep-df_battery_decline_probs.csv"
# Policy for high -> inactive from CSV:
#   strict-paper: force P(high->inactive)=0 and renormalize high/low row
#   strict-data : keep CSV as-is
BATTERY_HIGH_TO_INACTIVE_POLICY: str = "strict-paper"
BATTERY_MIN_PRIMARY_N_FROM: int = 30
SIM_IS_WEEKEND: int = 0
SIM_START_HOUR: int = 0

# Legacy constant-rate Markov parameters kept for fallback compatibility.
PHI_HL: float = 0.18
PHI_LN: float = 0.22

# ── Trip generation (Poisson) ──────────────────────────────────────────────────
TRIP_ARRIVAL_RATE: float = 0.5       # mean trips per minute  (λ)
TRIP_DURATION_MEAN: float = 15.0     # minutes
TRIP_DURATION_STD: float = 5.0
TRIP_DISTANCE_MEAN: float = 2.0      # km
TRIP_DISTANCE_STD: float = 0.5

# ── User types ─────────────────────────────────────────────────────────────────
USER_TYPES: list = ["price_sensitive", "time_sensitive", "normal"]
USER_TYPE_WEIGHTS: list = [0.3, 0.3, 0.4]

# ── User choice model — relocation acceptance (MNL) ───────────────────────────────
# Scooter selection is deterministic (highest-battery first, PRD §14);
# these parameters apply ONLY to the relocation acceptance decision.
BETA_INCENTIVE: float = 2.0          # utility weight: incentive amount
BETA_EXTRA_WALK: float = -1.5        # utility weight: extra walking (normalised)
BASE_RELOC_UTILITY: float = -0.5     # base utility of accepting relocation
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

# ── Metrics snapshot (aligned to planning period) ─────────────────────────────
SNAPSHOT_INTERVAL: float = PLANNING_PERIOD   # snapshot at every planning boundary
