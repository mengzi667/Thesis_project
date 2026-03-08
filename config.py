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

# ── Battery consumption ────────────────────────────────────────────────────────
BATTERY_CONSUMPTION_PER_KM: float = 0.02   # fraction lost per km travelled

# ── Trip generation (Poisson) ──────────────────────────────────────────────────
TRIP_ARRIVAL_RATE: float = 0.5       # mean trips per minute  (λ)
TRIP_DURATION_MEAN: float = 15.0     # minutes
TRIP_DURATION_STD: float = 5.0
TRIP_DISTANCE_MEAN: float = 2.0      # km
TRIP_DISTANCE_STD: float = 0.5

# ── User types ─────────────────────────────────────────────────────────────────
USER_TYPES: list = ["price_sensitive", "time_sensitive", "normal"]
USER_TYPE_WEIGHTS: list = [0.3, 0.3, 0.4]

# ── User choice model — scooter selection ──────────────────────────────────────
BETA_BATTERY: float = 1.5            # utility weight: battery level
BETA_WALKING: float = -0.5           # utility weight: walking distance (normalised)
BETA_PRICE: float = -1.0             # utility weight: price premium
OPT_OUT_UTILITY: float = 0.0         # baseline utility for not renting

# ── User choice model — relocation acceptance ──────────────────────────────────
BETA_INCENTIVE: float = 2.0          # utility weight: incentive amount
BETA_EXTRA_WALK: float = -1.5        # utility weight: extra walking (normalised)
BASE_RELOC_UTILITY: float = -0.5     # base utility of accepting relocation

# ── Relocation incentive ───────────────────────────────────────────────────────
RELOCATION_INCENTIVE: float = 1.5    # monetary units offered for relocation

# ── OR interface (placeholder) ─────────────────────────────────────────────────
RELOCATION_OFFER_PROB: float = 0.30  # probability a trip has a relocation offer

# ── Metrics snapshot ───────────────────────────────────────────────────────────
SNAPSHOT_INTERVAL: float = 60.0      # record inventory state every N minutes
