# Shared E-Scooter Rebalancing Simulation Environment  
## Project Requirement Document

---

## 1. Project Overview

This project develops a simulation environment for a shared e-scooter system, intended to support the later evaluation of real-time incentive-based rebalancing strategies.

The simulator models the evolution of the system under:
- stochastic trip demand
- battery dynamics
- user behavioral responses
- relocation incentives

The simulator will later serve as the environment for reinforcement learning (RL). However, the current phase focuses only on building the simulation environment itself.

The upstream Operations Research (OR) model is not implemented in this project. Its outputs will be treated as external inputs and connected to the simulator through a predefined interface.

At the current stage, OR outputs will be represented by structured placeholder inputs until the real outputs become available.

---

## 2. Project Scope

### 2.1 Included in Current Phase

- spatial system representation (Sara H3 station data, zone IDs 1..N)
- scooter fleet simulation
- battery dynamics
- stochastic trip request generation
- user behavior modeling
- relocation recommendation interface (OR external-input injection via `U_odit`)
- event-driven simulation engine
- system state updates
- KPI collection

### 2.2 Not Included in Current Phase

- OR model implementation
- reinforcement learning agent
- reward function learning
- policy optimization
- training pipeline

---

## 3. Project Objective

The objective of the current phase is to build a modular, extensible, event-driven shared e-scooter simulation environment that can:

1. represent the system state at zone level and scooter level
2. simulate trip arrivals and trip execution over time
3. model battery-aware scooter availability
4. model user response to relocation incentives
5. accept OR model outputs through a dedicated interface
6. generate reliable metrics for later evaluation and RL integration

---

## 4. High-Level System Logic

The basic simulation logic is:

```text
trip arrival
→ check relocation opportunity from OR interface
→ determine whether a relocation recommendation exists
→ simulate user choice
→ execute trip
→ update scooter location and battery
→ update zone-level inventories
→ record metrics
```

The simulator is event-driven. The primary event is a trip request arrival.

---

## 5. System Architecture

The simulation system contains the following components:

```text
Trip Request Generator
        ↓
Simulation Engine
        ↓
Relocation Interface (OR placeholder)
        ↓
User Choice Model
        ↓
State Transition Engine
        ↓
System State Update
        ↓
Metrics & Logging
```

Future extension:

```text
RL Agent
   ↓
Decision Layer
   ↓
Simulation Environment
```

---

## 6. Spatial System

### 6.1 Zones

The system operates over a set of zones. Each zone represents a geo-fenced parking area where scooters can be picked up or dropped off.

Each zone must include the following attributes:

* `zone_id`
* `capacity`
* `coordinates` (optional in current phase)
* `neighbor_zones`

### 6.2 Zone Capacity Constraint

For each zone `i`, the following capacity constraint must hold:

```math
X_i^n + X_i^l + X_i^h \leq C_i
```

Where:

* `X_i^n` = number of inactive scooters
* `X_i^l` = number of low-battery scooters
* `X_i^h` = number of high-battery scooters
* `C_i` = zone capacity

### 6.3 Initial Spatial Scope

The first prototype should use a small spatial scope.

Suggested initial scale:

* 10–20 zones

The system must remain scalable to:

* 100+ zones in later stages

### 6.4 Zone Adjacency

Zones must store neighboring zones within an acceptable walking distance threshold.

Formally:

```math
N_i = \{ j \mid walking\_distance(i,j) \leq threshold \}
```

This adjacency structure is required for user-based relocation incentives.

---

## 7. Scooter Fleet Model

Each scooter is represented individually.

### 7.1 Scooter Attributes

Each scooter must include:

* `scooter_id`
* `current_zone`
* `battery_level`
* `battery_category`
* `status`
* `available_time`

### 7.2 Battery Categories

Scooters are categorized into three battery states:

| Category | Meaning       |
| -------- | ------------- |
| inactive | battery < 10% |
| low      | 10%–25%       |
| high     | >25%          |

Only low and high scooters are rentable.

### 7.3 Scooter Status

Each scooter should have a status such as:

* `idle`
* `in_trip`
* `unavailable`

This allows the simulator to distinguish scooters currently available for rental from scooters currently traveling.

---

## 8. System State Representation

At time `t`, the system state is defined at zone level as:

```math
X_i^t = (X_i^n, X_i^l, X_i^h)
```

for each zone `i`.

This captures:

* spatial availability
* battery composition
* user-visible supply

The simulator should also maintain vehicle-level state information so that zone-level state can always be derived from individual scooter records.

---

## 9. Trip Request Model

Trip requests are the main driver of system evolution.

Each trip request can be defined as:

```text
τ = (origin, destination, request_time, trip_duration, user_attributes)
```

### 9.1 Trip Attributes

Each request should include:

* `request_id`
* `origin_zone`
* `destination_zone`
* `request_time`
* `trip_duration`
* `trip_distance`
* `user_type`

### 9.2 Trip Arrival Process

Trip requests follow a stochastic process.

Initial implementation can use either:

* a Poisson arrival process
* historical demand replay
* a synthetic demand generator based on time interval and OD rates

The simulator must allow future replacement of the trip generation mechanism without rewriting the main simulation logic.

---

## 10. User Choice Model

User behavior influences two key processes:

1. ride participation (ride vs opt-out)
2. relocation acceptance (accept vs reject offer)

### 10.1 Scooter Selection

For the current simulation phase, scooter assignment follows system default logic (high-battery priority) and is not modeled as a separate user-level choice process.

Future extension may re-enable a full scooter-level choice model, but this is not required for Scenario 1.

### 10.2 Two-Layer Decision Structure (Sara-Aligned)

To stay consistent with Sara's implementation logic and Burghardt-based parameterization, user behavior must follow a two-layer structure:

Layer 1 (participation):

* decide `ride` or `opt_out`
* this layer determines whether the request enters trip execution at all

Layer 2 (relocation acceptance, conditional on ride):

* if relocation offer exists, decide `accept_offer` or `reject_offer`
* if no offer exists, keep original destination by default

Opt-out belongs to Layer 1 only and must not be mixed into Layer 2.

### 10.3 Layer 1 Modes (Scenario 1 Requirement)

Scenario 1 must support two interchangeable modes for Layer 1:

* `aggregated_prob`: use fixed Sara-style aggregate participation rates
* `realtime_choice`: compute participation probability from Sara-consistent utility mapping at request time

In Scenario 1, scooter assignment remains rule-based (highest-battery-first) regardless of Layer 1 mode.

Current implementation note:

* Layer 1 decision realization is stochastic only (Bernoulli(P(ride))) in both modes
* deterministic toggle is not used for Layer 1

### 10.4 Incentive Acceptance (Layer 2)

Layer 2 must only model acceptance conditional on `ride=1`.

For Sara-aligned execution, the baseline acceptance representation is a structured acceptability mapping over `(o,d,i)` (or equivalent OR-compatible structure), while keeping the interface replaceable when real OR outputs are connected.

If a probabilistic acceptance extension is used later, it must still remain conditional on Layer 1 participation.

Current implementation note:

* Layer 2 currently uses online binary utility choice (offer vs ase)
* acceptance realization mode is configurable (deterministic or stochastic)

### 10.5 Parameter Consistency Requirement

If Burghardt coefficients are used in `realtime_choice`, variable mapping must be documented before simulation runs:

```math
P(k)=\frac{\exp(V_k)}{\sum_{m\in\mathcal{C}}\exp(V_m)}
```

* variable definition mapping
* unit and scaling mapping
* sign consistency checks
* unsupported variables marked as placeholder or excluded with justification

---

## 11. OR Model Interface (Placeholder)

The simulator does not implement the OR model. Instead, it must accept OR outputs through a structured external interface.

### 11.1 Relocation Opportunity Structure

The OR output is represented as:

```text
U_odit
```

Where:

* `o` = origin zone
* `d` = original destination zone
* `i` = recommended destination zone
* `t` = planning interval or decision time indicator

Interpretation:

If a trip from `o` to `d` occurs at time `t`, the system may recommend the user to drop off at zone `i`.

### 11.2 Placeholder Requirement

At the current stage, the simulator must support structured placeholder OR outputs.

This means:

* the relocation interface must already exist
* the data structure must already be defined
* mock relocation opportunities can be generated synthetically
* the simulator logic must remain unchanged when real OR outputs become available later

### 11.3 Design Principle

The simulator must not depend on OR model internals.

The simulator only needs to know:

* whether a relocation opportunity exists
* what the recommended target zone is
* when the recommendation applies

---

## 12. Simulation Engine

The simulator operates using an event-driven architecture.

### 12.1 Primary Event

The main event is:

```text
trip arrival
```

### 12.2 Simulation Loop

For each trip request, the simulator performs the following steps:

#### Step 1: Trip Arrival

A trip request arrives:

```text
τ = (o, d, t, user)
```

#### Step 2: Search for Relocation Opportunity

The simulator checks the OR interface for a matching relocation opportunity:

```text
check U_odit
```

#### Step 3: Determine Candidate Recommendation

If a relocation opportunity exists, the simulator retrieves the candidate target zone `i`.

If no relocation opportunity exists, the trip follows the default execution path.

#### Step 4: Simulate User Decision

Layer 1 decision: simulate `ride` vs `opt_out`.

Possible outcomes:

* `opt_out`: stop request processing
* `ride`: continue to Layer 2

#### Step 5: Simulate Offer Acceptance (Conditional)

If `ride=1` and relocation recommendation exists, evaluate Layer 2 acceptance:

* `accept_offer`: use recommended target zone
* `reject_offer`: keep original destination

#### Step 6: Execute Trip

The scooter is assigned and the trip is executed from:

* origin → original destination
* or origin → recommended destination

#### Step 7: Update Battery State

Battery degradation is applied after trip completion.

Possible transitions:

* high → low
* low → inactive

#### Step 8: Update Zone Inventories

The simulator updates:

* origin inventory decreases
* destination or target inventory increases

#### Step 9: Record Metrics

Relevant system and behavioral metrics are recorded.

---

## 13. State Transition Logic

System state changes occur through three mechanisms:

### 13.1 Pickup Event

A scooter leaves the origin zone.

### 13.2 Drop-off Event

A scooter arrives at the destination zone or recommended target zone.

### 13.3 Battery Transition

After a trip, battery state may change:

* high → low
* low → inactive

The simulator must correctly update both:

* scooter-level states
* zone-level inventory states

---

## 14. Default Execution Logic

The simulator must support a default, non-RL execution path.

If no relocation opportunity exists, or if no incentive is accepted, the trip must be executed normally.

Default logic should include:

* selecting an available scooter at the origin zone
* prioritizing high-battery scooters over low-battery scooters
* marking the request as unserved if no rentable scooter is available

This default logic is required even before RL is introduced.

---

## 15. Metrics and Outputs

The simulator must collect the following outputs.

### 15.1 System Metrics

* total trip requests
* served trips
* unserved trips

### 15.2 Behavioral Metrics

* relocation offers
* relocation acceptance count
* relocation rejection count

### 15.3 Operational Metrics

* zone inventory distribution over time
* battery composition over time
* fleet utilization

### 15.4 Future Metrics Support

The simulator should preserve sufficient state and event information to support later computation of:

* expected demand loss (EDL)
* reward signals for RL
* service level metrics

---

## 16. Extensibility Requirements

The simulator must be extensible in three directions.

### 16.1 RL Integration

In the future, rule-based logic may be replaced by:

```python
agent.select_action(state)
```

Therefore, the simulator must expose:

* a state representation
* a decision point interface
* a clean separation between environment dynamics and decision logic

### 16.2 OR Integration

The current placeholder relocation interface must later be replaceable with real OR outputs without changing the simulation loop.

### 16.3 Scalability

The simulator should support future expansion to:

* 100+ zones
* large scooter fleets
* large trip streams

---

## 17. Software Design Principles

The implementation must follow these principles:

### 17.1 Modular Design

Each major function should be implemented in an independent module.

### 17.2 Extensibility

Future RL and OR integration should require minimal code changes.

### 17.3 Reproducibility

Simulation runs must be reproducible using random seeds.

### 17.4 Configurability

Key parameters must be configurable, including:

* zone count
* fleet size
* arrival rates
* battery thresholds
* walking threshold
* incentive parameters

---

## 18. Recommended Modules

The project should at minimum include the following logical modules:

* `spatial_system`
* `fleet_manager`
* `trip_generator`
* `user_choice_model`
* `or_interface`
* `simulation_engine`
* `metrics_logger`

These can later be mapped into Python files or classes.

---

## 19. Current Development Tasks

The current development phase should complete the following tasks:

1. define the spatial zone structure
2. define scooter-level and zone-level states
3. implement the trip request generator
4. implement the user choice model
5. implement the OR placeholder interface
6. implement the event-driven simulation loop
7. implement state update logic
8. implement metrics collection

The following are explicitly not required in this phase:

* RL agent
* reward design
* policy learning
* OR model implementation

---

## 20. RL Scenario Design Requirements

To support controlled experimentation before full RL deployment, the project must support two explicit RL scenarios in later phases.

### 20.1 Scenario 1 (Binary Offer, Fixed Scooter Rule)

Decision logic:

* scooter assignment remains rule-based (highest battery priority at origin)
* RL action is binary only: offer incentive (`1`) or no offer (`0`)
* incentive amount is fixed
* Layer 1: user chooses `ride` vs `opt_out` using configurable mode:
* `aggregated_prob` (fixed aggregate probability)
* `realtime_choice` (Sara-consistent utility-based participation probability)
* Layer 2 (only when `ride=1`): user chooses `accept_offer` vs `reject_offer` if offer exists

Reward intent:

* reward is based on scenario-specific EDL marginal improvement
* in this scenario, reward can focus on destination-side impact (`d` and redirected target zone)
* incentive cost is deducted when an offer is accepted

### 20.2 Scenario 2 (Joint Offer + Battery-Level Decision)

Decision logic:

* RL action may include both offer decision and battery-level related operational choice
* this scenario may use a broader, joint action space than Scenario 1
* the simulator must preserve compatibility with OR output while allowing RL exploration beyond pure binary offer/no-offer
* if Scenario 2 uses a joint one-shot user choice set, it is treated as an extension track and not the baseline Sara-aligned track

Reward intent:

* reward remains EDL-centered (same core objective as Scenario 1)
* EDL evaluation scope must include more impacted zones (at minimum origin `o`, destination `d`, and redirected zone)
* incentive cost is deducted when accepted

---

## 21. Required Corrections to Current Logic

The following three corrections are mandatory for consistency with the current research design.

### 21.1 Correct User-Choice Probability Logic (with Opt-Out)

The user-choice logic must be corrected to the two-layer baseline:

* Layer 1 computes participation (`ride` vs `opt_out`)
* Layer 2 computes offer acceptance (`accept_offer` vs `reject_offer`) only if `ride=1`
* `opt_out` must not be part of Layer 2 choice

Implementation requirement:

* Layer 1 uses stochastic realization (Bernoulli(P(ride)))
* Layer 2 may use configured mode (deterministic or stochastic)
* Scenario 1 must keep scooter assignment as highest-battery-first system rule
* the `aggregated_prob` participation mode must be available as baseline Sara-aligned mode

### 21.2 Fixed Incentive Amount

Incentive must be fixed for controlled experiments:

* default fixed value: `1 EUR`
* if OR placeholder input contains different incentive values, the simulation execution layer must enforce the fixed experimental value

### 21.3 Battery Dynamics via Markov State Transition

Battery evolution must follow Sara-style Markov transitions instead of linear deterministic drain:

* state categories: `high`, `low`, `inactive`
* allowed transitions per trip completion:
* `high -> low` with transition probability `phi_hl`
* `low -> inactive` with transition probability `phi_ln`
* no direct `high -> inactive` transition in the baseline Markov setting

The simulator must keep transition parameters configurable for later calibration with empirical data.

---

## 22. Battery Markov Transition Upgrade (Sara CSV-Based)

The battery transition module must be upgraded from constant transition rates to a time-conditioned transition table based on Sara-provided data.

### 22.1 Data Source and Fields

Input file:

* `30sep-df_battery_decline_probs.csv`

Required columns:

* `is_weekend` (`0` weekday, `1` weekend)
* `hour` (`0-23`)
* `init_power_class` (`high` / `low` / `inactive`)
* `end_power_class` (`high` / `low` / `inactive`)
* `n` (transition count)
* `n_from` (total count from `init_power_class` under same condition)
* `p` (`n / n_from`)

### 22.2 Transition Formula

At trip completion, the simulator must apply:

```math
P(S_{t+}=j \mid S_t=i,\; is\_weekend=w,\; hour=h)=p_{w,h,i,j}
```

where `(w, h)` is determined from trip completion time.

### 22.3 Policy for `high -> inactive`

Sara's written description states no direct `high -> inactive` transition, while CSV may contain small non-zero values.

The project must explicitly choose one policy:

* strict-paper policy (recommended): set `P(high -> inactive)=0` and renormalize remaining `high` row probabilities
* strict-data policy: keep CSV probabilities as provided

This choice must be fixed and documented before RL experiments.

### 22.4 Sparsity Handling and Fallback

Because some `(is_weekend, hour, init_power_class)` groups have low `n_from`, the simulator must support fallback/smoothing:

* primary lookup: `(is_weekend, hour, init_power_class)`
* fallback 1: `(is_weekend, init_power_class)` aggregated across hours
* fallback 2: global `(init_power_class)` aggregated across all rows

Optional smoothing by sample size is encouraged.

### 22.5 Integration Requirements

The implementation must:

* replace fixed `phi_hl` / `phi_ln` execution with table-based categorical sampling
* keep state space as `high`, `low`, `inactive`
* preserve reproducibility via seeded random generator
* preserve backward-compatible interface for simulation loop and logging

### 22.6 Validation and Acceptance Criteria

The upgraded module is accepted only if:

* probabilities for each condition-row sum to 1 (after policy processing)
* no undefined condition causes runtime failure (fallback must work)
* observed simulated transitions remain within defined state space
* chosen `high -> inactive` policy is consistently enforced
* runs are reproducible under fixed random seed

---

## 23. One-Sentence Summary

This project builds a modular, extensible, event-driven shared e-scooter simulation environment that models trip demand, battery dynamics, and user behavior, while interfacing with upstream OR relocation recommendations and supporting future RL-based real-time decision making.

---

## 24. Current Code Baseline (Synced to Repository)

Current repository structure:

* `main.py` and `config.py`: simulation entry and global configuration
* `simulation/`: environment dynamics (`spatial_system`, `fleet_manager`, `battery_transition`, `trip_generator`, `simulation_engine`, `metrics_logger`, `user_choice_model`, `sara_environment`)
* `or_model/`: OR input schema, loaders, adapter, and synthetic OR output generator
* `docs/`: project requirement document
* `data/input/`: active simulation input files (currently `u_odit.csv`)
* `data/generated/`: generated OR artifacts and temporary converted files
* `results/`: simulation and OR run outputs
* `sara_repo/`: external reference implementation and raw Sara datasets

Path conventions currently used in code:

* `SARA_DATA_DIR = "sara_repo/data"`
* `BATTERY_TRANSITION_CSV = "sara_repo/data/30sep-df_battery_decline_probs.csv"`
* `OR_INPUT_PATH = "data/input/u_odit.csv"`
* Sara-output conversion target: `data/generated/u_odit_from_sara.csv`
