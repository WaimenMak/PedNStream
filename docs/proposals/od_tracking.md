# Proposal: OD-labelled flow tracking

Status: prototype on branch `od-tracking` (from `optimize_path` @ `bf9b02d`), opt-in, default off.

## Problem
At every junction the node model needs turning fractions:

    tf(up → down) = Σ_od  P(od | up) · P(down | up, od)

`P(down | up, od)` is the route choice (logit) and is fine. `P(od | up)`, "which OD pair is
this person on the incoming link?", is estimated in `PathFinder.update_turning_fractions` from
the **OD weights** of every pair whose routes pass that link. It does not use how many people
each origin actually sends, or who is actually on the link. Two consequences:

1. **An origin with no demand still steers other people.** In a 7-node test network (Gate A → Plaza,
   Gate B → Hall, exits X and Y), Gate A sends everyone to X and Gate B sends nobody, but has weights
   1:1. Corridor 2→4 is on Gate A's detour to X and on Gate B's route to Y, so the junction assumes
   half of 2→4 is going to Y. **Only 69.7% of Gate A reaches X** (asked: 100%).
2. **Realised OD totals drift from the weights** whenever OD pairs share corridors. In that network's
   "crossing" pattern (A → X, B → Y, equal demand), 44.9% arrive at X instead of 50%.

## Proposed change
Keep the node model on **totals** (sending/receiving flow, capacity, counter-flow and densities are
unchanged) and add **labels** to those totals:

| where | what |
|---|---|
| each link | `counts[link][k]`: people on the link belonging to OD pair k |
| origins | new people are labelled with P(d \| o) from the OD weights at entry |
| junctions | `P(od \| up)` = the tracked mix on the incoming link (replaces the weight estimate); the flow on each turn is split over OD pairs ∝ mix · P(down \| up, od) |
| destinations | each OD pair's people are removed and counted as arrived |

Order per step: turning fractions (from the mix at t−1) → node model (unchanged) → label each turn's
flow → apply all label moves → link states. Mixing is **fully mixed**: people leaving a link carry its
current mix (not strictly FIFO; see open questions).

### Code
- `ltm/od_tracker.py` (new): `ODTracker` with the counts, origin labelling, `record_node`, `commit`
  and `od_table()` (entered / arrived / inside per OD pair).
- `ltm/node.py`: `RegularNode.solve` keeps the per-turn flows it already computes (`edge_flows`).
- `ltm/path_finder.py`: `update_turning_fractions` uses the tracked mix when tracking is on
  (falls back to the weight estimate for an empty link).
- `ltm/network.py`: creates the tracker and calls it in `network_loading`.
- `utils/config.py`: `simulation.od_tracking` (default `false`).

Enable with
```yaml
simulation:
  od_tracking: true
```
or `gen.config["params"]["od_tracking"] = True` before `create_network`.

## Results
Same network, origin A only active unless stated (`tests/test_od_tracking.py`):

| case | asked X | today | with tracking |
|---|---|---|---|
| A 1:0, B empty with weights 1:1 | 100% | 69.7% | **100.0%** |
| A 3:1, B empty | 75% | 69.2% | **75.1%** |
| crossing, both gates | 50% | 44.9% | **49.9%** |
| parallel, both gates | 50% | 50.2% | 50.2% |
| baseline 1:1, both gates | 50% | 48.0% | **50.1%** |

Labelled counts equal each link's `num_pedestrians` to round-off on every network tried, and no flow
is left unlabelled. The integer rounding in `solve` no longer biases the split either: people held back
by rounding keep their label and leave in a later step.

### Cost (`network_loading` time)
| network | nodes | links | OD pairs | off | on | slowdown |
|---|---|---|---|---|---|---|
| butterfly_scA | 9 | 20 | 1 | 0.23 s | 0.27 s | ×1.13 |
| nine_intersections | 9 | 24 | 4 | 0.40 s | 0.56 s | ×1.39 |
| 45_intersections | 49 | 168 | 2 | 1.61 s | 1.97 s | ×1.22 |
| delft | 298 | 818 | 140 | 12.7 s | 15.1 s | ×1.19 |

Memory: one float per link per OD pair (current state) plus entered/arrived per OD pair over time.

### Tests
All existing tests pass with tracking off (the default), including the regression and
turn-probability equivalence tests. `test_network_env_generator::test_create_network_overwrite_dir`
errors with and without this change (it needs `pytest-mock`).

## Behaviour change to be aware of
Labels are fixed at entry. With **time-varying OD weights**, people already inside keep the destination
they had when they entered. Today they re-decide at every junction using the current weights. That is
more realistic for "information at the entrance" and less so for "announcement heard by everyone".
Mid-route changes of destination would need an explicit relabelling rule.

## Open questions
1. **FIFO vs fully mixed.** Fully mixed lets a sudden change in mix at the entrance appear downstream
   slightly early. Exact FIFO needs cumulative inflow curves per OD pair on each link.
2. **Destination-only labels** (K = destinations instead of OD pairs) are enough for routing if route
   choice doesn't depend on the origin. That makes it cheaper on large OD matrices, but loses the OD
   matrix of arrivals.
3. **Outputs.** `OutputHandler` and the dashboard don't save or show the labels yet.
4. **Controllers / `assign_flows_type: optimal`.** Edge flows are recorded for `optimal` too, but this
   wasn't tested with controller networks.
5. **Default.** Should tracking become the default (and the weight estimate the fallback) once 1–4 are settled?
