# Walk bench: DCM and ALIP Footstep

Closed-loop results from `python -m loka.walk_bench` on 29 September 2026, one run of each stack at the defaults in `loka/control/`. `legacy_dcm` places footholds with the DCM capture point. `alip_footstep` uses the same convex MPC and whole-body QP, and corrects those footholds with the ALIP stepping law.

A scenario passes when the robot does not fall and, where the scenario asks, tracks heading inside the turn-rate budget, holds at least 60% of the commanded speed over the last 3 s, stands by the deadline, stays inside the lateral bound, and arrives at a goal standing.

## Summary

| Group | Scenarios | DCM | ALIP Footstep |
|---|---:|---:|---:|
| Turning | 7 | 4 | 4 |
| Gait changes | 7 | 5 | 6 |
| Straight walk | 1 | 1 | 1 |
| Disturbance and drift | 4 | 2 | 4 |
| Walk to a goal | 4 | 1 | 0 |
| Pushes | 73 | 66 | 49 |
| **All** | **96** | **79** | **64** |

Both stacks pass 61 scenarios. DCM alone passes 18. ALIP Footstep alone passes 3. Both fail 14.

ALIP Footstep is the one that holds the 30 s straight walk inside 0.15 m of lateral drift (`S1`), and the one that leaves a treadmill start into a walk (`G5`, `B1`). DCM is the one that survives the larger pushes, especially 10–12 N·s, and the goal walk with a sideways shove (`W4`).

## Where the stacks differ

| Scenario | DCM | ALIP Footstep | What failed |
|---|---|---|---|
| `G5` | fail | pass | DCM: speed below 60% of command |
| `B1` | fail | pass | DCM: speed below 60% of command |
| `S1` | fail | pass | DCM: lateral drift 0.258 m (limit 0.15 m) |
| `W4` | pass | fail | ALIP: still walking at the goal |
| `P1_fwd_ds_12` | pass | fail | ALIP: fell at 5.9 s |
| `P1_fwd_mid_10` | pass | fail | ALIP: fell at 7.0 s |
| `P1_fwd_mid_12` | pass | fail | ALIP: fell at 6.0 s |
| `P1_fwd_late_12` | pass | fail | ALIP: fell at 6.2 s |
| `P1_back_ds_8` | pass | fail | ALIP: fell at 6.2 s; speed below 60% of command |
| `P1_left_ds_10` | pass | fail | ALIP: fell at 6.9 s; speed below 60% of command |
| `P1_left_ds_12` | pass | fail | ALIP: fell at 5.7 s; speed below 60% of command |
| `P1_left_mid_10` | pass | fail | ALIP: fell at 7.5 s; speed below 60% of command |
| `P1_left_mid_12` | pass | fail | ALIP: fell at 5.8 s; speed below 60% of command |
| `P1_left_late_10` | pass | fail | ALIP: fell at 6.6 s; speed below 60% of command |
| `P1_left_late_12` | pass | fail | ALIP: fell at 5.8 s; speed below 60% of command |
| `P1_right_ds_10` | pass | fail | ALIP: fell at 7.8 s; speed below 60% of command |
| `P1_right_ds_12` | pass | fail | ALIP: fell at 5.7 s; speed below 60% of command |
| `P1_right_mid_10` | pass | fail | ALIP: fell at 7.6 s; speed below 60% of command |
| `P1_right_mid_12` | pass | fail | ALIP: fell at 6.5 s; speed below 60% of command |
| `P1_right_late_10` | pass | fail | ALIP: fell at 6.6 s; speed below 60% of command |
| `P1_right_late_12` | pass | fail | ALIP: fell at 6.2 s; speed below 60% of command |

## Both fail

| Scenario | DCM | ALIP Footstep |
|---|---|---|
| `T4` | speed below 60% of command | speed below 60% of command |
| `T6` | heading error | fell at 9.8 s; heading error; speed below 60% of command |
| `T7` | heading late | heading late |
| `G6` | speed below 60% of command | speed below 60% of command |
| `W1` | still walking at the goal | still walking at the goal |
| `W2` | still walking at the goal | still walking at the goal |
| `W3` | still walking at the goal | still walking at the goal |
| `P1_back_ds_10` | speed below 60% of command | fell at 5.7 s; speed below 60% of command |
| `P1_back_ds_12` | speed below 60% of command | fell at 5.5 s; speed below 60% of command |
| `P1_back_mid_10` | speed below 60% of command | fell at 6.9 s; speed below 60% of command |
| `P1_back_mid_12` | speed below 60% of command | fell at 6.1 s; speed below 60% of command |
| `P1_back_late_8` | speed below 60% of command | fell at 6.5 s; speed below 60% of command |
| `P1_back_late_10` | speed below 60% of command | fell at 6.1 s; speed below 60% of command |
| `P1_back_late_12` | speed below 60% of command | fell at 5.9 s; speed below 60% of command |

## Every scenario

| Scenario | Group | DCM | ALIP Footstep |
|---|---|---|---|
| `T1` | Turning | pass | pass |
| `T2` | Turning | pass | pass |
| `T3` | Turning | pass | pass |
| `T4` | Turning | fail | fail |
| `T5` | Turning | pass | pass |
| `T6` | Turning | fail | fail |
| `T7` | Turning | fail | fail |
| `G1` | Gait changes | pass | pass |
| `G2` | Gait changes | pass | pass |
| `G3` | Gait changes | pass | pass |
| `G4` | Gait changes | pass | pass |
| `G5` | Gait changes | fail | pass |
| `G5b` | Gait changes | pass | pass |
| `G6` | Gait changes | fail | fail |
| `S0` | Straight walk | pass | pass |
| `B1` | Disturbance and drift | fail | pass |
| `B2` | Disturbance and drift | pass | pass |
| `N1` | Disturbance and drift | pass | pass |
| `S1` | Disturbance and drift | fail | pass |
| `W1` | Walk to a goal | fail | fail |
| `W2` | Walk to a goal | fail | fail |
| `W3` | Walk to a goal | fail | fail |
| `W4` | Walk to a goal | pass | fail |
| `P1_fwd_ds_2` | Pushes | pass | pass |
| `P1_fwd_ds_4` | Pushes | pass | pass |
| `P1_fwd_ds_6` | Pushes | pass | pass |
| `P1_fwd_ds_8` | Pushes | pass | pass |
| `P1_fwd_ds_10` | Pushes | pass | pass |
| `P1_fwd_ds_12` | Pushes | pass | fail |
| `P1_fwd_mid_2` | Pushes | pass | pass |
| `P1_fwd_mid_4` | Pushes | pass | pass |
| `P1_fwd_mid_6` | Pushes | pass | pass |
| `P1_fwd_mid_8` | Pushes | pass | pass |
| `P1_fwd_mid_10` | Pushes | pass | fail |
| `P1_fwd_mid_12` | Pushes | pass | fail |
| `P1_fwd_late_2` | Pushes | pass | pass |
| `P1_fwd_late_4` | Pushes | pass | pass |
| `P1_fwd_late_6` | Pushes | pass | pass |
| `P1_fwd_late_8` | Pushes | pass | pass |
| `P1_fwd_late_10` | Pushes | pass | pass |
| `P1_fwd_late_12` | Pushes | pass | fail |
| `P1_back_ds_2` | Pushes | pass | pass |
| `P1_back_ds_4` | Pushes | pass | pass |
| `P1_back_ds_6` | Pushes | pass | pass |
| `P1_back_ds_8` | Pushes | pass | fail |
| `P1_back_ds_10` | Pushes | fail | fail |
| `P1_back_ds_12` | Pushes | fail | fail |
| `P1_back_mid_2` | Pushes | pass | pass |
| `P1_back_mid_4` | Pushes | pass | pass |
| `P1_back_mid_6` | Pushes | pass | pass |
| `P1_back_mid_8` | Pushes | pass | pass |
| `P1_back_mid_10` | Pushes | fail | fail |
| `P1_back_mid_12` | Pushes | fail | fail |
| `P1_back_late_2` | Pushes | pass | pass |
| `P1_back_late_4` | Pushes | pass | pass |
| `P1_back_late_6` | Pushes | pass | pass |
| `P1_back_late_8` | Pushes | fail | fail |
| `P1_back_late_10` | Pushes | fail | fail |
| `P1_back_late_12` | Pushes | fail | fail |
| `P1_left_ds_2` | Pushes | pass | pass |
| `P1_left_ds_4` | Pushes | pass | pass |
| `P1_left_ds_6` | Pushes | pass | pass |
| `P1_left_ds_8` | Pushes | pass | pass |
| `P1_left_ds_10` | Pushes | pass | fail |
| `P1_left_ds_12` | Pushes | pass | fail |
| `P1_left_mid_2` | Pushes | pass | pass |
| `P1_left_mid_4` | Pushes | pass | pass |
| `P1_left_mid_6` | Pushes | pass | pass |
| `P1_left_mid_8` | Pushes | pass | pass |
| `P1_left_mid_10` | Pushes | pass | fail |
| `P1_left_mid_12` | Pushes | pass | fail |
| `P1_left_late_2` | Pushes | pass | pass |
| `P1_left_late_4` | Pushes | pass | pass |
| `P1_left_late_6` | Pushes | pass | pass |
| `P1_left_late_8` | Pushes | pass | pass |
| `P1_left_late_10` | Pushes | pass | fail |
| `P1_left_late_12` | Pushes | pass | fail |
| `P1_right_ds_2` | Pushes | pass | pass |
| `P1_right_ds_4` | Pushes | pass | pass |
| `P1_right_ds_6` | Pushes | pass | pass |
| `P1_right_ds_8` | Pushes | pass | pass |
| `P1_right_ds_10` | Pushes | pass | fail |
| `P1_right_ds_12` | Pushes | pass | fail |
| `P1_right_mid_2` | Pushes | pass | pass |
| `P1_right_mid_4` | Pushes | pass | pass |
| `P1_right_mid_6` | Pushes | pass | pass |
| `P1_right_mid_8` | Pushes | pass | pass |
| `P1_right_mid_10` | Pushes | pass | fail |
| `P1_right_mid_12` | Pushes | pass | fail |
| `P1_right_late_2` | Pushes | pass | pass |
| `P1_right_late_4` | Pushes | pass | pass |
| `P1_right_late_6` | Pushes | pass | pass |
| `P1_right_late_8` | Pushes | pass | pass |
| `P1_right_late_10` | Pushes | pass | fail |
| `P1_right_late_12` | Pushes | pass | fail |
| `P2` | Pushes | pass | pass |

JSON from this run is in `logs/bench/legacy_dcm.json` and `logs/bench/alip_footstep.json` (local, not committed).
