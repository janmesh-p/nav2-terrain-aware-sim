## Navigation outcome

| metric | v3d_baseline | v3d_terrain |
|---|---|---|
| runs | 3 | 3 |
| full route succeeded | 2/3 | 1/3 |
| bump_entry | 3/3 | 3/3 |
| bump_exit | 3/3 | 1/3 |
| rough_entry | 3/3 | 3/3 |
| rough_exit | 3/3 | 3/3 |
| ramp_entry | 3/3 | 3/3 |
| ramp_exit | 2/3 | 2/3 |
| median recoveries per run | 20 | 3 |
| Nav2 false successes (total) | 0 | 2 |
| median route time (sim s) | 318 | 113 |

## Localization (median across runs)

| metric | v3d_baseline | v3d_terrain |
|---|---|---|
| runs with localization data | 3 | 3 |
| AMCL error, median | 15.1 cm | 14.7 cm |
| AMCL error, p95 | 52.9 cm | 37.4 cm |
| AMCL error on rough patch | 24.9 cm | 10.1 cm |
| time on rough patch | 41.6 s | 24.5 s |
| tilt p95 | 14.0 deg | 7.1 deg |
| runs with AMCL error > 200 cm | 0/3 | 0/3 |

## Tilt episodes vs flat floor (pooled per condition)

| condition | episodes | controls | median change, tilt | median change, flat | difference [95% CI] | p |
|---|---|---|---|---|---|---|
| v3d_baseline | 27 | 522 | +0.1 cm | +0.0 cm | +0.1 cm [-4.8, +8.5] | 0.015 |
| v3d_terrain | 9 | 180 | -7.7 cm | +0.3 cm | -7.9 cm [-29.5, +24.8] | 0.993 |

Runs: v3d_baseline: v3d_baseline_run1, v3d_baseline_run2, v3d_baseline_run3; v3d_terrain: v3d_terrain_run1, v3d_terrain_run2, v3d_terrain_run3
