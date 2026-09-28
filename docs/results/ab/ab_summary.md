## Navigation outcome

| metric | baseline | terrain |
|---|---|---|
| runs | 3 | 6 |
| full route succeeded | 2/3 | 0/6 |
| bump_entry | 3/3 | 6/6 |
| bump_exit | 3/3 | 6/6 |
| rough_entry | 3/3 | 6/6 |
| rough_exit | 2/3 | 4/6 |
| ramp_entry | 2/3 | 1/6 |
| ramp_exit | 2/3 | 0/6 |
| median recoveries per run | 28 | 50 |

## Localization (median across runs)

| metric | baseline | terrain |
|---|---|---|
| runs with localization data | 3 | 5 |
| AMCL error, median | 44.9 cm | 49.0 cm |
| AMCL error, p95 | 160.0 cm | 75.9 cm |
| AMCL error on rough patch | 78.5 cm | 53.3 cm |
| time on rough patch | 60.3 s | 50.4 s |
| tilt p95 | 10.2 deg | 7.1 deg |
| runs with AMCL error > 200 cm | 2/3 | 2/5 |

## Tilt episodes vs flat floor (pooled per condition)

| condition | episodes | controls | median change, tilt | median change, flat | difference [95% CI] | p |
|---|---|---|---|---|---|---|
| baseline | 18 | 296 | +8.9 cm | -0.0 cm | +8.9 cm [+0.2, +25.7] | 0.000 |
| terrain | 17 | 320 | +10.8 cm | +0.0 cm | +10.8 cm [-8.9, +36.9] | 0.002 |

Runs: baseline: baseline_run1, baseline_run2, baseline_run3; terrain: terrain_run2, terrain_run4, terrain_run5, terrain_run6, terrain_run7, terrain_run8
